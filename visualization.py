from utils import *
from models import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Indian', help='dataset to use')
parser.add_argument('--flag', choices=['test', 'train', 'finetune'], default='test', help='model for test, train or finetune')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=7, help='number of patches')
parser.add_argument('--band_patches', type=int, default=1, help='number of related band')
parser.add_argument('--classes', type=int, default=1000, help='classes number')
parser.add_argument('--epoches', type=int, default=200, help='epoch number')
parser.add_argument('--learning_rate', type=float, default=5e-4, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--output_dir', default='./logs/',
                    help='path where to save, empty for no saving')
parser.add_argument('--save_ckpt_freq', default=200, type=int,
                    help='Frequency to save a checkpoint of the model')
parser.add_argument('--mask_ratio', default=0.75,type=float,
                    help='ratio of the visual tokens/patches need be masked')
parser.add_argument('--mask_method', choices=['random', 'high', 'low', 'high_random', 'gradcam'], default=None)
parser.add_argument('--mask_clf', choices=['random', 'high', 'low', 'high_random', 'gradcam'], default=None)
parser.add_argument('--model_path', default=None,
                    help='Location of save model')
# parser.add_argument('--model_path', default='./model_path/checkpoint-XXX.pth',
#                     help='Location of saved model')
parser.add_argument('--trained_model', default='./logs/final/checkpoint-Indian_clf_scratch_full_SF_7_1.pth',#old --finetune
                    help='location of trained model for fine-tune or last checkpoint')
# parser.add_argument('--trained_model', default=None,#old --finetune
#                     help='location of trained model for fine-tune or last checkpoint')
parser.add_argument('--model_key', default='model|module', type=str)
parser.add_argument('--device', default="0", type=str)
parser.add_argument('--model_prefix', default='', type=str)
parser.add_argument('--init_scale', default=0.001, type=float)
parser.add_argument('--use_mean_pooling', action='store_true')
parser.set_defaults(use_mean_pooling=True)
parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
parser.add_argument('--align', default=None, choices=['align', 'align_dec'])
parser.add_argument('--freeze_backbone', action='store_true') #default is False
parser.add_argument('--save_gradcam',action='store_true')
parser.add_argument('--use_class_attn', action='store_true')
parser.add_argument('--use_fulldata', action='store_true')
parser.add_argument('--spatial_attn', action='store_true', help='whether use spatial attention in transformer encoder blocks')
parser.add_argument('--use_sar', action='store_true', help='whether use spatial attention-based regularization')
parser.add_argument('--use_se', action='store_true', help='whether use squeeze and excitation')

args = parser.parse_args() # for python file

os.environ['CUDA_VISIBLE_DEVICES']=args.device
cudnn.benchmark = True
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

if args.dataset == 'Indian':
    data = loadmat('./data/IndianPine.mat')
elif args.dataset == 'Pavia':
    data = loadmat('./data/Pavia.mat')
elif args.dataset == 'Houston':
    data = loadmat('./data/Houston.mat')
else:
    raise ValueError("Unkknow dataset")
color_mat = loadmat('./data/AVIRIS_colormap.mat')
TR = data['TR']
TE = data['TE']
input = data['input']
# input, _ = applyPCA(input, input.shape[2]//10)
# input, _ = applyPCA(input, 50)
label = TR + TE
num_classes = np.max(TR)
args.classes = num_classes
color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]]
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:,:,i])
    input_min = np.min(input[:,:,i])
    input_normalize[:,:,i] = (input[:,:,i]-input_min)/(input_max-input_min)

height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
args.number_patches = band
#-------------------------------------------------------------------------------
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = choose_train_and_test_point(TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test, total_pos_true, patch=args.patches, band_patch=args.band_patches, flag=args.flag)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)

masked_positional_generator = RandomMaskingGenerator(args.number_patches, args.mask_ratio)
# if args.mask_clf == 'gradcam':
#     sorted_bands = np.load('./as_sorted_bands_IP_MAEST_align.csv.npy')  # asending
#     # np.random.shuffle(sorted_bands)
#     vis_num = int(args.number_patches * (1 - args.mask_ratio))

if args.mask_method == 'gradcam':
    if args.dataset == 'Indian':
        # sorted_bands = np.load('./as_sorted_bands_IP_MAEST_align.csv.npy')  # asending
        # gradcam_values = np.load('./gradcam_maps_maest.csv.npy') #gradcam_maps_maest.csv.npy
        gradcam_values = np.load('./all_correct_maps_mean_1_1.npy') # gradcam of ViT with 1_1 patch and near band size on test set
        # gradcam_values = np.load('./gradcam_maps_vit_fulldata_Indian.npy')  # gradcam from the model trained on the whole dataset
    elif args.dataset == 'Houston':
        gradcam_values = np.load('./gradcam_maps_vit_Houston_1_1.npy')  # gradcam_maps_maest.csv.npy #gradcam_maps_maest_Houston.npy
    elif args.dataset == 'Pavia':
        gradcam_values = np.load('./gradcam_maps_vit_Pavia_1_1.npy')  #
    else:
        print('No gradcam values fot the dataset')
        raise 1
    mask_num = int(args.number_patches * args.mask_ratio)

if args.use_fulldata:
    x_true_band = x_true_band[y_true!=0]
    y_true = y_true[y_true!=0]
    y_true -= 1
    # x_train, x_test, y_train, y_test = train_test_split(x_true_band,
    #                                                     y_true,
    #                                                     test_size=0.4,
    #                                                     random_state=0,
    #                                                     stratify=y_true)
    ## using the same split ratio as the best results of JigsawHSI shown in paper_with_code
    x_train, x_test_, y_train, y_test_ = train_test_split(x_true_band,
                                                        y_true,
                                                          test_size=0.7, random_state=345,
                                                        stratify=y_true)
    x_test, _, y_test, _ = train_test_split(x_test_,
                                                        y_test_,
                                                        test_size=0.7, random_state=None,
                                                        stratify=y_test_)

    # (3074, 27, 27, 200, 1)(5023, 27, 27, 200, 1)(3074, )(5023, )
    # (2152, 27, 27, 200, 1)(2152, )
    print('train_test split with sizes of x_train: {}, y_train: {}, x_test: {}, y_test: {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))
    x_train = torch.from_numpy(x_train.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    x_test = torch.from_numpy(x_test.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    bool_masked_pos_t = torch.zeros(x_train.shape[0], args.number_patches)
    for b in range(x_train.shape[0]):
        bool_masked_pos_t[b, :] = torch.from_numpy(masked_positional_generator())
    bool_masked_pos_t = bool_masked_pos_t > 0
    Label_train = Data.TensorDataset(x_train, bool_masked_pos_t)
    Label_tune = Data.TensorDataset(x_train, bool_masked_pos_t, y_train)
    label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
    label_tune_loader = Data.DataLoader(Label_tune, batch_size=args.batch_size, shuffle=True)
    bool_masked_pos_tt = torch.zeros(x_test.shape[0], args.number_patches)
    for b in range(x_test.shape[0]):
        bool_masked_pos_tt[b, :] = torch.from_numpy(masked_positional_generator())
    bool_masked_pos_tt = bool_masked_pos_tt > 0
    Label_test = Data.TensorDataset(x_test, bool_masked_pos_tt, y_test)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)

else:
    if (args.flag == 'train') or (args.flag == 'finetune'):
        x_train=torch.from_numpy(x_train_band.transpose(0,2,1)).type(torch.FloatTensor)
        y_train=torch.from_numpy(y_train).type(torch.LongTensor)

        bool_masked_pos_t = torch.zeros(x_train_band.shape[0], args.number_patches)
        # x_tune = x_train
        for b in range(x_train_band.shape[0]):
            # if args.mask_clf == 'gradcam':
            #     mask_pos = sorted_bands[y_train[b]][:-vis_num] #mask out the least important bands
            #     # mask_pos = sorted_bands[y_train[b]][vis_num:] #mask out the most important bands
            #     # x_tune[b, mask_pos, :] = 0
            #     bool_masked_pos_t[b, mask_pos] = 1
            #     print('Mask the least {} important bands'.format(args.number_patches - vis_num))
            # elif args.mask_clf == 'random':
            #     mask_pos = torch.from_numpy(masked_positional_generator())
            #     bool_masked_pos_t[b, :] = mask_pos
            #     # x_tune[b, torch.argwhere(mask_pos == 1).squeeze(), :] = 0

            if args.mask_method == 'gradcam':
                # sample_weights = 1- gradcam_values[y_train[b]] # to mask the least important
                sample_weights = gradcam_values[y_train[b]]
                mask_pos = list(torch.utils.data.WeightedRandomSampler(sample_weights, mask_num, replacement=False))
                bool_masked_pos_t[b, mask_pos] = 1
            else:
                bool_masked_pos_t[b, :] = torch.from_numpy(masked_positional_generator())
        bool_masked_pos_t = bool_masked_pos_t > 0
        Label_train=Data.TensorDataset(x_train,bool_masked_pos_t)
        Label_tune =Data.TensorDataset(x_train,bool_masked_pos_t,y_train)
        label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
        label_tune_loader=Data.DataLoader(Label_tune,batch_size=args.batch_size,shuffle=True)

    x_test=torch.from_numpy(x_test_band.transpose(0,2,1)).type(torch.FloatTensor)
    y_test=torch.from_numpy(y_test).type(torch.LongTensor)
    bool_masked_pos_tt = torch.zeros(x_test_band.shape[0], args.number_patches)
    for b in range(x_test_band.shape[0]):
        # if args.mask_clf == 'gradcam':
        #     mask_pos_tt = sorted_bands[y_test[b]][:-vis_num]
        #     # x_test[b, mask_pos_tt, :] = 0
        #     bool_masked_pos_tt[b, mask_pos_tt] = 1
        # elif args.mask_clf == 'random':
        #     mask_pos_tt = torch.from_numpy(masked_positional_generator())
        #     bool_masked_pos_tt[b, :] = mask_pos_tt
        #     # x_test[b, torch.argwhere(mask_pos_tt == 1).squeeze(), :] = 0
        # else:
        bool_masked_pos_tt[b,:] = torch.from_numpy(masked_positional_generator())
    bool_masked_pos_tt = bool_masked_pos_tt > 0
    Label_test=Data.TensorDataset(x_test,bool_masked_pos_tt,y_test)
    # label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
    label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=False) # not shuffle to keep the same order as original y_test

print(args)
size_patches = args.band_patches * args.patches ** 2
#-------------------------------------------------------------------------------

def valid_epoch_test(model, valid_loader,criterion, align=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])

    # for batch_idx, (batch_data, batch_mask, batch_target) in enumerate(valid_loader):
    for batch_idx, (batch_data, batch_mask, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_mask = batch_mask.cuda()
        out = model(batch_data, batch_mask)
        if align is not None:
            assert len(out) == 2
            batch_pred, loss_align = out
        else:
            batch_pred = out
        loss = criterion(batch_pred, batch_target)

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return tar, pre

def test_model(model, align=None):
    model.eval()
    criterion = nn.CrossEntropyLoss().cuda()
    tar_v, pre_v = valid_epoch_test(model, label_test_loader, criterion, align=align)
    # label_true_loader, _, _ = load_true(y_true)
    # tar_v, pre_v = valid_epoch(model, label_true_loader, criterion, align=align)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print(AA2)
    return AA2, pre_v, tar_v

def load_from_files(model_path, spatial_attn, align):
    model = VisionTransformerEncoder(
        image_size = args.patches,
        near_band = args.band_patches,
        num_patches = args.number_patches,
        num_classes = args.classes,
        dim = 64,
        depth = 5, #5
        heads = 4, #4
        mlp_dim = 8, #args.classes, #8
        dropout = 0.1,  #0.1
        emb_dropout = 0.1, #0.1
        mode = args.mode,
        mask_clf=args.mask_clf,
        mask_ratio=args.mask_ratio,
        mask_method=args.mask_method,
        use_class_attn=args.use_class_attn,
        align_loss=align,
        spatial_attn=spatial_attn,
        use_sar=args.use_sar,
        use_se=args.use_se,
    ).cuda()

    checkpoint = torch.load(model_path, map_location='cpu')
    print("Load ckpt from %s" % args.trained_model)
    checkpoint_model = None

    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    if checkpoint_model is None:
        checkpoint_model = checkpoint
    state_dict = model.state_dict()

    for k in ['mlp_head.weight', 'mlp_head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('backbone.'):
            new_dict[key[9:]] = checkpoint_model[key]
        elif key.startswith('encoder.'):
        # elif key == 'encoder.patch_to_embedding.weight':
        #     new_dict['patch_to_embedding.weight'] = checkpoint_model[key]
        # elif key == 'encoder.patch_to_embedding.bias':
        #     new_dict['patch_to_embedding.bias'] = checkpoint_model[key]
        # elif key == 'encoder.pos_embedding':
        #     new_dict['pos_embedding'] = checkpoint_model[key]
        # elif key == 'encoder.cls_token':
        #     new_dict['cls_token'] = checkpoint_model[key]
        # elif key.startswith('decoder.'):
            new_dict[key[8:]] = checkpoint_model[key]
        else:
            new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    if 'pos_embedding' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embedding']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.num_patches
        num_extra_tokens = model.pos_embedding.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)

        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embedding'] = new_pos_embed
    load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
    return model

def plot_classification_map(ground_truth, preds_sf, preds_b1):
    fig, ax =plt.subplots(1,3, figsize=(16,16))
    box = (67, 82)
    color = 'white'
    # color = [0.8, 0.8, 0.8]
    masked_label = np.ma.masked_where(ground_truth==0, ground_truth)
    cmap_label = matplotlib.cm.get_cmap("nipy_spectral").copy()
    cmap_label.set_bad(color=color)
    rect_label = patches.Rectangle(box, 9, 9, linewidth=1,
                             edgecolor='r', facecolor="none")
    ax[0].imshow(masked_label, cmap=cmap_label)
    ax[0].add_patch(rect_label)
    ax[0].grid(False)
    ax[0].axis(False)
    ax[0].set_title('Ground truth')

    masked_preds_sf = np.ma.masked_where(preds_sf==0, preds_sf)
    cmap_sf = matplotlib.cm.get_cmap("nipy_spectral").copy()
    cmap_sf.set_bad(color=color)
    rect = patches.Rectangle(box, 9, 9, linewidth=1,
                             edgecolor='r', facecolor="none")
    ax[1].imshow(masked_preds_sf, cmap=cmap_sf)
    ax[1].add_patch(rect)
    ax[1].set_title('Predicted by SpectralFormer')
    ax[1].grid(False)
    ax[1].axis(False)

    masked_preds_b1 = np.ma.masked_where(preds_b1 == 0, preds_b1)
    cmap_b1 = matplotlib.cm.get_cmap("nipy_spectral").copy()
    cmap_b1.set_bad(color='white')
    rect2 = patches.Rectangle(box, 9, 9, linewidth=1,
                              edgecolor='r', facecolor="none")
    ax[2].imshow(masked_preds_b1, cmap=cmap_b1)
    ax[2].add_patch(rect2)
    ax[2].set_title('Predicted by Ours')
    ax[2].grid(False)
    ax[2].axis(False)
    plt.savefig('./logs/predicts2.pdf')
    plt.show()

def get_preds_map(preds_sf, preds_b1):
    TE_prd_sf = np.zeros((width,height))
    TE_prd_b1 = np.zeros((width,height))
    for i, pos in enumerate(total_pos_test):
        TE_prd_sf[pos[0], pos[1]] = preds_sf[i] +1
        TE_prd_b1[pos[0], pos[1]] = preds_b1[i] +1
    return TE_prd_sf, TE_prd_b1

model_sf = load_from_files('./logs/final/checkpoint-Indian_clf_scratch_full_SF_7_1.pth', spatial_attn=False, align=None)
acc_class_sf, preds_sf, tar_v = test_model(model_sf, align=None)

model_b1 = load_from_files('./logs/final/checkpoint-Indian_clf_scratch_full_B1_7_1.pth', spatial_attn=True, align='align')
acc_class_b1, preds_b1, tar_v_b1 = test_model(model_b1, align='align')

## plot the truth and predicted map
groud_truth = TR + TE
preds_map_sf, preds_map_b1 = get_preds_map(preds_sf, preds_b1)
preds_map_sf = TR + preds_map_sf
preds_map_b1 = TR + preds_map_b1
plot_classification_map(groud_truth, preds_map_sf, preds_map_b1)

## plot the reflectance of two pixels
## class order from the website
# classes = np.array(['Alfalfa', 'Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 'Grass-pasture-mowed', 'Hay-windrowed',
#                                    'Oats', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean', 'Wheat', 'Woods', 'Buildings-Grass-Trees-drives', 'Stone-Steel-Towers'])
classes = np.array(['Corn-notill', 'Corn-mintill', 'Corn', 'Grass-pasture', 'Grass-trees', 'Hay-windrowed', 'Soybean-notill', 'Soybean-mintill', 'Soybean-clean',
                    'Wheat', 'Woods', 'Buildings-Grass-Trees-drives', 'Stone-Steel-Towers', 'Alfalfa', 'Grass-pasture-mowed', 'Oats'])
print('True label of pixel (82,73): ', classes[label[82,73].astype(int)-1])
print('True label of pixel (82,70): ', classes[label[82,70].astype(int)-1])
print('Predicted label of pixel (82,70) by SF: ', classes[preds_map_sf[82,70].astype(int)-1])
print('Predicted label of pixel (82,70) by ours: ', classes[preds_map_b1[82,70].astype(int)-1])
plt.plot(input_normalize[82,73], label='Corn Notill')
plt.plot(input_normalize[82,67], label='Soybean Mintill')
plt.legend()
plt.grid(False)
plt.xlabel('spectral band')
plt.ylabel('reflectance')
plt.savefig('./logs/reflectance.pdf')
plt.show()