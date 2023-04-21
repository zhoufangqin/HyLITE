import torch
import argparse
import torch.nn as nn
import torch.utils.data as Data
import torch.backends.cudnn as cudnn
import numpy as np
import time
import os
import json
import wandb

from utils import *
from models import VisionTransformerEncoder
from scipy.io import loadmat
from sklearn.metrics import confusion_matrix
from spectral import *
from collections import OrderedDict
from gradCAM import GradCAM
from sklearn.model_selection import train_test_split
import similaritymeasures
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'Pavia', 'Houston'], default='Indian', help='dataset to use')
parser.add_argument('--flag', choices=['test', 'train'], default='train',
                    help='model for test or train')
parser.add_argument('--mode', choices=['ViT', 'CAF'], default='CAF', help='mode choice')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--batch_size', type=int, default=32, help='number of batch size')
parser.add_argument('--test_freq', type=int, default=5, help='number of evaluation')
parser.add_argument('--patches', type=int, default=1, help='number of patches')
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
parser.add_argument('--mask_ratio', default=0.75, type=float,
                    help='ratio of the visual tokens/patches need be masked')
parser.add_argument('--mask_method', choices=['random', 'high', 'low', 'high_random', 'gradcam'], default=None)
parser.add_argument('--mask_clf', choices=['random', 'high', 'low', 'high_random', 'gradcam'], default=None)
parser.add_argument('--model_path', default=None,
                    help='Location of saved model')
# parser.add_argument('--model_path', default='./model_path/checkpoint-XXX.pth',
#                     help='Location of saved model')
# parser.add_argument('--trained_model', default='./trained_model_path/checkpoint-XXX.pth',#old --finetune
#                     help='location of trained model for fine-tune or last checkpoint')
parser.add_argument('--trained_model', default=None,  # old --finetune
                    help='location of trained model for fine-tune or last checkpoint')
parser.add_argument('--model_key', default='model|module', type=str)
parser.add_argument('--device', default="0", type=str)
parser.add_argument('--model_prefix', default='', type=str)
parser.add_argument('--init_scale', default=0.001, type=float)
parser.add_argument('--use_mean_pooling', action='store_true')
parser.set_defaults(use_mean_pooling=True)
parser.add_argument('--use_cls', action='store_false', dest='use_mean_pooling')
parser.add_argument('--align', default=None, choices=['align', 'align_dec'])
parser.add_argument('--freeze_backbone', action='store_true')  # default is False
parser.add_argument('--save_gradcam', action='store_true')
parser.add_argument('--use_class_attn', action='store_true')
parser.add_argument('--use_fulldata', action='store_true')
parser.add_argument('--spatial_attn', action='store_true',
                    help='whether use spatial attention in transformer encoder blocks')
parser.add_argument('--use_sar', action='store_true', help='whether use spatial attention-based regularization')
parser.add_argument('--use_se', action='store_true', help='whether use squeeze and excitation')
parser.add_argument('--train_size', default=1.0, type=float)
parser.add_argument('--weights', default=1.0, type=float, help='weights of alignment loss')

args = parser.parse_args()  # for python file
# args = parser.parse_args([]) # for notebook

os.environ['CUDA_VISIBLE_DEVICES'] = args.device
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
# input, _ = applyPCA(input, 10)
label = TR + TE
num_classes = np.max(TR)
args.classes = num_classes
color_mat_list = list(color_mat)
color_matrix = color_mat[color_mat_list[3]]
input_normalize = np.zeros(input.shape)
for i in range(input.shape[2]):
    input_max = np.max(input[:, :, i])
    input_min = np.min(input[:, :, i])
    input_normalize[:, :, i] = (input[:, :, i] - input_min) / (input_max - input_min)

height, width, band = input.shape
print("height={0},width={1},band={2}".format(height, width, band))
args.number_patches = band
# -------------------------------------------------------------------------------
total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true = choose_train_and_test_point(
    TR, TE, label, num_classes)
mirror_image = mirror_hsi(height, width, band, input_normalize, patch=args.patches)
x_train_band, x_test_band, x_true_band = train_and_test_data(mirror_image, band, total_pos_train, total_pos_test,
                                                             total_pos_true, patch=args.patches,
                                                             band_patch=args.band_patches, flag=args.flag)
y_train, y_test, y_true = train_and_test_label(number_train, number_test, number_true, num_classes)
# y_train = TR[TR!=0].astype(int) -1
# y_test = TE[TE!=0].astype(int) - 1
# y_true = label[label!=0].astype(int) - 1

masked_positional_generator = RandomMaskingGenerator(args.number_patches, args.mask_ratio)
# if args.mask_clf == 'gradcam':
#     sorted_bands = np.load('./as_sorted_bands_IP_MAEST_align.csv.npy')  # asending
#     # np.random.shuffle(sorted_bands)
#     vis_num = int(args.number_patches * (1 - args.mask_ratio))

if args.mask_method == 'gradcam':
    if args.dataset == 'Indian':
        # sorted_bands = np.load('./as_sorted_bands_IP_MAEST_align.csv.npy')  # asending
        # gradcam_values = np.load('./gradcam_maps_maest.csv.npy') #gradcam_maps_maest.csv.npy
        gradcam_values = np.load(
            './all_correct_maps_mean_1_1.npy')  # gradcam of ViT with 1_1 patch and near band size on test set
        # gradcam_values = np.load('./gradcam_maps_vit_fulldata_Indian.npy')  # gradcam from the model trained on the whole dataset
    elif args.dataset == 'Houston':
        gradcam_values = np.load(
            './gradcam_maps_vit_Houston_1_1.npy')  # gradcam_maps_maest.csv.npy #gradcam_maps_maest_Houston.npy
    elif args.dataset == 'Pavia':
        gradcam_values = np.load('./gradcam_maps_vit_Pavia_1_1.npy')  #
    else:
        print('No gradcam values fot the dataset')
        raise 1
    mask_num = int(args.number_patches * args.mask_ratio)

if args.use_fulldata:
    x_true_band = x_true_band[y_true != 0]
    y_true = y_true[y_true != 0]
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
    print('train_test split with sizes of x_train: {}, y_train: {}, x_test: {}, y_test: {}'.format(x_train.shape,
                                                                                                   y_train.shape,
                                                                                                   x_test.shape,
                                                                                                   y_test.shape))
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
    if args.flag == 'train':
        if args.train_size < 1:
            x_train_band, _, y_train, _ = train_test_split(x_train_band, y_train, train_size=args.train_size,
                                                           stratify=y_train, random_state=0, shuffle=True)
        x_train = torch.from_numpy(x_train_band.transpose(0, 2, 1)).type(torch.FloatTensor)
        y_train = torch.from_numpy(y_train).type(torch.LongTensor)

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

        # Label_train = Data.TensorDataset(x_train, bool_masked_pos_t)
        Label_tune = Data.TensorDataset(x_train, bool_masked_pos_t, y_train)
        # label_train_loader = Data.DataLoader(Label_train, batch_size=args.batch_size, shuffle=True)
        label_tune_loader = Data.DataLoader(Label_tune, batch_size=args.batch_size, shuffle=True)

    x_test = torch.from_numpy(x_test_band.transpose(0, 2, 1)).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
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
        bool_masked_pos_tt[b, :] = torch.from_numpy(masked_positional_generator())
    bool_masked_pos_tt = bool_masked_pos_tt > 0
    Label_test = Data.TensorDataset(x_test, bool_masked_pos_tt, y_test)
    label_test_loader = Data.DataLoader(Label_test, batch_size=args.batch_size, shuffle=True)

print(args)
size_patches = args.band_patches * args.patches ** 2
# -------------------------------------------------------------------------------
if (args.flag == 'test'):

    model = VisionTransformerEncoder(
        image_size=args.patches,
        near_band=args.band_patches,
        num_patches=args.number_patches,
        num_classes=args.classes,
        dim=64,
        depth=5,
        heads=4,
        mlp_dim=args.classes,
        pool='cls',
        dim_head=16,
        dropout=0.1,
        emb_dropout=0.1,
        mode=args.mode,
        init_scaler=args.init_scale,
        mask_clf=args.mask_clf,
        mask_method=args.mask_method,
        mask_ratio=args.mask_ratio,
        align_loss=args.align)

    checkpoint = torch.load(args.model_path, map_location='cuda')
    print("Load ckpt from %s" % args.model_path)
    checkpoint_model = None

    for model_key in args.model_key.split('|'):
        if model_key in checkpoint:
            checkpoint_model = checkpoint[model_key]
            print("Load state_dict by model_key = %s" % model_key)
            break

    all_keys = list(checkpoint_model.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        new_dict[key] = checkpoint_model[key]
    checkpoint_model = new_dict

    model.load_state_dict(checkpoint_model, strict=False)

elif args.flag == 'train':
    model = VisionTransformerEncoder(
        image_size=args.patches,
        near_band=args.band_patches,
        num_patches=args.number_patches,
        num_classes=args.classes,
        dim=64,
        depth=5,  # 5
        heads=4,  # 4
        mlp_dim=8,  # args.classes, #8
        dropout=0.1,  # 0.1
        emb_dropout=0.1,  # 0.1
        mode=args.mode,
        mask_clf=args.mask_clf,
        mask_ratio=args.mask_ratio,
        mask_method=args.mask_method,
        use_class_attn=args.use_class_attn,
        align_loss=args.align,
        spatial_attn=args.spatial_attn,
        use_sar=args.use_sar,
        use_se=args.use_se,
    )

    if args.trained_model:
        checkpoint = torch.load(args.trained_model, map_location='cpu')
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
# -------------------------------------------------------------------------------

if args.freeze_backbone:
    freezed = 'freezed'
    if args.trained_model is not None:
        for cnt, child in enumerate(model.children()):
            if cnt < 6:  ## layer 6 is last mlp head
                for param in child.parameters():
                    param.requires_grad = False
else:
    freezed = 'notfreezed'

if args.mask_method is not None:
    method = args.mask_method
else:
    method = 'full'

if args.align is None:
    l_a = ''
else:
    l_a = args.align
# if args.trained_model is not None or args.flag=='train':

run = wandb.init(project='LAHIT', config=args,
                 name=args.dataset + '_' + args.flag + '_scratch_' + l_a + '_' + str(args.patches) + '_' + str(
                     args.band_patches))
config = wandb.config

model = model.cuda()
n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Model = %s" % str(model))
print('number of params: {} M'.format(n_parameters / 1e6))

## get the gradcam on the whole dataset with removing background
# x_cam = x_true_band[y_true!=0]
# y_cam = y_true[y_true!=0]
# y_cam -= 1
# x_cam=torch.from_numpy(x_cam.transpose(0,2,1)).type(torch.FloatTensor)
# y_cam=torch.from_numpy(y_cam).type(torch.LongTensor)
# save_gradcams(model, x_cam, y_cam)
# raise 1

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches // 10, gamma=args.gamma, verbose=True)

# if args.flag == 'gracam':
#     print("start get gradcam")
#     model.eval()
#     tar_v, pre_v = get_gradcam(model, label_test_loader, criterion)
#     OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
#     print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
#     print(AA2)
# #-------------------------------------------------------------------------------
if args.flag == 'test':
    print("start test")
    model.eval()
    tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, args)
    OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
    print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print(AA2)
# -------------------------------------------------------------------------------
elif args.flag == 'train':
    print("start training")
    criterion = nn.CrossEntropyLoss().cuda()
    tic = time.time()
    if args.trained_model:
        nam = 'finetune_'
    else:
        nam = 'scratch_'
    OA2s = []
    AA_mean2s = []
    Kappa2s = []
    best_OA2 = 0.0
    best_AA_mean2 = 0.0
    best_Kappa2 = 0.0

    for epoch in range(args.epoches):
        model.train()
        tune_acc, tune_obj, tar_t, pre_t, loss_cls, loss_align, loss_sar = tune_epoch(model, label_tune_loader,
                                                                                      criterion, optimizer, args)
        scheduler.step()
        OA1, AA_mean1, Kappa1, AA1 = output_metric(tar_t, pre_t)
        print("Epoch: {:03d} train_loss: {:.4f} train_acc: {:.4f}"
              .format(epoch + 1, tune_obj, tune_acc))
        wandb.log({
            # 'epoch': epoch,
            'finetune_loss': tune_obj,
            'finetune_loss_cls': loss_cls,
            'finetune_loss_align': loss_align,
            'finetune_loss_sar': loss_sar,
            'finetune_acc': tune_acc
        })

        if args.output_dir:
            if epoch + 1 == args.epoches:
                save_model(
                    args=args, model=model, optimizer=optimizer,
                    epoch=args.dataset + '_clf_' + nam + method + str(args.patches) + '_' + str(args.band_patches))
        if (epoch % args.test_freq == 0) | (epoch == args.epoches - 1):
            model.eval()
            tar_v, pre_v = valid_epoch(model, label_test_loader, criterion, args)
            # tar_v, pre_v = valid_epoch_nomask(model, label_test_loader, criterion, optimizer)
            OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
            best_OA2 = max(best_OA2, OA2.item())
            best_AA_mean2 = max(best_AA_mean2, AA_mean2.item())
            best_Kappa2 = max(best_Kappa2, Kappa2.item())
            wandb.log({
                # 'test_epoch': epoch,
                'test_OA': OA2.item(),
                'test_AA': AA_mean2.item(),
                'test_Kappa': Kappa2.item()
            })

            print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
            print(AA2)
            OA2s.append(OA2)
            AA_mean2s.append(AA_mean2)
            Kappa2s.append(Kappa2)

            if epoch == args.epoches - 1 and args.save_gradcam:
                save_gradcams(model, x_test, y_test)

    toc = time.time()
    print("Running Time: {:.2f}".format(toc - tic))
    print("**************************************************")
    print("Final result:")
    print("Final test OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
    print("Average Test OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(np.mean(np.array(OA2s)),
                                                                        np.mean(np.array(AA_mean2s)),
                                                                        np.mean(np.array(Kappa2s))))
    print("Best Test OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(best_OA2, best_AA_mean2, best_Kappa2))
    print("**************************************************")
