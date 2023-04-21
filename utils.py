import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca
#-------------------------------------------------------------------------------
def choose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    percent = 1

    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        np.random.shuffle(each_class)
        per_data = round(len(each_class)*percent)
        each_class = each_class[:per_data]
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]]
    total_pos_train = total_pos_train.astype(int)

    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]]
    total_pos_test = total_pos_test.astype(int)

    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
def mirror_hsi(height,width,band,input_normalize,patch=5):
    padding=patch//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize

    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]

    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]

    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]

    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("**************************************************")
    print("patch is : {}".format(patch))
    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("**************************************************")
    return mirror_hsi
#-------------------------------------------------------------------------------
def gain_neighborhood_pixel(mirror_image, point, i, patch=5):
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    return temp_image
#-------------------------------------------------------------------------------
def gain_neighborhood_band(x_train, band, band_patch, patch=5):
    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0], patch*patch*band_patch, band),dtype=np.float32)
    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape

    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[:,:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[:,:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[:,0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[:,0:1,:(band-nn+i)]

    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[:,:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[:,:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[:,0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[:,0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
def ungain_neighborhood_band(x_pred, band, band_patch, patch=5, output=True):

    nn = band_patch // 2
    pp = (patch*patch) // 2
    x_pred_sub = np.zeros((x_pred.shape[0], patch*patch, band), dtype=np.float32)
    if output:
        x_pred = np.swapaxes(x_pred, 2, 1)
    x_pred_sub = x_pred[:,nn*patch*patch:(nn+1)*patch*patch, :]
    x_pred_sub = x_pred_sub.reshape(x_pred.shape[0], patch, patch, band)

    return x_pred_sub
#-------------------------------------------------------------------------------
def gain_neighborhood_band_div(x_train, band, band_patch, patch=5, div=0, sub=2):
    nn = band_patch // 2
    pp = (patch*patch) // 2

    x_train_reshape = x_train.reshape(x_train.shape[0], patch*patch, band)
    x_train_band = np.zeros((x_train.shape[0]//sub, patch*patch*band_patch, band),dtype=float)

    x_train_band[:,nn*patch*patch:(nn+1)*patch*patch,:] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,:]

    for i in range(nn):
        if pp > 0:
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,:i+1] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,band-i-1:]
            x_train_band[:,i*patch*patch:(i+1)*patch*patch,i+1:] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,:band-i-1]
        else:
            x_train_band[:,i:(i+1),:(nn-i)] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),0:1,(band-nn+i):]
            x_train_band[:,i:(i+1),(nn-i):] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),0:1,:(band-nn+i)]

    for i in range(nn):
        if pp > 0:
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,:band-i-1] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,i+1:]
            x_train_band[:,(nn+i+1)*patch*patch:(nn+i+2)*patch*patch,band-i-1:] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),:,:i+1]
        else:
            x_train_band[:,(nn+1+i):(nn+2+i),(band-i-1):] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),0:1,:(i+1)]
            x_train_band[:,(nn+1+i):(nn+2+i),:(band-i-1)] = x_train_reshape[0+((div)*(x_train.shape[0]//sub)):(div+1)*(x_train.shape[0]//sub),0:1,(i+1):]
    return x_train_band
#-------------------------------------------------------------------------------
def train_and_test_data(mirror_image, band, train_point, test_point, true_point, patch=5, band_patch=3, flag = 'train'):
    x_train = np.zeros((train_point.shape[0], patch, patch, band), dtype=np.float32)
    x_test = np.zeros((test_point.shape[0], patch, patch, band), dtype=np.float32)
    x_true = np.zeros((true_point.shape[0], patch, patch, band), dtype=np.float32)

    for i in range(train_point.shape[0]):
        x_train[i,:,:,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patch)
    for j in range(test_point.shape[0]):
        x_test[j,:,:,:] = gain_neighborhood_pixel(mirror_image, test_point, j, patch)
    for k in range(true_point.shape[0]):
        x_true[k,:,:,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patch)

    print("x_train shape = {}, type = {}".format(x_train.shape,x_train.dtype))
    print("x_test  shape = {}, type = {}".format(x_test.shape,x_test.dtype))
    print("x_true  shape = {}, type = {}".format(x_true.shape,x_test.dtype))
    print("**************************************************")

    if flag == 'test':
        x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
        x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)
        x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
        # x_train_band = x_train
    else:
        x_train_band = gain_neighborhood_band(x_train, band, band_patch, patch)
        x_test_band = gain_neighborhood_band(x_test, band, band_patch, patch)
        # x_true_band = x_true
        x_true_band = gain_neighborhood_band(x_true, band, band_patch, patch)

    print("x_train_band shape = {}, type = {}".format(x_train_band.shape,x_train_band.dtype))
    print("x_test_band  shape = {}, type = {}".format(x_test_band.shape,x_test_band.dtype))
    print("x_true_band  shape = {}, type = {}".format(x_true_band.shape,x_true_band.dtype))
    print("**************************************************")
    return x_train_band, x_test_band, x_true_band
#-------------------------------------------------------------------------------
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    print("**************************************************")
    return y_train, y_test, y_true
#-------------------------------------------------------------------------------
class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt
#-------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res, target, pred.squeeze()

def train_epoch(model, train_loader, criterion, optimizer, args):
    objs = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_mask) in enumerate(train_loader):
        if len(batch_data.size()) == 2:
            batch_data= batch_data.unsqueeze(0)
        B, _, C = batch_data.shape
        # print('batch_data.size {}, batch_mask.size {}'.format(batch_data.size(), batch_mask.size()))
        #batch_data.size torch.Size([64, 200, 147]), batch_mask.size torch.Size([64, 200])
        batch_target = batch_data[batch_mask].reshape(B, -1, C)
        batch_data = batch_data.cuda()
        batch_mask = batch_mask.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        if args.mask_method in ['high', 'low' ,'high_random']:
            batch_pred, voted_mask, out = model(batch_data, batch_mask)
            batch_target = batch_data[voted_mask].reshape(B, -1, C)
        else:
            batch_pred, out = model(batch_data, batch_mask)
        # print(batch_target.size(), batch_pred.size())

        if args.mask_method is not None:
            loss = criterion(batch_pred, batch_target)
        else:
            loss = criterion(batch_pred, batch_data)

        # if loss_align>loss*10:
        #     loss += 0.1*loss_align
        # # elif loss_align>loss*5:
        # #     loss += 0.1 * loss_align
        # else:
        #     loss += 0.2 * loss_align
        if args.align is not None:
            # loss_align = torch.cdist(out[:, 0], out[:, 1:].mean(1)).mean()
            loss_align = out
            loss += 0.3*loss_align
        loss.backward()
        optimizer.step()

        n = batch_data.shape[0]
        objs.update(loss.data, n)
        tar = np.append(tar, batch_target.data.cpu().numpy())
        pre = np.append(pre, batch_pred.data.cpu().numpy())

    return objs.avg, tar, pre
#-------------------------------------------------------------------------------
def valid_epoch(model, valid_loader, criterion, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_mask, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_mask = batch_mask.cuda()
        out = model(batch_data, batch_mask)
        if args.align is not None:
            if args.use_sar:
                assert len(out) == 3
                # batch_pred, loss_align, patch_pred = out
                batch_pred, loss_align, loss_sar = out
            else:
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
def valid_epoch_nomask(model, valid_loader, criterion, optimizer, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_mask, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        batch_mask = batch_mask.cuda()
        out = model(batch_data, None)
        if args.align is not None:
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

def test_epoch(model, test_loader, criterion, optimizer, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_target) in enumerate(test_loader):
        if len(batch_data.size()) == 2:
            batch_data= batch_data.unsqueeze(0)
        batch_data = batch_data.cuda()
        batch_target = batch_target.cuda()
        out = model(batch_data, None)
        if args.align is not None:
            assert len(out) == 2
            batch_pred, loss_align = out
        else:
            batch_pred = out
        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre
#-------------------------------------------------------------------------------
def tune_epoch(model, tune_loader, criterion, optimizer, args):
    loss_all = AvgrageMeter()
    loss_cls = AvgrageMeter()
    loss_alig = AvgrageMeter()
    loss_sars = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_mask, batch_target) in enumerate(tune_loader):
        if len(batch_data.size()) == 2:
            batch_data= batch_data.unsqueeze(0)
        batch_data = batch_data.cuda()
        batch_mask = batch_mask.cuda()
        batch_target = batch_target.cuda()

        optimizer.zero_grad()
        out = model(batch_data, batch_mask)
        if args.align is not None:
            if args.use_sar:
                assert len(out) == 3
                # batch_pred, loss_align, patch_pred = out
                # loss_patch = criterion(patch_pred.squeeze(), batch_target)
                batch_pred, loss_align, loss_sar = out
            else:
                assert len(out) == 2
                batch_pred, loss_align = out
                ## loss_align = torch.cdist(out[:, 0], out[:, 1:].mean(1)).mean()
                loss_sar = torch.tensor(0.0)
        else:
            batch_pred = out
            loss_align = torch.tensor(0.0)
            loss_sar = torch.tensor(0.0)
        loss_c = criterion(batch_pred, batch_target)
        # # loss_align = (1/(5*loss_c)) *loss_align
        loss = loss_c + loss_align + loss_sar
        # loss = 0.35*loss_c + 0.65*loss_align + loss_sar

        loss.backward()
        optimizer.step()

        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        loss_all.update(loss.data, n)
        loss_cls.update(loss_c.data, n)
        loss_alig.update(loss_align.data, n)
        loss_sars.update(loss_sar.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return top1.avg, loss_all.avg, tar, pre, loss_cls.avg, loss_alig.avg, loss_sars.avg
#-------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float64)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
#-------------------------------------------------------------------------------
# def recordLoss(name, epoch, loss=0.0, accur=0.0, args):
#     with open('./logs/'+name, 'a') as loss_file:
#         if epoch == 1:
#             loss_file.write('[')
#         new_loss = {"epoch": epoch, 'loss':loss, 'accuracy':accur}
#         loss_file.write(json.dumps(new_loss))
#         # loss_file.write('\n')
#         if epoch == args.epoches:
#             loss_file.write(']')
#         else:
#             loss_file.write(',\n')
#         loss_file.close()


class RandomMaskingGenerator:
    def __init__(self, number_patches, mask_ratio):
        self.number_patches = number_patches
        self.num_mask = int(mask_ratio * self.number_patches)

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.number_patches, self.num_mask
        )
        return repr_str

    def __call__(self):
        mask = np.hstack([
            np.zeros(self.number_patches - self.num_mask),
            np.ones(self.num_mask),
        ])
        np.random.shuffle(mask)
        return mask

def save_model(args, epoch, model, optimizer):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)
    checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
    for checkpoint_path in checkpoint_paths:
        to_save = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        torch.save(to_save, checkpoint_path)
#-------------------------------------------------------------------------------
def load_state_dict(model, state_dict, prefix='', ignore_missing="relative_position_index"):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix=prefix)

    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        keep_flag = True
        for ignore_key in ignore_missing.split('|'):
            if ignore_key in key:
                keep_flag = False
                break
        if keep_flag:
            warn_missing_keys.append(key)
        else:
            ignore_missing_keys.append(key)

    missing_keys = warn_missing_keys

    if len(missing_keys) > 0:
        print("Weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, missing_keys))
    if len(unexpected_keys) > 0:
        print("Weights from pretrained model not used in {}: {}".format(
            model.__class__.__name__, unexpected_keys))
    if len(ignore_missing_keys) > 0:
        print("Ignored weights of {} not initialized from pretrained model: {}".format(
            model.__class__.__name__, ignore_missing_keys))
    if len(error_msgs) > 0:
        print('\n'.join(error_msgs))


###code from https://github.com/1Konny/gradcam_plus_plus-pytorch/blob/master/utils.py for gradCAM
###***********************************************
# def visualize_cam(mask, img):
#     """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
#     Args:
#         mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
#         img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
#
#     Return:
#         heatmap (torch.tensor): heatmap img shape of (3, H, W)
#         result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
#     """
#     heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
#     heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
#     b, g, r = heatmap.split(1)
#     heatmap = torch.cat([r, g, b])
#
#     result = heatmap+img.cpu()
#     result = result.div(result.max()).squeeze()
#
#     return heatmap, result

def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer

def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer

def find_vgg_layer(arch, target_layer_name):
    """Find vgg layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_42'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer

def find_alexnet_layer(arch, target_layer_name):
    """Find alexnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_0'
            target_layer_name = 'classifier'
            target_layer_name = 'classifier_0'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')

    if len(hierarchy) >= 1:
        target_layer = arch.features

    if len(hierarchy) == 2:
        target_layer = target_layer[int(hierarchy[1])]

    return target_layer

def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features_12'
            target_layer_name = 'features_12_expand3x3'
            target_layer_name = 'features_12_expand3x3_activation'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2]+'_'+hierarchy[3]]

    return target_layer

def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)

def normalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.sub(mean).div(std)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return self.do(tensor)

    def do(self, tensor):
        return normalize(tensor, self.mean, self.std)

    def undo(self, tensor):
        return denormalize(tensor, self.mean, self.std)

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
###***********************************************