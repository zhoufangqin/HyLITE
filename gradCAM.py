## code from https://github.com/1Konny/gradcam_plus_plus-pytorch/blob/master/gradcam.py
import torch
import torch.nn.functional as F
from utils import find_alexnet_layer, find_vgg_layer, find_resnet_layer, find_densenet_layer, find_squeezenet_layer
import numpy as np

class GradCAM(object):
    """Calculate GradCAM salinecy map.

    A simple example:

        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)

        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalizer(img)

        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)

        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)


    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            # print(module)
            self.gradients['value'] = grad_output[0]
            # self.gradients['value'] = grad_output
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'vgg' in model_type.lower():
            target_layer = find_vgg_layer(self.model_arch, layer_name)
        elif 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'densenet' in model_type.lower():
            target_layer = find_densenet_layer(self.model_arch, layer_name)
        elif 'alexnet' in model_type.lower():
            target_layer = find_alexnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif 'maest':
            # target_layer = self.model_arch.mlp_head[1]
            t_layer4 = self.model_arch.transformer.layers[4][1]
            t_layer4.register_forward_hook(forward_hook)
            t_layer4.register_full_backward_hook(backward_hook)
            # t_layer3 = self.model_arch.transformer.layers[3][1]
            # t_layer3.register_forward_hook(forward_hook)
            # t_layer3.register_backward_hook(backward_hook)

        # target_layer.register_forward_hook(forward_hook)
        # target_layer.register_backward_hook(backward_hook)
        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])

    def forward(self, input, class_idx=None, retain_graph=False, return_attn=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx (int): class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        # b, c, h, w = input.size()
        b, c, p = input.size() #1,200,147

        if return_attn:
            logit, attn = self.model_arch(input, return_attn=return_attn)
        else:
            logit = self.model_arch(input) #1,16
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()
        self.model_arch.zero_grad()
        score.backward(retain_graph=retain_graph)
        if torch.isnan(self.gradients['value'].mean(1).squeeze()).any():
            print('gradients: ', self.gradients['value'])
        gradients = self.gradients['value'].mean(1) #1,1,64
        # print('score is {} and unique values in gradients {}'.format(score, gradients.unique()))
        activations = self.activations['value'][:,1:]
        if torch.isnan(activations.squeeze()).any():
            print('activations: ', activations)
        b, pp = gradients.size() #1,64
        # alpha = gradients.permute(0, 2, 1).mean(2) #1,64
        weights = gradients.view(b,pp,1) #1,64,1
        activations = activations.permute(0,2,1) #1,64,200
        saliency_map = (weights * activations).sum(1, keepdim=True) #1,200
        if torch.isnan(saliency_map.squeeze()).any():
            print('before relu: ', saliency_map)
        saliency_map = F.relu(saliency_map)
        if torch.isnan(saliency_map.squeeze()).any():
            print('after relu: ', saliency_map)
        # print(saliency_map.size(), saliency_map.unique())
        # saliency_map = F.interpolate(saliency_map, mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        if saliency_map_max != saliency_map_min:
            saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data
        else:
            saliency_map = saliency_map.data
        # saliency_map /= saliency_map.max()

        if return_attn:
            if len(attn.size() == 2):
                attn = attn.unsqueeze(0)
            return saliency_map, logit, attn.mean(1) #mean attn over heads and sum over
        else:
            return saliency_map, logit
    def __call__(self, input, class_idx=None, retain_graph=False, return_attn=False):
        return self.forward(input, class_idx, retain_graph, return_attn=return_attn)

def get_cam(model, x_test, y_test, class_num=1, return_attn=False):
    assert class_num >= 1
    model.eval()
    model_dict = dict(type='maest', arch=model, layer_name='mlp_head.Linear', input_size=(200, 147))
    gradcam = GradCAM(model_dict)
    data = x_test[y_test==(class_num-1)].cuda() #y_test real class starts from 0
    correct_maps = []
    wrong_maps = []
    correct_attns = []
    wrong_attns = []
    for i in range(data.size(0)):
    # for i in range(1):
        model.zero_grad()
        if return_attn:
            saliency_map, logit, attn = gradcam(data[i].unsqueeze(0), class_num-1, return_attn=return_attn)
            att = attn[:, 1:, 1:].sum(1).squeeze().detach().cpu().numpy()
            att = (att - att.min())/(att.max() - att.min())
            # att /= att.max()
            if (class_num-1) == logit.max(1)[-1]:
                correct_attns.append(att)
            else:
                wrong_attns.append(att)
        else:
            saliency_map, logit = gradcam(data[i].unsqueeze(0), class_num-1)
        if torch.isnan(saliency_map.squeeze()).any():
            print('final: ', saliency_map)
        if (class_num-1) == logit.max(1)[-1]:
            correct_maps.append(saliency_map.squeeze().detach().cpu().numpy())
        else:
            wrong_maps.append(saliency_map.squeeze().detach().cpu().numpy())
    if len(wrong_maps) == 0:
        wrong_maps = np.zeros((1,x_test.size(1)))
    if len(wrong_attns) == 0:
        wrong_attns = np.zeros((1,x_test.size(1)))
    if return_attn:
        return np.array(correct_maps).mean(0), np.array(wrong_maps).mean(0), np.array(correct_attns).mean(0), np.array(wrong_attns).mean(0)
    else:
        return np.array(correct_maps).mean(0), np.array(wrong_maps).mean(0)

def save_gradcams(model, x, y, num_classes, x_test, y_test, args):
    all_correct_maps = []
    all_wrong_maps = []
    return_attn = False
    all_correct_attns = []
    all_wrong_attns = []
    for c in range(1, num_classes + 1):
        if return_attn:
            r, w, correct_attn, wrong_attn = get_cam(model, x, y, c, return_attn=return_attn)
            all_correct_attns.append(correct_attn)
            all_wrong_attns.append(wrong_attn)
        else:
            r, w = get_cam(model, x_test, y_test, c)
        all_correct_maps.append(r)
        all_wrong_maps.append(w)
    if args.mask_method is not None:
        np.save('./attention_maps_maest_' + args.dataset, all_correct_attns)
    else:
        np.save('./gradcam_maps_vit_' + args.dataset, all_correct_maps)
        # np.save('./gradcam_maps_vit_fulldata_'+args.dataset, all_correct_maps)