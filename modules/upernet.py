import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from itertools import chain
import os
import numpy as np
import math
import PIL
import logging
from collections import OrderedDict
from .internimage import *


def set_trainable_attr(m,b):
    m.trainable = b
    for p in m.parameters(): p.requires_grad = b

    
def apply_leaf(m, f):
    c = m if isinstance(m, (list, tuple)) else list(m.children())
    if isinstance(m, nn.Module):
        f(m)
    if len(c)>0:
        for l in c:
            apply_leaf(l,f)

            
def set_trainable(l, b):
    apply_leaf(l, lambda m: set_trainable_attr(m,b))
    
    
class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling 
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s) 
                                                        for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels+(out_channels * len(bin_sizes)), in_channels, 
                                    kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)
    
    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear', 
                                        align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output

    
def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[64, 128, 256, 512], fpn_out=64):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                    for ft_size in feature_channels[1:]])
        self.smooth_conv =  nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)] 
                                    * (len(feature_channels)-1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels)*fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]
        P = [up_and_add(features[i], features[i-1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1]) #P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x

    
def summary(model, input_shape, batch_size=-1, intputshow=True):

    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) 
                    and not (module == model)) and 'torch' in str(module.__class__):
            if intputshow is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(torch.zeros(input_shape))

    # remove these hooks
    for h in hooks:
        h.remove()

    model_info = ''

    model_info += "-----------------------------------------------------------------------\n"
    line_new = "{:>25}  {:>25} {:>15}".format("Layer (type)", "Input Shape", "Param #")
    model_info += line_new + '\n'
    model_info += "=======================================================================\n"

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        line_new = "{:>25}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["input_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )

        total_params += summary[layer]["nb_params"]
        if intputshow is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        model_info += line_new + '\n'

    model_info += "=======================================================================\n"
    model_info += "Total params: {0:,}\n".format(total_params)
    model_info += "Trainable params: {0:,}\n".format(trainable_params)
    model_info += "Non-trainable params: {0:,}\n".format(total_params - trainable_params)
    model_info += "-----------------------------------------------------------------------\n"

    return model_info


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
        #return summary(self, input_shape=(2, 3, 224, 224))

class Config:
    def __init__(self, 
                 depths=[4, 4, 18, 4], 
                 groups=[4, 8, 16, 32],
                 mlp_ratio=4., 
                 drop_path_rate=0.2,
                 norm_layer='LN',
                 layer_scale=1.0,
                 offset_scale=1.0,
                 post_norm=False,
                 with_cp=False, 
                 out_indices=(0, 1, 2, 3),
                 feature_channels=[64, 128, 256, 512],
                 num_labels=4):
        self.num_labels = num_labels
        self.depths = depths
        self.groups = groups
        self.mlp_ratio = mlp_ratio
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.offset_scale = offset_scale
        self.post_norm = post_norm
        self.with_cp = with_cp
        self.out_indices = out_indices
        self.feature_channels = feature_channels
        
    
class UperNet(BaseModel):
    # Implementing only the object path
    def __init__(self, in_channels=3, config=None, pretrained=False, use_aux=True, freeze_bn=False, freeze_backbone=False,**_):
        super(UperNet, self).__init__()
        
        if config == None:
            config = Config()
#         if backbone == 'resnet34' or backbone == 'resnet18':
#             feature_channels = [64, 128, 256, 512]
#         else:
#             feature_channels = [256, 512, 1024, 2048]
        # self.backbone = ResNet(in_channels, pretrained=pretrained)
        feature_channels = config.feature_channels
        fpn_out = config.feature_channels[0]
        self.backbone = InternImage(
                        channels = config.feature_channels[0],
                        depths = config.depths,
                        groups = config.groups,
                        mlp_ratio = config.mlp_ratio,
                        drop_path_rate = config.drop_path_rate,
                        norm_layer = config.norm_layer,
                        offset_scale = config.offset_scale,
                        post_norm = config.post_norm,
                        with_cp = config.with_cp,
                        out_indices = config.out_indices,
                        feature_channels = config.feature_channels)
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=fpn_out)
        # self.head = F.softmax(nn.Conv2d(fpn_out, config.num_labels, kernel_size=3, padding=1), dim=1)
        self.head = nn.Sequential(
                    nn.Conv2d(fpn_out, config.num_labels, kernel_size=1),
                    nn.Softmax(dim=1)
                )
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.backbone], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode='bilinear')
        return x 

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
