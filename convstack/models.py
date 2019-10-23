import torch
import torch.nn as nn
import torchvision
from convstack.torch_utils import *
import numpy as np
from scipy import signal

DEVICE = torch.device('cuda:0')

def try_kwarg(kwargs, key, default):
    try:
        return kwargs[key]
    except:
        return default

class TDRModel(nn.Module):
    def __init__(self, n_units=5, bias=True, linear_bias=None, chans=[8,8],
                                                bn_moment=.01, softplus=True, 
                                                inference_exp=False, img_shape=(3,224,224), 
                                                ksizes=(15,11), recurrent=False, 
                                                kinetic=False, centers=None, 
                                                bnorm_d=1, **kwargs):
        super().__init__()
        self.n_units = n_units
        self.chans = chans 
        self.softplus = softplus 
        self.infr_exp = inference_exp 
        self.bias = bias 
        self.img_shape = img_shape 
        self.ksizes = ksizes 
        self.linear_bias = linear_bias 
        self.bn_moment = bn_moment 
        self.recurrent = recurrent
        self.kinetic = kinetic
        self.centers = centers
        assert bnorm_d == 1 or bnorm_d == 2,\
                    "Only 1 and 2 dimensional batchnorm are currently supported"
        self.bnorm_d = bnorm_d
    
    def forward(self, x):
        return x

    def extra_repr(self):
        try:
            s = 'n_units={}, bias={}, linear_bias={}, chans={}, bn_moment={}, '+\
                                    'softplus={}, inference_exp={}, img_shape={}, ksizes={}'
            return s.format(self.n_units, self.bias, self.linear_bias,
                                        self.chans, self.bn_moment, self.softplus,
                                        self.inference_exp, self.img_shape, self.ksizes)
        except:
            pass
    
    def requires_grad(self, state):
        for p in self.parameters():
            try:
                p.requires_grad = state
            except:
                pass

class AlexNet(TDRModel):
    def __init__(self, drop_p=0.03, bnorm=False, pretrained=False, locrespnorm=False, 
                                                          stackconvs=False, **kwargs):
        """
        This class recreates AlexNet from the paper "ImageNet Classification with Deep 
        Convolutional Neural Networks". It allows for easy manipulations.

        drop_p: float
            refers to dropout probability of stacked convolutions. Only applies if stackconvs is
            true.
        bnorm: bool
            if true, bnorm layers are used after each convolution and each linear layer
        pretrained: bool
            if true and not using stackconvs, all trainable parameters are initialized to a 
            fully trained state of the original model architecture. 
            i.e. bnorm=False,locrespnorm=False,stackconvs=False
        locrespnorm: bool
            if true, uses local response norm as described in the original paper "ImageNet 
            Classification with Deep Convolutional Neural Networks"
        stackconvs: bool
            if true, replaces convolutional layers with equivalent linearstacked convolutions
        """
        super().__init__(**kwargs)
        alexnet = torchvision.models.alexnet(pretrained=pretrained)

        features = alexnet.features
        avgpool = alexnet.avgpool
        fc_layers = alexnet.classifier

        modules = []
        self.shapes = []
        self.chans = []
        shape = [x for x in self.img_shape[1:]]

        ### Add features
        for i,modu in enumerate(features):
            if isinstance(modu, nn.Conv2d):
                in_chans = modu.in_channels
                out_chans = modu.out_channels
                ksize = modu.kernel_size
                padding = modu.padding if isinstance(modu.padding, int) else modu.padding[0]
                stride = modu.stride
                shape = update_shape(shape, kernel=ksize, padding=padding, stride=stride)
                self.shapes.append(shape)
                self.chans.append(out_chans)
                if stackconvs:
                    bias = modu.bias is not None
                    modu = LinearStackedConv2d(in_chans, out_chans, ksize, bias=bias, 
                                                drop_p=drop_p, padding=padding, stride=stride,
                                                stack_chan=stack_chan, bnorm=stackbnorm)
            if isinstance(modu, nn.ReLU) and locrespnorm and i <= 4:
                modules.append(nn.LocalResponseNorm(5,k=2, alpha=1e-4, beta=.75))
            if isinstance(modu, nn.ReLU) and bnorm:
                modules.append(nn.BatchNorm2d(out_chans,momentum=self.bn_moment))
            modules.append(modu)

        ### AvgPool and Flatten
        modules.append(avgpool)
        modules.append(Flatten())
        flat_size = out_chans*shape[0]*shape[1]

        ### Add Fully Connected Layers
        for i,mod in enumerate(fc_layers):
            if isinstance(modu, nn.ReLU) and bnorm:
                modules.append(nn.BatchNorm1d(flat_size,momentum=self.bn_moment))
        
        self.sequential = nn.Sequential(*modules)

    def forward(self, x):
        return self.sequential(x)

