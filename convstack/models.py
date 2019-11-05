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
                                                ksizes=[15,11], recurrent=False, 
                                                kinetic=False, centers=None, 
                                                bnorm_d=1, stackconvs=False, 
                                                strides=None, paddings=None,
                                                bnorm=False, drop_p=0.03, 
                                                stackchans=None, fc_drp=0,
                                                n_shakes=1, lin_shakes=None, 
                                                batch_size=1000,**kwargs):
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
        self.stackconvs = stackconvs if isinstance(stackconvs, list) else [stackconvs
                                                                for x in self.ksizes]
        self.strides = strides if strides is not None else [1 for x in self.ksizes]
        self.paddings = paddings if paddings is not None else [0 for x in self.ksizes]
        assert len(self.strides) == len(self.ksizes) and len(self.paddings) == len(self.ksizes)
        self.stackchans = stackchans if stackchans is not None else [c for c in self.chans]
        self.bnorm = bnorm
        self.drop_p = drop_p if isinstance(drop_p, list) else [drop_p for x in self.ksizes]
        self.fc_drp = fc_drp
        self.n_shakes = n_shakes
        if lin_shakes is None:
            lin_shakes = n_shakes
        self.lin_shakes = lin_shakes
        self.batch_size = batch_size
    
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
    def __init__(self, pretrained=False, locrespnorm=False, stackbnorm=False,
                                                                **kwargs):
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
                # TODO: fix stackconvs and drop_p
                assert False , "Need to change stack convs to list"
                if self.stackconvs:
                    bias = modu.bias is not None
                    modu = LinearStackedConv2d(in_chans, out_chans, ksize, bias=bias, 
                                                drop_p=drop_p, padding=padding, stride=stride,
                                                stack_chan=stack_chan, bnorm=stackbnorm)
                if self.n_shakes > 1:
                    modu = ShakeShakeModule(modu,self.n_shakes)
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

class SmallNet(TDRModel):
    def __init__(self, stackbnorms=None, mid_dims=128, **kwargs):
        super(SmallNet,self).__init__(**kwargs)
        self.shapes = []
        shape = self.img_shape[1:]
        self.stackbnorms = stackbnorms if stackbnorms is not None else [c for c in self.chans]

        modules = []
        chans = [self.img_shape[0],*self.chans]
        n_layers = len(self.ksizes)
        for i in range(len(self.ksizes)):
            in_chan = chans[i]
            out_chan = chans[i+1]
            ksize = self.ksizes[i]
            stride = self.strides[i]
            padding = self.paddings[i]
            drop_p = self.drop_p[i]
            if self.stackconvs[i]:
                stack_chan = self.stackchans[i]
                stackbnorm = self.stackbnorms[i]
                modules.append(LinearStackedConv2d(in_chan, out_chan, stride=stride, 
                                                       kernel_size=ksize, padding=padding,
                                                       stack_chan=stack_chan, 
                                                       drop_p=drop_p, bnorm=stackbnorm))
            else:
                modules.append(nn.Conv2d(in_chan, out_chan,stride=stride, padding=padding,
                                                                        kernel_size=ksize))
            if self.n_shakes > 1:
                modules[-1] = ShakeShakeModule(modules[-1], n_shakes=self.n_shakes, 
                                                        batch_size=self.batch_size)
            shape = update_shape(shape, kernel=ksize, padding=padding, stride=stride)
            self.shapes.append(shape)
            modules.append(nn.ReLU())
            if self.bnorm:
                modules.append(nn.BatchNorm2d(out_chan, momentum=self.bn_moment))

        modules.append(Flatten())
        in_chan = out_chan*shape[0]*shape[1]
        modules.append(nn.Linear(in_chan,mid_dims))
        if self.lin_shakes > 1:
            modules[-1] = ShakeShakeModule(modules[-1], n_shakes=self.lin_shakes, 
                                                        batch_size=self.batch_size)
        if self.fc_drp > 0 and self.fc_drp < 1:
            modules.append(nn.Dropout(self.fc_drp))
        modules.append(nn.LeakyReLU())
        modules.append(nn.Linear(mid_dims,self.n_units))
        if self.lin_shakes > 1:
            modules[-1] = ShakeShakeModule(modules[-1], n_shakes=self.lin_shakes, 
                                                        batch_size=self.batch_size)
        self.sequential = nn.Sequential(*modules)

    def forward(self,x):
        fx = self.sequential(x)
        return fx

# Example Loader
def load_model(fpath,steps=1000):
    epoch_data={}
    try:
        temp = torch.load(fpath)
        epoch_data['loss'] = temp['loss']
        epoch_data['acc'] = temp['acc']
    except:
        pass
    

    return epoch_data
