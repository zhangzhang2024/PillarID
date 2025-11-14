import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

def down_stage(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
    m = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),)
    
    return m

class Conv(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride, dilation=1, groups=1,
                 conv_layer=nn.Conv2d, bias=False, **kwargs):
        super(Conv, self).__init__()
        padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = conv_layer(inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding, groups=groups, bias=bias)
         
    def forward(self, x):
        return self.conv(x)

class ConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, dilation=1, padding=1,
                 conv_layer=nn.Conv2d,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU, **kwargs):
        super(ConvBlock, self).__init__()
        # padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = Conv(inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding, bias=False, conv_layer=conv_layer)

        self.norm = norm_layer(planes)
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out

class DWConvBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, dilation=1, padding=1,
                 conv_layer=nn.Conv2d,
                 norm_layer=nn.BatchNorm2d,
                 act_layer=nn.ReLU, **kwargs):
        super(DWConvBlock, self).__init__()
        # padding = kwargs.get('padding', kernel_size // 2)  # dafault same size

        self.conv = Conv(inplanes, planes, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=padding, groups=inplanes, bias=False, conv_layer=conv_layer)

        self.norm = norm_layer(planes)
        self.act = act_layer()

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        return out

class BasicBlock(nn.Module):
    def __init__(self, inplanes, kernel_size=3):
        super(BasicBlock, self).__init__()
        self.block1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.block2 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.act(out)

        return out

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        # import pdb;pdb.set_trace()
        # draw_feature_map(x[0].unsqueeze(0))
        data_dict['spatial_features_2d'] = x

        return data_dict

class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)
        # import pdb;pdb.set_trace()
        data_dict['spatial_features_2d'] = x

        return data_dict

class Pillar16xEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            
            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)
        # import pdb;pdb.set_trace()
        # draw_feature_map(x[0].unsqueeze(0))
        data_dict['spatial_features_2d'] = x

        return data_dict

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class ConvC2f(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class BottleneckC2f(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, d=1):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = ConvC2f(c1, c_, k[0], 1)
        self.cv2 = ConvC2f(c_, c2, k[1], 1, g=g, d=d)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = ConvC2f(c1, 2 * self.c, 1, 1)
        self.cv2 = ConvC2f((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(BottleneckC2f(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
    
    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        
        return self.cv2(torch.cat(y, 1))
    
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        
        return self.cv2(torch.cat(y, 1))

import warnings
class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

class PillarC2fSPPFEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.conv_input = nn.Sequential(
            C2f(64, 64, 2, True, 32),
            # SPPF(64, 64, 5),
            )
        
        self.conv1 = nn.Sequential(
            down_stage(64, 64, 3, 2, 1),
            C2f(64, 64, 2, True, 32),
            SPPF(64, 64, 5),
            )
        
        self.conv2 = nn.Sequential(
            down_stage(64, 128, 1, 1, 0),
            C2f(128, 128, 6, True, 64),
            SPPF(128, 128, 7),
            )
        
        self.conv3 = nn.Sequential(
            C2f(128, 128, 2, True, 64),
            SPPF(128, 128, 7),
            )
        
        self.fuse_conv = C2f(320, 320, 2, True, 160)

        self.num_bev_features = 64

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        x = data_dict['spatial_features']
        x = self.conv_input(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x_fuse = self.fuse_conv(torch.cat((x1, x2, x3), dim=1))
        # import pdb;pdb.set_trace()
        # draw_feature_map(x_fuse[0].unsqueeze(0))
        data_dict['spatial_features_2d'] = x_fuse

        return data_dict

class SPDC(nn.Module):
    # Spatial Pyramid Dilation Conv (SPDC) 
    def __init__(self, c1, c2, dilation=[2, 4, 6, 8]):  
        super().__init__()
        c_ = c1 // 4  
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_*5, c2, 1, 1)
        self.di1 = ConvC2f(c_, c_, k=3, s=1, g=c_, d=dilation[0])
        self.di2 = ConvC2f(c_, c_, k=3, s=1, g=c_, d=dilation[1])
        self.di3 = ConvC2f(c_, c_, k=3, s=1, g=c_, d=dilation[2])
        self.di4 = ConvC2f(c_, c_, k=3, s=1, g=c_, d=dilation[3])

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.di1(x)
        y2 = self.di2(y1)
        y3 = self.di3(y2)
        y4 = self.di4(y3)
        
        return self.cv2(torch.cat((x, y1, y2, y3, y4), 1))

class PillarC2fSPDCEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.conv_input = nn.Sequential(
            C2f(64, 64, 2, True, 32),
            # SPPF(64, 64, 5),
            )
        
        self.conv1 = nn.Sequential(
            down_stage(64, 64, 3, 2, 1),
            C2f(64, 64, 2, True, 32),
            SPDC(64, 64),
            )
        
        self.conv2 = nn.Sequential(
            down_stage(64, 128, 1, 1, 0),
            C2f(128, 128, 6, True, 64),
            SPDC(128, 128),
            )
        
        self.conv3 = nn.Sequential(
            C2f(128, 128, 2, True, 64),
            SPDC(128, 128),
            )
        
        self.fuse_conv = C2f(320, 320, 2, True, 160)

        self.num_bev_features = 64

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        x = data_dict['spatial_features']
        x = self.conv_input(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x_fuse = self.fuse_conv(torch.cat((x1, x2, x3), dim=1))
        # import pdb;pdb.set_trace()
        # draw_feature_map(x_fuse[0].unsqueeze(0))
        data_dict['spatial_features_2d'] = x_fuse

        return data_dict

class PillarIDEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.conv_input = nn.Sequential(
            C2f(64, 64, 2, True, 32),
            )
        
        self.conv1 = nn.Sequential(
            down_stage(64, 64, 3, 2, 1),
            C2f(64, 64, 2, True, 32),
            SPDC(64, 64),
            )
        
        self.conv2 = nn.Sequential(
            down_stage(64, 128, 1, 1, 0),
            C2f(128, 128, 6, True, 64),
            SPDC(128, 128),
            )
        
        self.conv3 = nn.Sequential(
            C2f(128, 128, 2, True, 64),
            SPDC(128, 128),
            )
        
        self.fuse_conv = C2f(320, 320, 2, True, 160)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(320, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
            )

        self.num_bev_features = 64

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        x = data_dict['spatial_features']
        x = self.conv_input(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x_fuse = self.fuse_conv(torch.cat((x1, x2, x3), dim=1))

        data_dict['spatial_features_2d'] = self.deconv(x_fuse)

        return data_dict

class PillarSCDEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.conv_input = nn.Sequential(
            C2f(64, 64, 2, True, 32),
            )
        
        self.conv1 = nn.Sequential(
            down_stage(64, 64, 3, 2, 1),
            C2f(64, 64, 2, True, 32),
            # SPDC(64, 64),
            )
        
        self.conv2 = nn.Sequential(
            down_stage(64, 128, 1, 1, 0),
            C2f(128, 128, 6, True, 64),
            # SPDC(128, 128),
            )
        
        self.conv3 = nn.Sequential(
            C2f(128, 128, 2, True, 64),
            # SPDC(128, 128),
            )
        
        self.fuse_conv = C2f(320, 320, 2, True, 160)
        
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(320, 128, 2, stride=2, bias=False),
            nn.BatchNorm2d(128, eps=1e-3, momentum=0.01),
            nn.ReLU()
            )

        self.num_bev_features = 64

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        x = data_dict['spatial_features']
        x = self.conv_input(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x_fuse = self.fuse_conv(torch.cat((x1, x2, x3), dim=1))
        # import pdb;pdb.set_trace()
        # draw_feature_map(x_fuse[0].unsqueeze(0))
        data_dict['spatial_features_2d'] = self.deconv(x_fuse)

        return data_dict

class BasicNextBlock(nn.Module):
    def __init__(self, inplanes, kernel_size=3):
        super(BasicNextBlock, self).__init__()
        self.block1 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.block2 = ConvBlock(inplanes, inplanes, kernel_size=kernel_size)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.block1(x)
        out = self.block2(out)
        out = out + identity
        out = self.act(out)

        return out
    
class ASPPNeck(nn.Module):
    def __init__(self, in_channels):

        super(ASPPNeck, self).__init__()

        self.pre_conv = BasicNextBlock(in_channels)
        self.conv1x1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, bias=False, padding=0)
        self.weight = nn.Parameter(torch.randn(in_channels, in_channels, 3, 3))
        self.post_conv = ConvBlock(in_channels * 6, in_channels, kernel_size=1, stride=1, padding=0)

    def _forward(self, x):
        x = self.pre_conv(x)
        branch1x1 = self.conv1x1(x)
        branch1 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=1, dilation=1)
        branch6 = F.conv2d(x, self.weight, stride=1,
                           bias=None, padding=6, dilation=6)
        branch12 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=12, dilation=12)
        branch18 = F.conv2d(x, self.weight, stride=1,
                            bias=None, padding=18, dilation=18)
        x = self.post_conv(
            torch.cat((x, branch1x1, branch1, branch6, branch12, branch18), dim=1))
        return x

    def forward(self, x):
        
        out = self._forward(x)

        return out
    
class PillarC2fASPPEncoder(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        self.conv_input = nn.Sequential(
            C2f(64, 64, 2, True, 32),
            # SPPF(64, 64, 5),
            )
        
        self.conv1 = nn.Sequential(
            down_stage(64, 64, 3, 2, 1),
            C2f(64, 64, 2, True, 32),
            # SPDC(64, 64),
            ASPPNeck(64),
            )
        
        self.conv2 = nn.Sequential(
            down_stage(64, 128, 1, 1, 0),
            C2f(128, 128, 6, True, 64),
            # SPDC(128, 128),
            ASPPNeck(128),
            )
        
        self.conv3 = nn.Sequential(
            C2f(128, 128, 2, True, 64),
            # SPDC(128, 128),
            ASPPNeck(128),
            )
        
        self.fuse_conv = C2f(320, 320, 2, True, 160)

        self.num_bev_features = 64

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        x = data_dict['spatial_features']
        x = self.conv_input(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # import pdb;pdb.set_trace()
        x_fuse = self.fuse_conv(torch.cat((x1, x2, x3), dim=1))
        # import pdb;pdb.set_trace()
        # draw_feature_map(x_fuse[0].unsqueeze(0))
        data_dict['spatial_features_2d'] = x_fuse

        return data_dict
  