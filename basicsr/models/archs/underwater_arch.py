import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse
from pdb import set_trace as stx
import numbers

from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False, groups=n_feat),
                                  nn.Conv2d(n_feat, n_feat//2, kernel_size=1, stride=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=False, groups=n_feat),
                                  nn.Conv2d(n_feat, 2*n_feat, kernel_size=1, stride=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
    
    
    
##########################################################################
##-------- Low Frequency Block----------------

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=4):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.GELU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Sequential(*[nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class LowFrequencyBlock(nn.Module): # ref: CBAM 
    def __init__(self,
        channels,     
        kernel_size=7,
        reduction=2,
    ):
        super(LowFrequencyBlock, self).__init__()

        self.prj1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.GELU()
        )
        self.ca = ChannelAttention(channels, reduction)
        self.prj2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, bias=False),
            nn.GELU()
        )
        self.sa = SpatialAttention(kernel_size)
        
        

    def forward(self, x):
        out = self.prj1(x)
        channels_attn = self.ca(out)
        out = out * channels_attn
        out = self.prj2(out)
        spatial_attn = self.sa(out)
        out = out * spatial_attn
        out = out + x
        return out


##########################################################################
##-------- High Frequency Block---------------

class HighFrequencyBlock(nn.Module):
    def __init__(self,
        channels,     
        ffn_expansion_factor=2,
    ):
        super(HighFrequencyBlock, self).__init__()
        hidden_channels = int(channels * ffn_expansion_factor)
        self.scale = nn.Parameter(torch.rand(1, channels, 3, 1, 1))
        
        self.fuse_conv = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.Conv2d(channels, hidden_channels, kernel_size=1),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
        ])
        self.attn = nn.Sequential(*[
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.Sigmoid()
        ])
        
        self.prj = nn.Sequential(*[
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        ])
        self.mask_conv = nn.Sequential(*[
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels),
            nn.Conv2d(channels, 4, kernel_size=1),
            nn.PixelShuffle(2),
            nn.Sigmoid()
        ])

    def forward(self, x):
        b, c, n, h, w = x.shape
        feat = x * self.scale
        feat = feat.sum(dim=2)  # (B, C, H, W)
        out = self.fuse_conv(feat)
        out = self.prj(out * self.attn(out))
        mask = self.mask_conv(feat + out)
        return mask
    
class DWTBlock(nn.Module):
    def __init__(self, channels, ffn_expansion_factor, LayerNorm_type):
        super(DWTBlock, self).__init__()
        self.norm = LayerNorm(channels, LayerNorm_type)
        self.xfm = DWTForward(J=1, mode='zero', wave='haar')   # DWT
        self.ifm = DWTInverse(mode='zero', wave='haar')        # IDWT
        self.high_branch = HighFrequencyBlock(channels, ffn_expansion_factor)
        self.low_branch = LowFrequencyBlock(channels)
        self.prj_conv = nn.Conv2d(channels, channels, 1)
        
    
    def forward(self, x):
        out = self.norm(x)
        x_low, x_high  = self.xfm(out)
        mask = self.high_branch(x_high[0])
        out_low = self.low_branch(x_low)
        out = self.ifm((out_low, x_high))
        out = mask * out
        out = self.prj_conv(out)
        out = out + x
        return out

class Block(nn.Module):
    def __init__(self, 
        channels, 
        ffn_expansion_factor, 
        LayerNorm_type,
    ):
        super(Block, self).__init__()
        self.dwtblock = DWTBlock(channels, ffn_expansion_factor, LayerNorm_type)
    
    def forward(self, x):
        out = self.dwtblock(x)
        return out
    
class UWNet(nn.Module):
    def __init__(self, 
        inp_channels=3,
        out_channels=3,
        dim = 32,
        ffn_expansion_factor = 2,
        stages = 2,
        LayerNorm_type = 'WithBias',   ## Other option 'BiasFree'
    ):
        super(UWNet, self).__init__()
        self.embed = OverlapPatchEmbed(inp_channels, dim)
        self.recon = nn.Conv2d(dim, out_channels, 3, 1, 1)
        self.encoders = []
        self.decoders = []
        
        channels = dim
        for i in range(stages):
            encoder = [
                Block(channels, ffn_expansion_factor, LayerNorm_type),
                Downsample(channels)
            ]
            self.encoders += encoder
            channels = channels * 2
        
        self.middle = Block(channels, ffn_expansion_factor, LayerNorm_type)
        
        for i in reversed(range(stages)):
            channels = channels // 2
            decoder = [
                Upsample(channels * 2),
                nn.Conv2d(channels * 2, channels, 1),
                Block(channels, ffn_expansion_factor, LayerNorm_type)
            ]
            self.decoders += decoder
        
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
    
    def forward(self, x):
        out = self.embed(x)
        encodes = []
        for i in range(0, len(self.encoders), 2):
            out = self.encoders[i](out)
            encodes.append(out)
            out = self.encoders[i+1](out)
        
        out = self.middle(out)
        
        for i in range(0, len(self.decoders), 3):
            out = self.decoders[i](out)
            out = torch.concat([out, encodes[-1]], dim=1)
            out = self.decoders[i+1](out)
            out = self.decoders[i+2](out)
            encodes.pop()

        out = self.recon(out) + x
        
        return out
    
if __name__ == '__main__':
    img = torch.randn((1, 3, 512, 512))

    model = UWNet(ffn_expansion_factor=2, stages=2)
    print(model(img).shape)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Tổng số tham số: {total_params}')
