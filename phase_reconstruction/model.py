# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
# from arch_util import LayerNorm2d
# from basicsr.models.archs.local_arch import Local_Base
from .layers import LargeKernelBlock, switch_model_to_deploy
import numpy as np
import torch_dct as dct

class UncertaintyHead(nn.Module):
    def __init__(self, in_ch, mid=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 3, padding=1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)) + 0.01



class DifferentiableUnwrapWLS(nn.Module):
    def __init__(self, iter=1):
        super().__init__()
        self.iter = iter

    def forward(self, phi_wrapped, bayesian_weight):
        B, C, H, W = phi_wrapped.shape
        device = phi_wrapped.device
        dx = phi_wrapped[:, :, :, 1:] - phi_wrapped[:, :, :, :-1]
        dx = torch.cat([dx, torch.zeros((B, C, H, 1), device=device)], dim=-1)
        dx = torch.remainder(dx + np.pi, 2 * np.pi) - np.pi
        dy = phi_wrapped[:, :, 1:, :] - phi_wrapped[:, :, :-1, :]
        dy = torch.cat([dy, torch.zeros((B, C, 1, W), device=device)], dim=-2)
        dy = torch.remainder(dy + np.pi, 2 * np.pi) - np.pi

        wx = bayesian_weight[:, :, :, :-1] * bayesian_weight[:, :, :, 1:]
        wx = torch.cat([wx, bayesian_weight[:, :, :, -1:]], dim=-1)
        wy = bayesian_weight[:, :, :-1, :] * bayesian_weight[:, :, 1:, :]
        wy = torch.cat([wy, bayesian_weight[:, :, -1:, :]], dim=-2)

        wx_s, wy_s = torch.roll(wx, 1, -1), torch.roll(wy, 1, -2)
        dx_s, dy_s = torch.roll(dx, 1, -1), torch.roll(dy, 1, -2)

        r = (wx * dx - wx_s * dx_s) + (wy * dy - wy_s * dy_s)

        ky = torch.cos(torch.arange(H, device=device) * (np.pi / H)).view(1, 1, H, 1)
        kx = torch.cos(torch.arange(W, device=device) * (np.pi / W)).view(1, 1, 1, W)
        denom = 2 * (kx + ky - 2)
        denom[:, :, 0, 0] = 1.0

        phi_unwrapped = torch.zeros_like(phi_wrapped)
        b0, p = None, None
        for k in range(self.iter):
            z = dct.idct_2d(dct.dct_2d(r) / denom)
            b1 = torch.sum(r * z, dim=(-1, -2), keepdim=True)
            if k == 0:
                p = z
            else:
                p = z + (b1 / (b0 + 1e-9)) * p
            b0 = b1
            p_dx = torch.cat([p[:, :, :, 1:] - p[:, :, :, :-1], torch.zeros((B, C, H, 1), device=device)], -1)
            p_dy = torch.cat([p[:, :, 1:, :] - p[:, :, :-1, :], torch.zeros((B, C, 1, W), device=device)], -2)
            rp = (wx * p_dx - wx_s * torch.roll(p_dx, 1, -1)) + (wy * p_dy - wy_s * torch.roll(p_dy, 1, -2))
            alpha = b1 / (torch.sum(p * rp, dim=(-1, -2), keepdim=True) + 1e-9)
            r, phi_unwrapped = r - alpha * rp, phi_unwrapped + alpha * p

        return phi_unwrapped


class PhaseRefineNAF(nn.Module):
    def __init__(self, in_channel=3, width=16, num_block=2):
        super().__init__()
        self.intro = nn.Conv2d(in_channel, width, 3, padding=1)
        self.blocks = nn.Sequential(*[NAFBlock(width) for _ in range(num_block)])
        self.out_conv = nn.Conv2d(width, 1, 3, padding=1)

    def forward(self, phi0_norm, sin_phi, cos_phi):
        x = torch.cat([phi0_norm, sin_phi, cos_phi], dim=1)
        return phi0_norm + self.out_conv(self.blocks(self.intro(x)))

class EndToEndPhaseNet(nn.Module):
    def __init__(self, backbone, width=16, num_blocks=4):
        super().__init__()
        self.backbone = backbone
        self.unwrapper = DifferentiableUnwrapWLS(iter=10)
        self.refine = PhaseRefineNAF(width=width, num_block=num_blocks)
        self.unc_head = UncertaintyHead(in_ch=2, mid=32)

    def forward(self, I, mask):
        sin_cos = self.backbone(I)
        sin_phi = sin_cos[:, 0:1]
        cos_phi = sin_cos[:, 1:2]

        # Stop uncertainty gradients from changing the backbone.
        uncertainty = self.unc_head(torch.cat([sin_phi, cos_phi], dim=1).detach())
        uncertainty = uncertainty * mask

        bayesian_weight = torch.exp(-uncertainty / 0.5) * mask

        phi_wrapped = torch.atan2(sin_phi, cos_phi) * mask
        phi0 = self.unwrapper(phi_wrapped, bayesian_weight)
        
        phi0_norm = ((phi0 + 40.0) / 80.0) * mask

        phi_pred = self.refine(phi0_norm, sin_phi, cos_phi) * mask

        return (
            phi_pred * mask,
            phi0_norm * mask,
            sin_phi * mask,
            cos_phi * mask,
            phi_wrapped * mask,
            uncertainty * mask
        )

    def switch_to_deploy(self):
        """Fuse reparameterizable branches in the complete inference model."""
        return switch_model_to_deploy(self)
class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.05):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, 1, bias=True)

        # conv2: dilated 3x3 + 3x3 stacked (effective larger receptive field)
        self.conv2 = nn.Sequential(
            nn.Conv2d(dw_channel, dw_channel, 3, padding=2, dilation=2, groups=dw_channel, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dw_channel, dw_channel, 3, padding=1, groups=dw_channel, bias=True)
        )

        self.conv3 = nn.Conv2d(dw_channel // 2, c, 1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, bias=True),
        )
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, 1, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, 1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.norm1(inp)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class CrossAttention(nn.Module):
    def __init__(self, in_channel=2, c=16):
        super().__init__()

        if in_channel != 2:
            raise ValueError("CrossAttention requires exactly two input frames.")

        self.intro1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            LargeKernelBlock(16, 13),
        )
        self.intro2 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            LargeKernelBlock(16, 13),
        )

        self.NAFBlock1_1 = NAFBlock(16)
        self.NAFBlock1_2 = NAFBlock(16)
        self.NAFBlock2_1 = NAFBlock(16)
        self.NAFBlock2_2 = NAFBlock(16)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(32, 32, kernel_size=1),
        )
        self.norm2 = LayerNorm2d(32)
        self.sg = SimpleGate()
        self.intro = nn.Conv2d(16, c, kernel_size=1)

    def forward(self, inp):
        if inp.shape[1] != 2:
            raise ValueError(
                f"Expected two input frames, got {inp.shape[1]} channels."
            )

        x1 = self.intro1(inp[:, 0:1])
        x2 = self.intro2(inp[:, 1:2])

        x1_1 = self.NAFBlock1_1(x1) + x2
        x1_2 = self.NAFBlock1_2(x2) + x1

        x2_1 = self.NAFBlock2_1(x1_1) + x1_2
        x2_2 = self.NAFBlock2_2(x1_2) + x1_1

        x = torch.cat((x2_1, x2_2), dim=1)
        x = x * self.sca(x)
        x = self.norm2(x)
        x = self.sg(x)
        return self.intro(x)

class TFDNet(nn.Module):

    def __init__(self, in_channel=1, out_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 1],
                 dec_blk_nums=[2, 2, 2, 2]):
        super().__init__()

        self.intro = CrossAttention(in_channel=in_channel, c=width)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, return_feat=False):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)
        feat = x

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)

        if return_feat:
            return x[:, :, :H, :W], feat
        else:
            return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

    def switch_to_deploy(self):
        """Fuse all large-kernel training branches for inference."""
        return switch_model_to_deploy(self)


