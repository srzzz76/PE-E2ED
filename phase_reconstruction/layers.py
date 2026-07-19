import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        _, channels, _, _ = x.size()
        mean = x.mean(1, keepdim=True)
        variance = (x - mean).square().mean(1, keepdim=True)
        normalized = (x - mean) / (variance + eps).sqrt()
        ctx.save_for_backward(normalized, variance, weight)
        return weight.view(1, channels, 1, 1) * normalized + bias.view(1, channels, 1, 1)

    @staticmethod
    def backward(ctx, grad_output):
        _, channels, _, _ = grad_output.size()
        normalized, variance, weight = ctx.saved_tensors
        grad = grad_output * weight.view(1, channels, 1, 1)
        mean_grad = grad.mean(dim=1, keepdim=True)
        mean_grad_normalized = (grad * normalized).mean(dim=1, keepdim=True)
        grad_input = (grad - normalized * mean_grad_normalized - mean_grad) / (variance + ctx.eps).sqrt()
        grad_weight = (grad_output * normalized).sum(dim=(0, 2, 3))
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias, None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SEBlock(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.down = nn.Conv2d(channels, hidden_channels, kernel_size=1)
        self.up = nn.Conv2d(hidden_channels, channels, kernel_size=1)

    def forward(self, x):
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.down(scale))
        scale = torch.sigmoid(self.up(scale))
        return x * scale


class DilatedReparamBlock(nn.Module):
    """Large depthwise convolution with auxiliary dilated branches during training."""

    def __init__(self, channels, kernel_size, deploy=False):
        super().__init__()
        self.lk_origin = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=deploy,
        )
        if not deploy:
            # BatchNorm can be folded into the convolution at deployment time.
            # LayerNorm cannot be used here because its statistics depend on the
            # current input and therefore cannot be represented by fixed weights.
            self.origin_bn = nn.BatchNorm2d(channels)
            if kernel_size == 13:
                self.dil_conv1 = nn.Conv2d(
                    channels, channels, 5, padding=4, dilation=2, groups=channels, bias=False
                )
                self.dil_bn1 = nn.BatchNorm2d(channels)
                self.dil_conv2 = nn.Conv2d(
                    channels, channels, 3, padding=4, dilation=4, groups=channels, bias=False
                )
                self.dil_bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        if not hasattr(self, "origin_bn"):
            return self.lk_origin(x)
        output = self.origin_bn(self.lk_origin(x))
        if hasattr(self, "dil_conv1"):
            output = output + self.dil_bn1(self.dil_conv1(x))
            output = output + self.dil_bn2(self.dil_conv2(x))
        return output

    @staticmethod
    def _fuse_conv_bn(conv, bn):
        """Return the kernel and bias of an equivalent Conv2d."""
        kernel = conv.weight
        if conv.bias is None:
            bias = torch.zeros(
                kernel.size(0), device=kernel.device, dtype=kernel.dtype
            )
        else:
            bias = conv.bias

        std = torch.sqrt(bn.running_var + bn.eps)
        scale = bn.weight / std
        fused_kernel = kernel * scale.reshape(-1, 1, 1, 1)
        fused_bias = bn.bias + (bias - bn.running_mean) * scale
        return fused_kernel, fused_bias

    @staticmethod
    def _expand_dilated_kernel(kernel, dilation):
        """Convert a dilated kernel to an equivalent dense kernel."""
        if dilation == 1:
            return kernel
        size = dilation * (kernel.size(-1) - 1) + 1
        expanded = kernel.new_zeros(kernel.size(0), kernel.size(1), size, size)
        expanded[:, :, ::dilation, ::dilation] = kernel
        return expanded

    @staticmethod
    def _pad_kernel(kernel, target_size):
        padding = target_size - kernel.size(-1)
        if padding < 0 or padding % 2 != 0:
            raise ValueError(
                f"Cannot center a {kernel.size(-1)}x{kernel.size(-1)} kernel "
                f"inside a {target_size}x{target_size} kernel."
            )
        padding //= 2
        return F.pad(kernel, (padding, padding, padding, padding))

    def get_equivalent_kernel_bias(self):
        """Merge all training branches into one depthwise convolution."""
        if not hasattr(self, "origin_bn"):
            return self.lk_origin.weight, self.lk_origin.bias

        kernel, bias = self._fuse_conv_bn(self.lk_origin, self.origin_bn)
        target_size = kernel.size(-1)

        for conv_name, bn_name in (("dil_conv1", "dil_bn1"), ("dil_conv2", "dil_bn2")):
            if not hasattr(self, conv_name):
                continue
            conv = getattr(self, conv_name)
            bn = getattr(self, bn_name)
            branch_kernel, branch_bias = self._fuse_conv_bn(conv, bn)
            branch_kernel = self._expand_dilated_kernel(
                branch_kernel, conv.dilation[0]
            )
            kernel = kernel + self._pad_kernel(branch_kernel, target_size)
            bias = bias + branch_bias

        return kernel, bias

    @torch.no_grad()
    def switch_to_deploy(self):
        """Replace the training branches with their equivalent single conv."""
        if not hasattr(self, "origin_bn"):
            return self

        kernel, bias = self.get_equivalent_kernel_bias()
        origin = self.lk_origin
        reparam_conv = nn.Conv2d(
            origin.in_channels,
            origin.out_channels,
            origin.kernel_size,
            stride=origin.stride,
            padding=origin.padding,
            dilation=origin.dilation,
            groups=origin.groups,
            bias=True,
            padding_mode=origin.padding_mode,
        ).to(device=kernel.device, dtype=kernel.dtype)
        reparam_conv.weight.copy_(kernel)
        reparam_conv.bias.copy_(bias)
        self.lk_origin = reparam_conv

        for name in ("origin_bn", "dil_conv1", "dil_bn1", "dil_conv2", "dil_bn2"):
            if hasattr(self, name):
                delattr(self, name)
        return self


class LargeKernelBlock(nn.Module):
    """Large-kernel residual block based on the UniRepLKNet design."""

    def __init__(
        self,
        dim,
        kernel_size=13,
        drop_path=0.0,
        layer_scale_init_value=1e-6,
        deploy=False,
        with_cp=False,
        use_sync_bn=False,
        ffn_factor=4,
    ):
        super().__init__()
        del use_sync_bn
        self.with_cp = with_cp
        if kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy)
        else:
            self.dwconv = nn.Conv2d(
                dim, dim, kernel_size, padding=kernel_size // 2, groups=dim, bias=deploy
            )
        # This normalization is outside the reparameterized convolution and must
        # remain present in both training and deployment models.
        self.norm = nn.Identity() if kernel_size == 0 else LayerNorm2d(dim)
        self.se = SEBlock(dim, dim // 4)
        hidden_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Conv2d(dim, hidden_dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(hidden_dim, dim, 1)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim))
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _forward(self, x):
        residual = x
        x = self.norm(self.dwconv(x))
        x = self.se(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1) * x
        return residual + self.drop_path(x)

    def forward(self, x):
        if self.with_cp and x.requires_grad:
            return torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=False)
        return self._forward(x)

    def switch_to_deploy(self):
        if isinstance(self.dwconv, DilatedReparamBlock):
            self.dwconv.switch_to_deploy()
        return self


@torch.no_grad()
def switch_model_to_deploy(model):
    """Convert every DilatedReparamBlock in ``model`` in place."""
    model.eval()
    for module in model.modules():
        if isinstance(module, DilatedReparamBlock):
            module.switch_to_deploy()
    return model
