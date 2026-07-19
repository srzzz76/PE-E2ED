import torch
import torch.nn as nn
import torch.nn.functional as F


def gradient_x(value):
    return value[:, :, :, 1:] - value[:, :, :, :-1]


def gradient_y(value):
    return value[:, :, 1:, :] - value[:, :, :-1, :]


class PhaseLoss(nn.Module):
    def forward(self, phase_prediction, phase_target, mask, sin_prediction,
                cos_prediction, sin_target, cos_target, uncertainty, config):
        sin_loss = F.l1_loss(sin_prediction * mask, sin_target * mask)
        cos_loss = F.l1_loss(cos_prediction * mask, cos_target * mask)
        unit_error = torch.abs(sin_prediction.square() + cos_prediction.square() - 1.0)
        unit_loss = (unit_error * mask).sum() / (mask.sum() + 1e-6)

        variance = torch.clamp(uncertainty, min=1e-4, max=1.0)
        residual = (sin_prediction - sin_target).square() + (cos_prediction - cos_target).square()
        nll = residual / (2 * variance) + 0.5 * torch.log(variance)
        bayesian_loss = (nll * mask).sum() / (mask.sum() + 1e-6)

        if config["w_l1"] > 0:
            phase_l1 = F.l1_loss(phase_prediction * mask, phase_target * mask)
            gradient_loss = F.l1_loss(gradient_x(phase_prediction * mask), gradient_x(phase_target * mask)) + F.l1_loss(gradient_y(phase_prediction * mask), gradient_y(phase_target * mask))
            tv_loss = torch.mean(torch.abs(gradient_x(phase_prediction) * mask[:, :, :, 1:])) + torch.mean(torch.abs(gradient_y(phase_prediction) * mask[:, :, 1:, :]))
        else:
            phase_l1 = gradient_loss = tv_loss = torch.tensor(0.0, device=variance.device)

        total = config["w_l1"] * phase_l1 + config["w_grad"] * gradient_loss + config["w_tv"] * tv_loss + config["w_sc"] * (sin_loss + cos_loss) + config["w_nll"] * bayesian_loss + config["w_unit"] * unit_loss
        active_pixels = mask.sum() + 1e-6
        metrics = {
            "total": total.item(), "l1": phase_l1.item(),
            "sc": (sin_loss + cos_loss).item(), "nll": bayesian_loss.item(),
            "nll_term": ((residual / (2 * variance)) * mask).sum().item() / active_pixels.item(),
            "logvar_term": ((0.5 * torch.log(variance)) * mask).sum().item() / active_pixels.item(),
            "unit": unit_loss.item(),
        }
        return total, metrics
