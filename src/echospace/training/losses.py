"""BCE + soft Dice supervised only on valid, unobserved cells (M=0)."""
import torch
from torch.nn import functional as F


def masked_loss_per_sample(output, batch, bce_weight=1.0, dice_weight=1.0):
    if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
        raise ValueError("loss weights must be nonnegative with a positive sum")
    observed, valid = batch["observed_cells"], batch["valid_cells"]
    if not (((observed == 0) | (observed == 1)).all() and ((valid == 0) | (valid == 1)).all()):
        raise ValueError("loss masks must be binary")
    mask = (observed == 0) & (valid == 1)
    axes = tuple(range(1, mask.ndim))
    count = mask.sum(axes)
    active = count > 0
    total = torch.zeros(len(mask), device=mask.device, dtype=torch.float32)
    for head in ("occupancy", "boundary"):
        logits, target = output[f"{head}_logits"].float(), batch[f"target_{head}"].float()
        if logits.shape != mask.shape or target.shape != mask.shape:
            raise ValueError("heads, targets and masks must share BxHxW shape")
        # Mask before all arithmetic, so observed targets/logits cannot affect
        # loss (including NaNs outside the supervision domain).
        logits = torch.where(mask, logits, torch.zeros_like(logits))
        target = torch.where(mask, target, torch.zeros_like(target))
        if not torch.isfinite(logits).all() or not ((target == 0) | (target == 1)).all():
            raise ValueError("unobserved logits must be finite and targets binary")
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum(axes) / count.clamp_min(1)
        probability = logits.sigmoid() * mask
        target = target * mask
        dice = 1 - (2 * (probability * target).sum(axes) + 1e-6) / (probability.sum(axes) + target.sum(axes) + 1e-6)
        total = total + torch.where(active, bce_weight * bce + dice_weight * dice, torch.zeros_like(dice)) / 2
    return total, active


def masked_bce_dice(output, batch, bce_weight=1.0, dice_weight=1.0):
    values, active = masked_loss_per_sample(output, batch, bce_weight, dice_weight)
    return values.sum() / active.sum().clamp_min(1)
