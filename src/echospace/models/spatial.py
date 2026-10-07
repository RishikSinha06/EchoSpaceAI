"""P7 A/P U-Nets: observations alone, or observations plus paired positions.

No forward path reads targets, valid_cells, waveforms, room identity or split.
Positions are P6 grid-frame metres (column, row, height), not scene coordinates.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SpatialConfig:
    base_channels: int = 16
    grid_size: int = 64
    cell_size_m: float = 0.2
    height_scale_m: float = 3.0

    def __post_init__(self):
        if self.base_channels < 4 or self.grid_size != 64 or self.cell_size_m != 0.2 or self.height_scale_m <= 0:
            raise ValueError("P7 requires base_channels >=4 and the frozen 64x64 / 0.2 m grid")


def observations(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
    values = [batch[k].float() for k in ("observed_free", "observed_wall", "observed_cells")]
    if any(v.ndim != 3 or v.shape[-2:] != (64, 64) for v in values):
        raise ValueError("observations must be Bx64x64")
    result = torch.stack(values, dim=1)
    if not torch.isfinite(result).all() or not ((result == 0) | (result == 1)).all():
        raise ValueError("observations must be finite binary masks")
    return result


class ConvBlock(nn.Sequential):
    def __init__(self, input_channels: int, output_channels: int):
        super().__init__(
            nn.Conv2d(input_channels, output_channels, 3, padding=1),
            nn.GroupNorm(4, output_channels), nn.SiLU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1),
            nn.GroupNorm(4, output_channels), nn.SiLU(),
        )


class SpatialUNet(nn.Module):
    """A: three observation channels. P: same network + masked position set.

    Shared pair MLP preserves source/receiver pairing, then masked mean pooling
    makes order irrelevant. P also rasterizes source/mic densities onto the
    grid to retain location. Invalid rows are zeroed before either operation.
    """
    def __init__(self, variant: str = "A", config: SpatialConfig = SpatialConfig()):
        super().__init__()
        if variant not in ("A", "P"):
            raise ValueError("SpatialUNet variant must be A or P")
        if config.base_channels % 4:
            raise ValueError("base_channels must be divisible by 4 for GroupNorm")
        self.variant, self.config = variant, config
        c = config.base_channels
        self.enc1 = ConvBlock(5 if variant == "P" else 3, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 8)
        if variant == "P":
            self.position_mlp = nn.Sequential(nn.Linear(6, c * 2), nn.SiLU(), nn.Linear(c * 2, c * 2))
            self.position_fusion = nn.Conv2d(c * 10, c * 8, 1)
        self.dec3 = ConvBlock(c * 12, c * 4)
        self.dec2 = ConvBlock(c * 6, c * 2)
        self.dec1 = ConvBlock(c * 3, c)
        self.head = nn.Conv2d(c, 2, 1)

    def _positions(self, batch, geometry):
        valid = batch["rir_valid"]
        source, mic = batch["src_pos"].float(), batch["mic_pos"].float()
        if valid.ndim != 2 or source.shape != (*valid.shape, 3) or mic.shape != source.shape:
            raise ValueError("positions must be BxKx3, with BxK validity")
        if valid.shape[0] != geometry.shape[0] or not ((valid == 0) | (valid == 1)).all():
            raise ValueError("position validity must be binary and match the batch")
        valid = valid.bool()
        poses = torch.stack((source, mic), dim=2)
        poses = torch.where(valid[:, :, None, None], poses, torch.zeros_like(poses))
        if not torch.isfinite(poses).all():
            raise ValueError("valid positions must be finite")
        scale = poses.new_tensor([12.8, 12.8, self.config.height_scale_m])
        pair = (poses / scale).flatten(2)
        encoded = self.position_mlp(pair)
        encoded = torch.where(valid[:, :, None], encoded, torch.zeros_like(encoded))
        pooled = encoded.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        # Broadcasting avoids nondeterministic CUDA scatter_add with repeated cells.
        cells = torch.floor(poses[..., :2] / self.config.cell_size_m)
        if (((cells < 0) | (cells >= 64)) & valid[:, :, None, None]).any():
            raise ValueError("valid positions fall outside the frozen grid")
        rows = torch.arange(64, device=poses.device)[None, None, None, :, None]
        cols = torch.arange(64, device=poses.device)[None, None, None, None, :]
        hits = ((cols == cells[..., 0, None, None]) & (rows == cells[..., 1, None, None])
                & valid[:, :, None, None, None])
        density = hits.float().sum(1) / valid.sum(1).clamp_min(1)[:, None, None, None]
        return density, pooled

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = observations(batch)
        pooled = None
        if self.variant == "P":
            density, pooled = self._positions(batch, x)
            x = torch.cat((x, density), dim=1)
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.enc3(F.max_pool2d(b, 2))
        d = self.bottleneck(F.max_pool2d(c, 2))
        if pooled is not None:
            d = self.position_fusion(torch.cat((d, pooled[:, :, None, None].expand(-1, -1, *d.shape[-2:])), 1))
        up = lambda value, skip: F.interpolate(value, size=skip.shape[-2:], mode="nearest")
        x = self.dec3(torch.cat((up(d, c), c), 1))
        x = self.dec2(torch.cat((up(x, b), b), 1))
        x = self.dec1(torch.cat((up(x, a), a), 1))
        logits = self.head(x)
        return {"occupancy_logits": logits[:, 0], "boundary_logits": logits[:, 1]}

    def specification(self):
        return {"version": "p7.model.1", "variant": self.variant, "config": asdict(self.config)}
