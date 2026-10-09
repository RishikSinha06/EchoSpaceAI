"""EchoFusion (model D) and its switchable siblings A, P, S and C (plan, Stages 2-3).

One network, five variants, so every comparison shares the same backbone:

| Variant | Geometry | RIR audio | Positions | Fusion |
| --- | --- | --- | --- | --- |
| A | yes | no | no | none (self-attention only) |
| P | yes | no (zeroed) | yes | cross-attention |
| S | no (coordinates only) | yes | yes | learned queries cross-attend |
| C | yes | yes | yes | pooled + concat at the bottleneck |
| D | yes | yes | yes | cross-attention (EchoFusion) |

Shapes (B batch, K <= 8 measurements, T = 1280 samples):

- spatial input ``B x 5 x 64 x 64``: observed free, observed wall, observed
  mask M, and x/y cell-centre coordinates in metres (scaled by 12.8 m);
- encoder 32 -> 64 -> 128 -> 128 channels at 64, 32, 16, 8 cells; the 8 x 8
  bottleneck is 64 tokens of width 128 with a 2D Fourier encoding of their
  centres;
- acoustic encoder: shared 1D CNN (kernel 7, stride 2, 32 -> 64 -> 128 -> 128,
  global average pool) per RIR, plus a Fourier-feature MLP of the six source
  and mic coordinates, summed into K tokens;
- 2-3 fusion blocks: cross-attention (spatial queries, acoustic keys/values,
  4 heads, key padding mask for invalid rows) -> self-attention -> feed-forward,
  each pre-norm with a residual;
- decoder: transposed convolutions mirroring the encoder with U-Net skips, two
  logit heads (occupancy, boundary), ``B x 64 x 64`` each.

No forward path reads targets, ``valid_cells``, room identity, coverage or
split. Positions are P6 grid-frame metres ``(u, v, h)``. Invalid RIR rows are
zeroed before encoding and masked out of attention and pooling, so they cannot
influence the output.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("A", "P", "S", "C", "D")
GRID = 64
GRID_SIZE_M = 12.8


@dataclass(frozen=True)
class FusionConfig:
    width: int = 128
    channels: tuple[int, int, int, int] = (32, 64, 128, 128)
    audio_channels: tuple[int, int, int, int] = (32, 64, 128, 128)
    fusion_layers: int = 3
    heads: int = 4
    feedforward: int = 256
    fourier_bands: int = 8
    dropout: float = 0.1
    height_scale_m: float = 3.0
    rir_samples: int = 1280
    time_channel: bool = True  # see AcousticEncoder: without it, pooling discards arrival time

    def __post_init__(self) -> None:
        if self.channels[-1] != self.width or self.audio_channels[-1] != self.width:
            raise ValueError("the last encoder stages must produce the token width")
        if self.width % self.heads or not 1 <= self.fusion_layers <= 3:
            raise ValueError("width must divide by heads; 1-3 fusion layers (the plan uses 2 or 3)")
        if not 0 <= self.dropout < 1 or self.height_scale_m <= 0 or self.fourier_bands < 1:
            raise ValueError("invalid dropout, height scale or Fourier bands")


def fourier(values: torch.Tensor, bands: int) -> torch.Tensor:
    """sin/cos at octave frequencies of values already scaled to about [0, 1]."""
    frequencies = (2.0 ** torch.arange(bands, device=values.device, dtype=values.dtype)) * math.pi
    angles = values[..., None] * frequencies
    return torch.cat((torch.sin(angles), torch.cos(angles)), dim=-1).flatten(-2)


def _norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(min(8, channels // 4), channels)


class Stage(nn.Sequential):
    def __init__(self, inputs: int, outputs: int) -> None:
        super().__init__(
            nn.Conv2d(inputs, outputs, 3, padding=1), _norm(outputs), nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1), _norm(outputs), nn.SiLU(),
        )


class AcousticEncoder(nn.Module):
    """Shared 1D CNN per RIR + Fourier position MLP -> one token per measurement.

    The plan's encoder (stride-2 convolutions, then global average pooling over
    time) is translation-invariant in time: an echo at sample 300 and the same
    echo at sample 700 give the same feature vector, so reflection delay, the
    quantity that encodes wall distance, cannot reach the model. A second input
    channel holding normalised time t/T (on by default, ``time_channel``) lets
    the convolutions tie features to arrival time before pooling.
    """

    def __init__(self, config: FusionConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        self.time_channel = config.time_channel
        previous = 2 if config.time_channel else 1
        for channels in config.audio_channels:
            layers += [nn.Conv1d(previous, channels, 7, stride=2, padding=3), nn.GroupNorm(min(8, channels // 4), channels), nn.SiLU()]
            previous = channels
        self.cnn = nn.Sequential(*layers)
        self.bands = config.fourier_bands
        self.height_scale = config.height_scale_m
        self.position = nn.Sequential(nn.Linear(6 * 2 * config.fourier_bands, config.width), nn.SiLU(),
                                      nn.Linear(config.width, config.width))

    def encode_positions(self, source: torch.Tensor, mic: torch.Tensor) -> torch.Tensor:
        scale = source.new_tensor([GRID_SIZE_M, GRID_SIZE_M, self.height_scale])
        coordinates = torch.cat((source / scale, mic / scale), dim=-1)  # B x K x 6
        return self.position(fourier(coordinates, self.bands))

    def encode_audio(self, rir: torch.Tensor) -> torch.Tensor:
        b, k, t = rir.shape
        x = rir.reshape(b * k, 1, t)
        if self.time_channel:
            time = torch.linspace(0, 1, t, device=rir.device, dtype=rir.dtype)
            x = torch.cat((x, time.expand(b * k, 1, t)), dim=1)
        features = self.cnn(x)
        return features.mean(-1).reshape(b, k, -1)


class FusionBlock(nn.Module):
    """Pre-norm cross-attention (optional) -> self-attention -> feed-forward."""

    def __init__(self, config: FusionConfig, cross: bool) -> None:
        super().__init__()
        w = config.width
        self.cross = cross
        if cross:
            self.norm_q, self.norm_kv = nn.LayerNorm(w), nn.LayerNorm(w)
            self.cross_attention = nn.MultiheadAttention(w, config.heads, dropout=config.dropout, batch_first=True)
        self.norm_self = nn.LayerNorm(w)
        self.self_attention = nn.MultiheadAttention(w, config.heads, dropout=config.dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(w)
        self.feedforward = nn.Sequential(nn.Linear(w, config.feedforward), nn.SiLU(), nn.Dropout(config.dropout),
                                         nn.Linear(config.feedforward, w))

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor | None, padding: torch.Tensor | None) -> torch.Tensor:
        if self.cross:
            keys = self.norm_kv(memory)
            attended, _ = self.cross_attention(self.norm_q(tokens), keys, keys, key_padding_mask=padding, need_weights=False)
            tokens = tokens + attended
        x = self.norm_self(tokens)
        attended, _ = self.self_attention(x, x, x, need_weights=False)
        tokens = tokens + attended
        return tokens + self.feedforward(self.norm_ff(tokens))


class EchoFusion(nn.Module):
    def __init__(self, variant: str = "D", config: FusionConfig = FusionConfig()) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        self.variant, self.config = variant, config
        c1, c2, c3, c4 = config.channels
        w = config.width
        self.enc1, self.enc2, self.enc3, self.enc4 = Stage(5, c1), Stage(c1, c2), Stage(c2, c3), Stage(c3, c4)
        self.token_position = nn.Linear(2 * 2 * config.fourier_bands, w)
        uses_audio = variant in ("P", "S", "C", "D")
        if uses_audio:
            self.acoustic = AcousticEncoder(config)
        if variant == "S":
            self.queries = nn.Parameter(torch.randn(GRID * GRID // 64, w) * 0.02)
        if variant == "C":
            self.concat = nn.Conv2d(2 * w, w, 1)
        cross = variant in ("P", "S", "D")
        self.blocks = nn.ModuleList(FusionBlock(config, cross) for _ in range(config.fusion_layers))
        self.token_norm = nn.LayerNorm(w)
        self.up4, self.dec4 = nn.ConvTranspose2d(w, c3, 2, stride=2), Stage(2 * c3, c3)
        self.up3, self.dec3 = nn.ConvTranspose2d(c3, c2, 2, stride=2), Stage(2 * c2, c2)
        self.up2, self.dec2 = nn.ConvTranspose2d(c2, c1, 2, stride=2), Stage(2 * c1, c1)
        self.head = nn.Conv2d(c1, 2, 1)

    # -- inputs -------------------------------------------------------------
    @staticmethod
    def _coordinates(batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        centres = (torch.arange(GRID, device=device, dtype=dtype) + 0.5) / GRID  # metres / 12.8
        rows, cols = torch.meshgrid(centres, centres, indexing="ij")
        return torch.stack((cols, rows))[None].expand(batch_size, -1, -1, -1)

    def _spatial_input(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        masks = [batch[k].float() for k in ("observed_free", "observed_wall", "observed_cells")]
        if any(m.ndim != 3 or m.shape[-2:] != (GRID, GRID) for m in masks):
            raise ValueError("observations must be B x 64 x 64")
        geometry = torch.stack(masks, dim=1)
        if self.variant == "S":
            geometry = torch.zeros_like(geometry)  # audio-only: coordinates are the only spatial input
        coordinates = self._coordinates(geometry.shape[0], geometry.device, geometry.dtype)
        return torch.cat((geometry, coordinates), dim=1)

    def _acoustic_tokens(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        valid = batch["rir_valid"].bool()
        source, mic = batch["src_pos"].float(), batch["mic_pos"].float()
        rir = batch["rir"].float()
        if valid.ndim != 2 or source.shape != (*valid.shape, 3) or mic.shape != source.shape:
            raise ValueError("positions must be B x K x 3 with B x K validity")
        if rir.shape[:2] != valid.shape or rir.shape[-1] != self.config.rir_samples:
            raise ValueError(f"rir must be B x K x {self.config.rir_samples}")
        if not valid.any(dim=1).all():
            raise ValueError("every sample needs at least one valid measurement")
        keep = valid[..., None]
        source, mic = torch.where(keep, source, torch.zeros_like(source)), torch.where(keep, mic, torch.zeros_like(mic))
        tokens = self.acoustic.encode_positions(source, mic)
        if self.variant != "P":  # P: positions only, RIR content withheld
            audio = torch.where(keep, rir, torch.zeros_like(rir))
            tokens = tokens + self.acoustic.encode_audio(audio)
        tokens = torch.where(keep, tokens, torch.zeros_like(tokens))
        return tokens, ~valid  # key padding mask: True = ignore

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = self._spatial_input(batch)
        s1 = self.enc1(x)
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        s4 = self.enc4(F.max_pool2d(s3, 2))  # B x w x 8 x 8
        b, w, h, v = s4.shape
        centres = self._coordinates(1, s4.device, s4.dtype)[:, :, 4::8, 4::8]  # 8 x 8 token centres
        position = self.token_position(fourier(centres.flatten(2).transpose(1, 2), self.config.fourier_bands))
        memory, padding = (self._acoustic_tokens(batch) if self.variant != "A" else (None, None))
        if self.variant == "C":
            weights = (~padding).float()[..., None]
            pooled = (memory * weights).sum(1) / weights.sum(1).clamp_min(1)
            s4 = self.concat(torch.cat((s4, pooled[:, :, None, None].expand(-1, -1, h, v)), dim=1))
            memory = padding = None
        if self.variant == "S":
            tokens = self.queries[None].expand(b, -1, -1) + position
        else:
            tokens = s4.flatten(2).transpose(1, 2) + position
        for block in self.blocks:
            tokens = block(tokens, memory, padding)
        z = self.token_norm(tokens).transpose(1, 2).reshape(b, w, h, v)
        y = self.dec4(torch.cat((self.up4(z), s3), 1))
        y = self.dec3(torch.cat((self.up3(y), s2), 1))
        y = self.dec2(torch.cat((self.up2(y), s1), 1))
        logits = self.head(y)
        return {"occupancy_logits": logits[:, 0], "boundary_logits": logits[:, 1]}

    def specification(self) -> dict:
        config = asdict(self.config)
        config["channels"], config["audio_channels"] = list(config["channels"]), list(config["audio_channels"])
        return {"version": "p8.model.1", "family": "EchoFusion", "variant": self.variant, "config": config}


def build(variant: str, config: dict | None = None) -> EchoFusion:
    settings = dict(config or {})
    for key in ("channels", "audio_channels"):
        if key in settings:
            settings[key] = tuple(settings[key])
    return EchoFusion(variant, FusionConfig(**settings))
