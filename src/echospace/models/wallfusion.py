"""Experimental delay-preserving, first-order wall-guided EchoSpace model.

Planes pass through observation-frame query locations with sampled horizontal
normals. They are hypotheses, not measured walls. Specular finite-segment
visibility is unknown and is not claimed by the infinite-plane evidence layer.
The temporal residual route retains cues from higher-order and non-wall echoes.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .echofusion import EchoFusion, FusionConfig, GRID_SIZE_M, fourier


@dataclass(frozen=True)
class WallConfig(FusionConfig):
    temporal_channels: int = 16
    temporal_tokens: int = 40
    wall_resolution: int = 32
    wall_orientations: int = 32
    wall_chunk: int = 128
    speed_of_sound_m_s: float = 343.0
    sample_rate_hz: int = 16000
    delay_tolerance_ms: float = 0.5
    physics_enabled: bool = True
    multiscale_enabled: bool = True

    def __post_init__(self):
        super().__post_init__()
        if (self.temporal_channels < 4 or not 1 <= self.temporal_tokens <= self.rir_samples // 4
                or self.wall_resolution not in (8, 16, 32, 64) or self.wall_orientations < 4
                or self.wall_chunk < 1 or not 250 <= self.speed_of_sound_m_s <= 400
                or (self.rir_samples//4) % self.temporal_tokens != 0
                or self.sample_rate_hz != 16000 or not 0 <= self.delay_tolerance_ms <= 5):
            raise ValueError('invalid wall/temporal configuration')


def reflection_delays(source, microphone, points, normals, speed=343.0):
    """B,K,3 poses; Q,2 points; N,2 normals -> B,K,Q,N seconds and sidedness.

Uses full 3D distances to vertical infinite planes. Sources/receivers on
opposite sides have no physical same-side first-order reflection support.
"""
    rho = points @ normals.T
    ds = torch.einsum('bki,ni->bkn', source[..., :2], normals)[:, :, None] - rho
    dm = torch.einsum('bki,ni->bkn', microphone[..., :2], normals)[:, :, None] - rho
    image_xy = source[:, :, None, None, :2] - 2 * ds[..., None] * normals
    horizontal = image_xy - microphone[:, :, None, None, :2]
    height = source[..., 2] - microphone[..., 2]
    distance = (horizontal.square().sum(-1) + height[:, :, None, None].square()).clamp_min(1e-12).sqrt()
    return distance / speed, (ds * dm > 0)


def sample_delay_features(features, delays, rate_hz=16000, stride=4):
    """Interpolate B,K,C,T features at B,K,Q,N emission-relative delays.

Two symmetric padding convolutions (stride 2 each) place feature i at original
sample 4*i. No endpoint linspace approximation or direct-arrival realignment.
"""
    b, k, c, length = features.shape
    x = delays * rate_hz / stride
    # Gather-based interpolation avoids grid_sample's nondeterministic CUDA
    # backward. Keep physical calculations in float32 during AMP.
    with torch.autocast(device_type=features.device.type, enabled=False):
        flat = features.float().reshape(b*k, c, length)
        lower = x.floor().long()
        fraction = x.float() - lower
        def gather(index):
            supported = (index >= 0) & (index < length)
            indices = index.clamp(0, length-1).reshape(b*k, 1, -1).expand(-1, c, -1)
            return flat.gather(2, indices).reshape(b, k, c, *x.shape[-2:]) * supported[:, :, None]
        sampled = gather(lower) * (1-fraction[:, :, None]) + gather(lower+1) * fraction[:, :, None]
    return sampled


class TemporalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config.temporal_channels
        self.cnn = nn.Sequential(nn.Conv1d(2, c, 7, stride=2, padding=3), nn.SiLU(),
                                 nn.Conv1d(c, c, 5, stride=2, padding=2), nn.SiLU())
        self.position = nn.Sequential(nn.Linear(6*2*config.fourier_bands, config.width), nn.SiLU(),
                                      nn.Linear(config.width, config.width))
        self.project = nn.Linear(c+1, config.width)
        self.config = config

    def forward(self, rir, source, mic):
        b, k, t = rir.shape
        time = torch.arange(t, device=rir.device, dtype=rir.dtype) / self.config.sample_rate_hz
        x = torch.stack((rir, (time / (t/self.config.sample_rate_hz)).expand_as(rir)), 2)
        features = self.cnn(x.reshape(b*k, 2, t)).reshape(b, k, self.config.temporal_channels, -1)
        pooled = F.avg_pool1d(features.flatten(0, 1), features.shape[-1]//self.config.temporal_tokens)
        # Local windows, not one global token; explicit window-centre time.
        centres = ((torch.arange(self.config.temporal_tokens, device=rir.device) + .5)
                   / self.config.temporal_tokens).to(pooled.dtype)
        pooled = torch.cat((pooled, centres.expand(b*k, 1, -1)), 1).transpose(1, 2)
        scale = source.new_tensor([GRID_SIZE_M, GRID_SIZE_M, self.config.height_scale_m])
        positions = self.position(fourier(torch.cat((source/scale, mic/scale), -1), self.config.fourier_bands))
        tokens = self.project(pooled).reshape(b, k, self.config.temporal_tokens, self.config.width)
        return features, tokens + positions[:, :, None]


class WallEvidence(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        size = config.wall_resolution
        v = (torch.arange(size) + .5) * (GRID_SIZE_M / size)
        yy, xx = torch.meshgrid(v, v, indexing='ij')
        angles = torch.arange(config.wall_orientations) * math.pi / config.wall_orientations
        self.register_buffer('points', torch.stack((xx, yy), -1).reshape(-1, 2))
        self.register_buffer('normals', torch.stack((angles.cos(), angles.sin()), -1))
        self.score = nn.Conv2d(4, 1, 1)
        self.project = nn.Conv2d(4, config.temporal_channels, 1, bias=False)

    def forward(self, rir, source, mic, valid):
        # Exact sample-time evidence: signed waveform and three fixed envelope
        # scales. Learned temporal features remain in the residual attention
        # branch. No large gather backward through an acoustic CNN is needed.
        b, k, length = rir.shape
        wave = rir.float().reshape(b*k, 1, length)
        features = torch.cat([wave] + [F.avg_pool1d(wave.abs(), n, stride=1, padding=n//2)
                                      for n in (5, 17, 65)], 1).reshape(b, k, 4, length)
        pieces = []
        b, k, channels, _ = features.shape
        with torch.autocast(device_type=features.device.type, enabled=False):
            source, mic = source.float(), mic.float()
            for points in self.points.split(self.config.wall_chunk):
                delays, same_side = reflection_delays(source, mic, points, self.normals,
                                                       self.config.speed_of_sound_m_s)
                evidence = 0
                tolerance = self.config.delay_tolerance_ms / 1000
                for shift in (-tolerance, 0, tolerance):
                    evidence = evidence + sample_delay_features(features, delays + shift, stride=1) / 3
                legal = same_side & valid[:, :, None, None] & (delays >= 0) & (
                    delays <= (self.config.rir_samples-1)/self.config.sample_rate_hz)
                # Align across pairs for each shared plane BEFORE mixing
                # orientations. Distinct orientations cannot independently
                # explain each pair and then claim a single agreed wall.
                pooled = (evidence * legal[:, :, None]).sum(1) / valid.sum(1).clamp_min(1)[:, None, None, None]
                scores = self.score(pooled).squeeze(1)
                # Null evidence has weight one, so no support -> exactly zero.
                weights = scores.clamp(-10, 10).exp() * legal.any(1)
                numerator = (pooled * weights[:, None]).sum(-1)
                denominator = 1 + weights.sum(-1)
                pieces.append(numerator / denominator[:, None])
        field = torch.cat(pieces, -1).reshape(b, channels, self.config.wall_resolution, self.config.wall_resolution)
        return self.project(field)


class WallFusion(EchoFusion):
    def __init__(self, variant='D', config=WallConfig()):
        if variant not in ('P', 'D'):
            raise ValueError('WallFusion supports D and its matched positions-only P control')
        super().__init__(variant, config)
        self.acoustic = TemporalEncoder(config)
        self.wall = WallEvidence(config)
        self.wall8 = nn.Conv2d(config.temporal_channels, config.width, 1, bias=False)
        self.wall16 = nn.Conv2d(config.temporal_channels, config.channels[2], 1, bias=False)
        self.gate8 = nn.Conv2d(config.width*2, config.width, 1)
        self.gate16 = nn.Conv2d(config.channels[2]*2, config.channels[2], 1)

    def forward(self, batch):
        valid = batch['rir_valid'].bool()
        source, mic, rir = batch['src_pos'].float(), batch['mic_pos'].float(), batch['rir'].float()
        if (valid.ndim != 2 or source.shape != (*valid.shape, 3) or mic.shape != source.shape
                or rir.shape != (*valid.shape, self.config.rir_samples)):
            raise ValueError('invalid batched RIR/position shapes')
        keep = valid[..., None]
        source = torch.where(keep, source, torch.zeros_like(source))
        mic = torch.where(keep, mic, torch.zeros_like(mic))
        rir = torch.where(keep, rir, torch.zeros_like(rir)) if self.variant == 'D' else torch.zeros_like(rir)
        features, tokens = self.acoustic(rir, source, mic)
        features = torch.where(valid[:, :, None, None], features, torch.zeros_like(features))
        tokens = torch.where(valid[:, :, None, None], tokens, torch.zeros_like(tokens))
        b, k, nt, width = tokens.shape
        # Always-valid null token prevents all-masked attention for K=0.
        memory = torch.cat((tokens.reshape(b, k*nt, width), tokens.new_zeros(b, 1, width)), 1)
        padding = torch.cat((~valid[:, :, None].expand(-1, -1, nt).reshape(b, -1),
                             torch.zeros(b, 1, device=valid.device, dtype=torch.bool)), 1)
        wall = (self.wall(rir, source, mic, valid) if self.config.physics_enabled else
                features.new_zeros(b, self.config.temporal_channels, self.config.wall_resolution, self.config.wall_resolution))
        s1 = self.enc1(self._spatial_input(batch))
        s2 = self.enc2(F.max_pool2d(s1, 2))
        s3 = self.enc3(F.max_pool2d(s2, 2))
        if self.config.multiscale_enabled:
            e16 = self.wall16(resize_evidence(wall, 16))
            s3 = s3 + torch.sigmoid(self.gate16(torch.cat((s3, e16), 1))) * e16
        s4 = self.enc4(F.max_pool2d(s3, 2))
        e8 = self.wall8(resize_evidence(wall, 8))
        s4 = s4 + torch.sigmoid(self.gate8(torch.cat((s4, e8), 1))) * e8
        position = self.token_position(fourier(self._coordinates(1, s4.device, s4.dtype)[:, :, 4::8, 4::8]
                                               .flatten(2).transpose(1, 2), self.config.fourier_bands))
        z = s4.flatten(2).transpose(1, 2) + position
        for block in self.blocks:
            z = block(z, memory, padding)
        z = self.token_norm(z).transpose(1, 2).reshape(b, width, 8, 8)
        y = self.dec4(torch.cat((self.up4(z), s3), 1))
        y = self.dec3(torch.cat((self.up3(y), s2), 1))
        y = self.dec2(torch.cat((self.up2(y), s1), 1))
        logits = self.head(y)
        return {'occupancy_logits': logits[:, 0], 'boundary_logits': logits[:, 1]}

    def specification(self):
        config = asdict(self.config)
        config['channels'], config['audio_channels'] = list(config['channels']), list(config['audio_channels'])
        return {'version': 'wallfusion.model.1', 'family': 'WallFusion', 'variant': self.variant, 'config': config}


def build(variant='D', config=None):
    settings = dict(config or {})
    for key in ('channels', 'audio_channels'):
        if key in settings:
            settings[key] = tuple(settings[key])
    return WallFusion(variant, WallConfig(**settings))


def resize_evidence(wall, size):
    """Integer-grid mean/nearest resize with deterministic CUDA backward."""
    source = wall.shape[-1]
    if source >= size:
        return F.avg_pool2d(wall, source//size) if source != size else wall
    scale = size//source
    return wall.repeat_interleave(scale, -1).repeat_interleave(scale, -2)
