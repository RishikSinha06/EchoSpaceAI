"""B0 observation-only straight-wall extrapolation; no fitted parameters."""
from collections import deque
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from .spatial import observations


@dataclass(frozen=True)
class WallConfig:
    minimum_run_cells: int = 3
    extension_cells: int = 64

    def __post_init__(self):
        if self.minimum_run_cells < 2 or self.extension_cells < 0:
            raise ValueError("invalid wall extrapolation settings")


def extrapolate_walls(free: np.ndarray, wall: np.ndarray, config: WallConfig = WallConfig()):
    """Extend contiguous horizontal/vertical observed runs, blocked by free cells.

    Flood fill from observed free cells through the resulting barriers defines
    occupancy. With an open outline this can fill the canvas; B0 is deliberately
    a simple axis-aligned heuristic, not a fitted or hidden-shape baseline.
    """
    free, wall = np.asarray(free, bool), np.asarray(wall, bool)
    if free.shape != (64, 64) or wall.shape != free.shape or (free & wall).any():
        raise ValueError("B0 needs disjoint 64x64 observed free/wall masks")
    barrier = wall.copy()
    for transpose in (False, True):
        lines = wall.T if transpose else wall
        free_lines = free.T if transpose else free
        output = barrier.T if transpose else barrier
        for row, line in enumerate(lines):
            indices = np.flatnonzero(line)
            for run in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
                if len(run) < config.minimum_run_cells:
                    continue
                for start, step in ((int(run[0]) - 1, -1), (int(run[-1]) + 1, 1)):
                    for offset in range(config.extension_cells):
                        col = start + step * offset
                        if not 0 <= col < 64 or free_lines[row, col]:
                            break
                        output[row, col] = True
    interior = free.copy()
    queue = deque(zip(*np.nonzero(free)))
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r-1, c), (r+1, c), (r, c-1), (r, c+1)):
            if 0 <= nr < 64 and 0 <= nc < 64 and not barrier[nr, nc] and not interior[nr, nc]:
                interior[nr, nc] = True
                queue.append((nr, nc))
    return (interior | barrier).astype(np.float32), barrier.astype(np.float32)


class WallExtrapolation(nn.Module):
    def __init__(self, config: WallConfig = WallConfig()):
        super().__init__()
        self.config = config

    def forward(self, batch):
        observed = observations(batch)
        result = [extrapolate_walls(x[0], x[1], self.config) for x in observed.detach().cpu().numpy()]
        values = torch.as_tensor(np.asarray(result), device=observed.device).clamp(1e-6, 1-1e-6)
        return {"occupancy_logits": torch.logit(values[:, 0]), "boundary_logits": torch.logit(values[:, 1])}

    def specification(self):
        return {"version": "p7.model.1", "variant": "B0", "config": asdict(self.config)}
