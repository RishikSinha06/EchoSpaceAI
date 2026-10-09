"""Synthetic rooms where only the sound knows where the far wall is (P8 gate and tests).

Each room is a rectangle on the 64 x 64 grid. The scan sees the left part only
(free cells and the left wall); the right wall's column is hidden and varies
from room to room. Every RIR has a direct-path pulse and a reflection pulse at
the delay of the path source -> right wall -> mic (image source), at 16 kHz,
plus a decaying diffuse tail. Pulses are band-limited (Gaussian, sigma 3
samples) like resampled real RIRs; one-sample spikes were tried first and no
encoder could learn from them, which says nothing about real data.
Geometry alone cannot locate the hidden wall; the reflection delay can. This
is an artificial check that the acoustic path of EchoFusion carries
information. It is not AcousticRooms data and supports no research claim.
"""

from __future__ import annotations

import numpy as np

from echospace.geometry.frames import GridFrame

SPEED_OF_SOUND_M_S = 343.0
RATE_HZ = 16000
CELL_M = 0.2


def synthetic_room(room: int, index: int, seed: int = 0, k: int = 8) -> dict:
    rng = np.random.default_rng([seed, room, index])
    top, bottom = 18, 46
    left = 6 + int(rng.integers(0, 4))
    right = int(rng.integers(26, 60))  # hidden wall column: the quantity only audio reveals
    seen_until = left + 10
    target = np.zeros((64, 64), np.uint8)
    target[top:bottom + 1, left:right + 1] = 1
    boundary = np.zeros_like(target)
    boundary[top, left:right + 1] = boundary[bottom, left:right + 1] = 1
    boundary[top:bottom + 1, left] = boundary[top:bottom + 1, right] = 1
    free = np.zeros_like(target)
    free[top + 1:bottom, left + 1:seen_until] = 1
    wall = np.zeros_like(target)
    wall[top:bottom + 1, left] = 1
    seen = free | wall
    cells = np.argwhere(free)
    poses = np.zeros((8, 2, 3), np.float32)
    rir = np.zeros((8, 1280), np.float32)
    valid = np.zeros(8, np.uint8)
    wall_u = (right + 0.5) * CELL_M
    pulse = np.exp(-0.5 * (np.arange(-12, 13) / 3.0) ** 2).astype(np.float32)
    for row in range(k):
        (sr, sc), (mr, mc) = cells[rng.integers(len(cells))], cells[rng.integers(len(cells))]
        source = np.array([(sc + 0.5) * CELL_M, (sr + 0.5) * CELL_M, 1.5], np.float32)
        mic = np.array([(mc + 0.5) * CELL_M, (mr + 0.5) * CELL_M, 1.2], np.float32)
        image = source.copy()
        image[0] = 2 * wall_u - source[0]
        direct = int(round(np.linalg.norm(source - mic) / SPEED_OF_SOUND_M_S * RATE_HZ))
        for distance, amplitude in ((np.linalg.norm(source - mic), 1.0), (np.linalg.norm(image - mic), 0.6)):
            t = int(round(distance / SPEED_OF_SOUND_M_S * RATE_HZ))
            lo, hi = max(0, t - 12), min(1280, t + 13)
            rir[row, lo:hi] += amplitude * pulse[lo - (t - 12):hi - (t - 12)]
        tail = np.arange(1280 - direct)
        rir[row, direct:] += (rng.normal(0, 0.05, len(tail)) * np.exp(-tail / 400)).astype(np.float32)
        rir[row] += rng.normal(0, 0.01, 1280).astype(np.float32)
        poses[row] = (source, mic)
        valid[row] = 1
    return {
        "sample_id": f"synthetic_room_{room}_mask_{index}", "room_id": f"synthetic_room_{room}", "room_group": room,
        "grid": GridFrame((0.0, 1.1, 0.0), 0.0).manifest_grid(),
        "spatial_transform": {"quarter_turns": 0, "flip_columns": False},
        "occlusion_type": "missing_wall", "coverage_bin": "low", "coverage": 0.3,
        "observed_cells": seen, "observed_free": free, "observed_wall": wall,
        "target_occupancy": target, "target_boundary": boundary, "valid_cells": np.ones_like(target),
        "src_pos": poses[:, 0], "mic_pos": poses[:, 1], "rir_valid": valid, "rir": rir,
    }


def synthetic_splits(train_rooms: int = 48, val_rooms: int = 8, test_rooms: int = 8, masks_per_room: int = 4,
                     seed: int = 0) -> dict[str, list[dict]]:
    """Disjoint synthetic rooms per split (one room = one group)."""
    ranges = {"train": range(0, train_rooms), "val": range(train_rooms, train_rooms + val_rooms),
              "test": range(train_rooms + val_rooms, train_rooms + val_rooms + test_rooms)}
    return {split: [synthetic_room(room, i, seed) for room in rooms for i in range(masks_per_room)]
            for split, rooms in ranges.items()}
