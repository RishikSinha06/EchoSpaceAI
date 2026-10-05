"""Simulated partial scans and the observation-only grid frame (module 4)."""

from .frame import FrameError, dominant_wall_heading, observed_anchor, observed_frame
from .occlusion import (
    COVERAGE_BINS,
    OCCLUSION_TYPES,
    Scan,
    ScanNotApplicable,
    ScanNotFeasible,
    generate_scan,
    l_wing_candidates,
)
from .raycast import RayFan, cast_rays
from .room import RoomGeometry, RoomGeometryError
from .sample import ObservedGrid, ScanSample, build_scan_sample, observe_on_grid

__all__ = [
    "COVERAGE_BINS",
    "OCCLUSION_TYPES",
    "FrameError",
    "ObservedGrid",
    "RayFan",
    "RoomGeometry",
    "RoomGeometryError",
    "Scan",
    "ScanNotApplicable",
    "ScanNotFeasible",
    "ScanSample",
    "build_scan_sample",
    "cast_rays",
    "dominant_wall_heading",
    "generate_scan",
    "l_wing_candidates",
    "observe_on_grid",
    "observed_anchor",
    "observed_frame",
]
