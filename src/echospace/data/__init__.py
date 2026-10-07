"""Room caches, room-group folds, fixed evaluation masks and the dataset (module 6)."""

from .dataset import K_MAX, DatasetError, EchoSpaceDataset, collate, grid_from_record, spatial_transform, to_grid_metres
from .evalmasks import EVAL_FORMAT, EvalMaskError, content_sha256, load_eval_spec, masks_for_room, read_eval_sample
from .roomcache import CACHE_FORMAT, RoomCache, RoomCacheError, eligible_mask, load_room_cache, save_room_cache
from .splits import RoomInfo, SplitError, load_folds, make_folds, save_folds, validate_folds

__all__ = [
    "CACHE_FORMAT", "EVAL_FORMAT", "K_MAX", "DatasetError", "EchoSpaceDataset", "EvalMaskError", "RoomCache",
    "RoomCacheError", "RoomInfo", "SplitError", "collate", "content_sha256", "eligible_mask", "grid_from_record",
    "load_eval_spec", "load_folds", "load_room_cache", "make_folds", "masks_for_room", "read_eval_sample",
    "save_folds", "save_room_cache", "spatial_transform", "to_grid_metres", "validate_folds",
]
