import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from echospace.training.quality import boundary_quality


def test_boundary_quality_uses_metres_and_has_defined_empty_cases():
    target = np.zeros((8,8),bool)
    target[2,2:6] = True
    perfect = boundary_quality(target,target)
    assert perfect['boundary_f1_20cm']==1 and perfect['boundary_chamfer_m']==0
    shifted = np.roll(target,2,axis=0)
    result = boundary_quality(shifted,target)
    assert result['boundary_f1_20cm']==0
    assert result['boundary_f1_40cm']==1
    assert result['boundary_chamfer_m']==pytest.approx(.4)
    empty = np.zeros_like(target)
    assert boundary_quality(empty,empty)['boundary_f1_20cm']==1
    assert boundary_quality(empty,target)['boundary_f1_40cm']==0
