"""The nest couplers interpolate on the space ratio, which is not the time ratio (CPU only).

A 1.5 km nest in a 12 km parent is space ratio 8 and time ratio 12 (25 s / 2+1/12 s); NAM's fire-weather nest is space 2
and time 3 in its 3 km parent. Before the fix the couplers took the time ratio for both, and a 12 -> 1.5 km nest read
its boundary values from a parent box 2/3 of its size.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from namgpu.nest import coupled, twoway


def _cfg(dlmd, ratio=None):
    raw = {} if ratio is None else {"parent_child_space_ratio": str(ratio)}
    return SimpleNamespace(raw=raw, dlmd=dlmd)


def test_space_ratio_from_configfile_key():
    assert twoway.space_ratio(_cfg(0.126), _cfg(0.01575, 8)) == 8


def test_space_ratio_from_spacing_when_no_key():
    assert twoway.space_ratio(_cfg(0.0315), _cfg(0.01575)) == 2


def test_space_ratio_refuses_a_fraction():
    with pytest.raises(SystemExit):
        twoway.space_ratio(_cfg(0.126), _cfg(0.05))


@pytest.mark.parametrize("sratio", [2, 4, 8])
def test_boundary_window_spans_the_nest(sratio):
    # a nest of 8 * 20 + 1 points on a side, SW corner on parent H (11, 11): its boundary stencils must reach the
    # parent column that lies under the nest's east edge, i_ps + (im - 1) / sratio, and not stop short of it
    imc = jmc = 8 * 20 + 1
    imp = jmp = 11 + (imc - 1) // sratio + 12
    t = coupled.tables(imp, jmp, jmp, imc, jmc, jmc, 11, 11, sratio, 5, 5, 1, 0)
    east = 11 + (imc - 1) // sratio          # 1-based parent column of the nest's east edge
    assert t["i0w"] + t["iww"] - 1 >= east - 1
    assert t["i0w"] + t["iww"] - 1 <= east + 2
    ri, rj, iw, js, w, n = t["ring"]
    assert np.all(np.abs(w.sum(axis=1) - 1) < 1e-5)
