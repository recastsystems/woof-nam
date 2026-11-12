"""A start from the archived NAM analysis takes the file's TKE record as the NMMB q2, not as half of it.

Breakage this prevents: NCEP's analysis file carries the NMMB q2 in its TKE record (free-air value 0.02, the EPSQ2 floor of q2). The
native start doubled it, so every run from the NAM's own analysis began with twice the NAM's q2: the +0 h awphys
TKE record was 2.00 times NCEP's at every level (1000 hPa mean 1.085 against 0.542 on 2024-11-05 12Z) and equal
to it from +1 h on.
"""
from __future__ import annotations

import inspect

from namgpu.init import sources


def test_the_native_tke_record_is_q2_itself():
    assert sources.NATIVE_TKE_TO_Q2 == 1.0


def test_the_warm_decode_uses_it_with_the_epsq2_floor():
    src = inspect.getsource(sources.decode_warm)
    assert "cp.maximum(NATIVE_TKE_TO_Q2 * a, cp.float32(0.02))" in src
    assert "2.0 * a" not in src
