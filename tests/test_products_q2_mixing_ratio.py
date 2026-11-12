"""The products hand-off frame carries the 2 m humidity as a mixing ratio.

Breakage this prevents: the history Q2 is NMMB QSHLTR, a specific humidity, and the hand-off frame passed it under the
WRF name Q2, which the WOOF post reads as a mixing ratio and converts to a specific humidity (w / (1 + w)). So the
awphys 2 m SPFH record was q / (1 + q), the 2 m dew point 0.13 K low on average (0.3 K over warm water) and the 2 m RH
1.1 % low against NCEP's awphys file from the same analysis (2024-11-05 12Z, +0 h).
"""
from __future__ import annotations

import inspect

import numpy as np

from namgpu.io import products


def test_frame_q2_is_the_mixing_ratio_of_the_history_specific_humidity():
    q = np.array([[2.0e-5, 0.004, 0.0115, 0.020, 0.0252]], dtype=np.float32)   # the analysis range of QSHLTR
    w = products.frame_q2(q)
    assert w.dtype == np.float32
    # the WOOF post's conversion of the frame value gives the model's specific humidity back, not q / (1 + q)
    back = w.astype(np.float64) / (1.0 + w.astype(np.float64))
    np.testing.assert_allclose(back, q.astype(np.float64), rtol=2e-6)
    # and the frame value is not the specific humidity itself (the defect): 2 % apart at 20 g/kg
    assert abs(float(w[0, 3]) / 0.020 - 1.0) > 0.019


def test_frame_writer_hands_q2_through_frame_q2():
    src = inspect.getsource(products._wrf_frame_write)
    assert 'frame_q2(get(n)) if n == "Q2"' in src
    assert 'put("NAM_Q2_SPECIFIC", S2, get("Q2"))' in src       # the LCL carrier stays a specific humidity
