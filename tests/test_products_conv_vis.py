"""The products hand-off frame carries the convective rain rate at the output time for the writer's visibility.

Breakage this prevents: the BMJ scheme rains without leaving rain water in the column, and the writer (rw_rrfs_upp,
NAM_VIS) built visibility from the column's water only, so a cell under convective rain read the visibility of clear
air. Against NCEP's archived NAM awphys files (2024-11-05 12Z), the cells whose rain was convective in both read
22.9 km in woof-nam and 13.2 km in the NAM at +12 h. The writer now takes
the frame carrier NAM_CONV_PRATE (kg m-2 s-1); this test holds the hand-off side: the carrier's units, and that the
frame writer reads the history CPRATE and writes the carrier with the run's physics step.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from namgpu.io import products


def test_conv_prate_is_the_history_cprate_over_the_physics_step():
    # CPRATE is m of water per physics step; 50 s step (DT 25 s x NPHS 2): 1.389e-5 m per step = 1 mm/h
    step_m = np.float32(1.0e-3 * 50.0 / 3600.0)
    cp = np.array([[0.0, step_m, 10.0 * step_m, -1.0e-9]], dtype=np.float32)
    got = products.conv_prate(cp, {"dt": 25.0, "nphs": 2, "dtq2": 50.0})
    assert got.dtype == np.float32
    np.testing.assert_allclose(got * 3600.0, [[0.0, 1.0, 10.0, 0.0]], rtol=1e-5, atol=1e-7)   # mm/h
    # no rate, or no step: no carrier (the frame then says so in its notes), never a guessed step
    assert products.conv_prate(None, {"dtq2": 50.0}) is None
    assert products.conv_prate(cp, {}) is None


def test_conv_prate_reads_the_step_from_the_runs_configfile(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "nam_configfile_step").write_text("dt_int: 25\ndt_num: 0\ndt_den: 1000\nnphs: 2\n")
    ctx = products._run_ctx(run)
    got = products.conv_prate(np.array([[1.0e-4]], np.float32), ctx)
    np.testing.assert_allclose(float(got[0, 0]), 1.0e-4 * 1000.0 / 50.0, rtol=1e-6)


def test_frame_writer_hands_the_convective_rate_to_the_writer():
    assert products.CONV_RATE == "CPRATE" and products.CONV_CARRIER == "NAM_CONV_PRATE"
    src = inspect.getsource(products.wrf_frame)
    assert "(CONV_RATE,)" in src                                    # the field pool reads the history CPRATE
    src = inspect.getsource(products._wrf_frame_write)
    assert "put(CONV_CARRIER, S2, conv)" in src
    assert "_run_ctx(Path(history).parent)" in src                 # the parent run's step, from its configfile
