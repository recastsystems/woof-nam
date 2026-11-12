"""The 0 h radar reflectivity is the microphysics' own: one call on the 0 h state, its REFL_10CM kept and every
other field the call writes put back (drv-dfi-refl-restore). Before this the 0 h map used a fixed rain intercept,
11.3 dB too strong in drizzle against NCEP's own analysis composite (2024-11-05 12Z, columns with rain and no ice).
CPU only."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.steps import drv_dfi_refl_restore as R  # noqa: E402

S3 = (3, 4, 5)


def test_keep_only_refl_puts_every_other_written_field_back():
    rng = np.random.default_rng(0)
    f = {n: rng.random(S3, dtype=np.float32) for n in ("T", "Q", "CW", "F_RAIN", "PD")}
    f["REFL_10CM"] = np.full(S3, -20.0, np.float32)
    tracers = rng.random((2,) + S3, dtype=np.float32)
    f["TRACERS"], f["QC"] = tracers, tracers[1]                  # QC shares memory with its tracer slot
    f["MPRATES"] = np.zeros((7,) + S3, np.float32)               # the state table's rank-4 row
    before = {n: (a, a.copy()) for n, a in f.items()}

    def call():                                                   # what a microphysics call does to the state
        for n in ("T", "Q", "CW", "F_RAIN", "QC"):
            f[n] += 1.0
        f["MPRATES"] = np.ones((1,) + S3, np.float32)             # replaced, as its first call does
        f["RAINNCV"] = np.ones(S3[1:], np.float32)                # added
        f["REFL_10CM"][...] = 33.0

    kept = R.keep_only_refl(f, call)
    assert np.all(f["REFL_10CM"] == 33.0)
    for n, (obj, val) in before.items():
        if n != "REFL_10CM":
            assert f[n] is obj and np.array_equal(f[n], val), n
    assert "RAINNCV" not in f
    assert kept == ["CW", "F_RAIN", "MPRATES", "Q", "QC", "T"]


def test_only_the_ferrier_route_takes_the_microphysics_call():
    def cfg(mp="fer_hires", col=""):
        return SimpleNamespace(microphysics=mp, mp_column=col, raw={})
    assert R.ferrier_route(cfg())
    assert not R.ferrier_route(cfg(mp="thompson"))
    assert not R.ferrier_route(cfg(mp="thompson", col="tempo"))
