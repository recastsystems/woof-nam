"""The shipped Noah tables give the NAM's snow cover and albedo over snow (NCEP NMMB VEGPARM / GENPARM).

Origin (2026-11-10, Rocky Mountain hindcasts, arctic front of 2020-10-24 00Z): over plains cells with more than
10 cm of snow, WOOF-NAM 0.2.0 wrote a surface albedo of 31.7 % where the archived NAM had 51.9 %, and day 3 ran
too warm over the new snow. The Noah tables were a WRF V3.0 stand-in with SNUP 0.04 m (grass, crop) and
SALP 2.6. NCEP's NMMB tables carry SNUP 0.02 m and SALP 4.0 (HMON v3.2.13 fix/nmmb, US Government data).
SFLX makes the cover with SNUP = 4*SNUP
(module_LS_NOAHLSM.F90:1989), SNFRAC (3941-3943), SNCOVR <= 0.98 (1993) and ALCALC (2309, 2326).
CPU only (numpy), no card. Run: python tests/test_noah_snow_albedo.py
"""
import hashlib
import os
import sys
from pathlib import Path

if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from namgpu import paths  # noqa: E402
from namgpu.physics.noah.phys_noah_parm_tables import parse_tables  # noqa: E402

TBL = paths.FIX / "noah"
PINS = {"VEGPARM.TBL": "4dc8d3ea7a27551bc3ddfbfdc995913bfc2f4aaebf704d420a274a38d1a29cdd",
        "GENPARM.TBL": "7f82cb68458f9a3ba7dcac08a0455edafee4705dcea9e06ad780b3d2ee2e7a31"}
GRASS, CROP = 10, 12   # IGBP classes (1-based)


def albedo(swe_m, snup_tbl, salp, alb, snoalb):
    """SFLX's land branch for SNEQV > 0, in float32 as the Fortran (kfpt single)."""
    f = np.float32
    snup = f(4.0) * f(snup_tbl)
    r = f(swe_m) / snup
    sc = f(1.0) if swe_m >= snup else f(1.0) - (np.exp(-f(salp) * r) - r * np.exp(-f(salp)))
    sc = min(sc, f(0.98))
    a = f(alb) + sc * (f(snoalb) - f(alb))
    return float(min(a, f(snoalb))), float(sc)


def test_tables_are_ncep():
    for name, sha in PINS.items():
        assert hashlib.sha256((TBL / name).read_bytes()).hexdigest() == sha, name
    t = parse_tables(TBL, 1)
    assert t.mminlu == "IGBP"
    assert abs(float(t.veg["SNUPTBL"][GRASS - 1]) - 0.02) < 1e-7
    assert abs(float(t.veg["SNUPTBL"][CROP - 1]) - 0.02) < 1e-7
    assert float(t.gen["SALP_DATA"]) == 4.0
    # the IGBP SNUP list of NCEP's HMON NMMB VEGPARM.TBL (hmon.v3.2.13 fix/nmmb, US Government data)
    snup = [0.080] * 5 + [0.020, 0.020, 0.060, 0.040, 0.020, 0.010, 0.020, 0.020, 0.020, 0.013, 0.013, 0.010,
                          0.020, 0.020, 0.020]
    assert np.allclose(np.asarray(t.veg["SNUPTBL"][:20], dtype=np.float64), snup, atol=1e-7)
    assert int(t.veg["NROTBL"][16]) == 0     # water class: no root layers, read as NCEP wrote it


def test_case_albedo():
    """The case's deep plains snow: our start SWE 16 mm, the archived NAM's 21 mm, grass ALB 0.21 and SNOALB 0.70.
    The shipped tables give 0.48 and 0.53 (archived NAM ALBDO 0.52); the old stand-in gave 0.33, as the run did."""
    t = parse_tables(TBL, 1)
    snup, salp = float(t.veg["SNUPTBL"][GRASS - 1]), float(t.gen["SALP_DATA"])
    ours, _ = albedo(0.016, snup, salp, 0.21, 0.70)
    nam, _ = albedo(0.021, snup, salp, 0.21, 0.70)
    old, _ = albedo(0.016, 0.04, 2.6, 0.21, 0.70)
    assert 0.46 < ours < 0.50, ours
    assert 0.50 < nam < 0.55, nam
    assert 0.31 < old < 0.34, old
    deep, sc = albedo(0.10, snup, salp, 0.21, 0.70)        # past 4*SNUP: full cover, capped at 0.98
    assert sc == np.float32(0.98) and abs(deep - (0.21 + 0.98 * 0.49)) < 1e-6


if __name__ == "__main__":
    test_tables_are_ncep()
    test_case_albedo()
    print("test_noah_snow_albedo: 2 passed")
