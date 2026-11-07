"""A start from another model's land state maps the soil moisture of NAM urban points onto NMMB Noah's urban range.

Origin (2026-11-06, hindcast station check): WOOF-NAM started from GFS ran the 2 m dewpoint 10-20 F too dry at
KGSO/KRDU/KINT/KCLT in the Jan 2026 wedge (2 m T fine). Those airports sit on urban cells (IGBP 13), where NAM's SFLX
sets SMCWLT = SMCDRY = 0.40 and SMCMAX = 0.45 (module_LS_NOAHLSM.F90:1851-1858). The GFS soil moisture there (0.22)
is below that wilting point, so the cells never evaporated; the NAM analysis holds them at 0.43.
CPU only (numpy), no card. Run: python tests/test_urban_soilw.py
"""
import os
import sys
from pathlib import Path

if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from namgpu.init import init_native_sfc as ins  # noqa: E402

# STAS SOILPARM wilting point and porosity, classes 1-19 (the table init_native_sfc reads; checked below)
WLT = np.array([0.010, 0.028, 0.047, 0.084, 0.061, 0.066, 0.069, 0.120, 0.103, 0.100, 0.126, 0.138, 0.066,
                0.0, 0.006, 0.028, 0.030, 0.006, 0.010], np.float32)
MAX = np.array([0.339, 0.421, 0.434, 0.476, 0.484, 0.439, 0.404, 0.464, 0.465, 0.406, 0.468, 0.468, 0.439,
                1.0, 0.200, 0.421, 0.468, 0.200, 0.339], np.float32)


def raw_case():
    # 1 x 4 points: urban sandy loam (KGSO-like), cropland sandy loam, urban on the water class (0-1), urban over water
    soilw = np.tile(np.array([0.22, 0.22, 0.30, 1.0], np.float32)[None, None, :], (4, 1, 1))
    soilw[3, 0, 0] = 0.50           # wetter than porosity: clipped to the urban SMCMAX
    return {"SOILW": soilw,
            "VGTYP": np.array([[13, 12, 13, 13]], np.int32),
            "SOTYP": np.array([[3, 3, 14, 3]], np.int32),
            "LAND": np.array([[1.0, 1.0, 1.0, 0.0]], np.float32)}


def main() -> int:
    raw = raw_case()
    out, n = ins.soilw_for_source(raw, "gfs", WLT, MAX)
    assert n == 2, n            # two urban land points (points 0 and 2)
    want = 0.40 + 0.05 * (0.22 - 0.047) / (0.434 - 0.047)          # 0.4224
    assert abs(float(out[0, 0, 0]) - want) < 1e-6, out[0, 0, 0]
    assert abs(float(out[3, 0, 0]) - 0.45) < 1e-6, out[3, 0, 0]       # above porosity -> urban SMCMAX
    assert float(out[0, 0, 0]) > 0.40, "urban GFS soil must be above the urban wilting point, else no evaporation"
    assert np.array_equal(out[:, 0, 1], raw["SOILW"][:, 0, 1]), "non-urban point changed"
    assert abs(float(out[0, 0, 2]) - 0.415) < 1e-6, out[0, 0, 2]      # class 14 spans 0-1: 0.30 -> 0.415
    assert np.array_equal(out[:, 0, 3], raw["SOILW"][:, 0, 3]), "water point changed"
    for own in ("native", "parent", None):
        same, m = ins.soilw_for_source(raw, own, WLT, MAX)
        assert same is raw["SOILW"] and m is None, f"{own}: the model's own land state must pass untouched"
    # the table the run reads is the one this test assumes
    try:
        w, m = ins._stas_wlt_max()
    except Exception as e:     # arwen Noah tables not importable in this environment
        print(f"table check skipped: {type(e).__name__}: {e}")
    else:
        assert np.allclose(w[:19], WLT) and np.allclose(m[:19], MAX), (w[:19], m[:19])
    print("urban soil moisture mapping: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
