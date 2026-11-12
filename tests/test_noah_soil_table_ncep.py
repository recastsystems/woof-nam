"""The Noah soil table the forecast reads is NCEP's operational NMMB SOILPARM.TBL (HMON v3.2.13 fix/nmmb, the same
fix set as the shipped VEGPARM and GENPARM tables), not WRF V3.0's STAS table.

WRF V3.0's table differed in several classes from NCEP's (sand MAXSMC 0.339 against 0.395 and BB 2.79 against
4.05; clay MAXSMC 0.468 against 0.457; organic material, bedrock, playa, lava). The shipped file must be the pinned
NCEP bytes, parsed the way NMMB's SOIL_VEG_GEN_PARM reads it, and the awphys soil-table records
(namgpu/products/land_cloud.py) carry the same values the forecast reads. CPU only."""
import hashlib
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu import paths  # noqa: E402
from namgpu.physics.noah import phys_noah_parm_tables as parm  # noqa: E402
from namgpu.products import land_cloud as LC  # noqa: E402

FIX = paths.FIX / "noah"


def test_soil_table_is_the_pinned_ncep_file():
    pin = re.search(r'"SOILPARM\.TBL": \("SOILPARM\.TBL", "([0-9a-f]{64})"\)',
                    (ROOT / "tools" / "build_noah_fix.py").read_text())
    assert pin, "tools/build_noah_fix.py pins no NCEP SOILPARM.TBL"
    assert hashlib.sha256((FIX / "SOILPARM.TBL").read_bytes()).hexdigest() == pin.group(1)


def test_soil_table_values_are_ncep_nmmb():
    t = parm.parse_soil_table(FIX)
    maxsmc, bb = t.soil["MAXSMC"], t.soil["BB"]
    refsmc, wltsmc = t.soil["REFSMC"], t.soil["WLTSMC"]
    assert abs(float(maxsmc[0]) - 0.395) < 1e-6 and abs(float(bb[0]) - 4.05) < 1e-6        # sand
    assert abs(float(maxsmc[11]) - 0.457) < 1e-6                                             # clay
    # the products' soil-table records read the forecast's own values, class for class (STATSGO classes 1-19)
    rec = LC.soil_table()
    assert rec["MAXSMC"].size == 19
    for k in range(19):
        assert float(rec["MAXSMC"][k]) == float(maxsmc[k]), k
        assert float(rec["REFSMC"][k]) == float(refsmc[k]), k
        assert float(rec["WLTSMC"][k]) == float(wltsmc[k]), k
        assert float(rec["DRYSMC"][k]) == float(t.soil["DRYSMC"][k]), k


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
