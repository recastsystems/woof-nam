"""woof-nam run --products in a folder that already holds a receipt (--resume, or a rerun into the same --out).

The GRIB2 follower stops once <out>/forecast-receipt.json is there and every hour it sees is written. Before the fix
a resumed run (+56 to +84 h of the 2020-10-24 00Z arctic-front case) kept the earlier leg's receipt in its folder,
so its follower stopped after the hours already on disk and +57..84 got no GRIB2. _start_products now sets the old
receipt aside (renamed, kept) and drops the old DONE marker before it starts the follower.
CPU only, no card: the follower process is not started (subprocess.Popen is replaced).
Run: python tests/test_products_stale_receipt.py
"""
import os
import sys
import tempfile
from pathlib import Path

if not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from namgpu import run_forecast  # noqa: E402


def test_stale_receipt_is_set_aside():
    started = []
    real = run_forecast.subprocess.Popen
    run_forecast.subprocess.Popen = lambda *a, **k: started.append(a)
    try:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            (out / "products").mkdir()
            (out / "forecast-receipt.json").write_text('{"leg": 2}\n')
            (out / "products" / "DONE").write_text("{}\n")
            run_forecast._start_products(out, "auto")
            assert started, "the follower was not started"
            assert not (out / "forecast-receipt.json").exists(), "the earlier receipt would stop the new follower"
            kept = list(out.glob("forecast-receipt.before-*.json"))
            assert len(kept) == 1 and kept[0].read_text() == '{"leg": 2}\n', kept
            assert not (out / "products" / "DONE").exists()
            # a fresh folder: nothing to set aside
            fresh = out / "fresh"
            run_forecast._start_products(fresh, "auto")
            assert not list(fresh.glob("forecast-receipt*"))
    finally:
        run_forecast.subprocess.Popen = real


if __name__ == "__main__":
    test_stale_receipt_is_set_aside()
    print("test_products_stale_receipt: 1 passed")
