"""woof-nam run --products starts its GRIB2 follower only after the door's case and memory checks.

Breakage it prevents (door-products-3, 2026-11-11): the follower was started first, so a run that stopped at once (a
case folder that does not exist, a card too small for the grid) had started a follower process and then had to stop
it at exit. CPU only: the follower start is a stand-in that records the call; the writer check is skipped."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import exporter  # noqa: E402
from namgpu import run_forecast as rf  # noqa: E402


def _main(argv, root: Path, memory_rc=None):
    started = []
    keep = (rf._start_products, exporter.ensure, rf.memory_check, os.environ.get("NAMGPU_ROOT"))
    rf._start_products = lambda *a, **k: started.append(a)
    exporter.ensure = lambda *a, **k: None
    if memory_rc is not None:
        rf.memory_check = lambda a: memory_rc
    os.environ["NAMGPU_ROOT"] = str(root)
    try:
        try:
            rc = rf.main(argv)
        except SystemExit as e:
            rc = e.code
    finally:
        rf._start_products, exporter.ensure, rf.memory_check = keep[:3]
        if keep[3] is None:
            os.environ.pop("NAMGPU_ROOT", None)
        else:
            os.environ["NAMGPU_ROOT"] = keep[3]
    return rc, started


def test_missing_case_folder_starts_no_follower():
    with tempfile.TemporaryDirectory() as t:
        rc, started = _main(["--case", "2024-05-06_12z", "--source", "native", "--hours", "1", "--out",
                             str(Path(t) / "run"), "--products", "--no-maps", "--no-columns"], Path(t))
        assert rc and "no case folder" in str(rc), rc
        assert not started, "a run that stopped at its case check had started a GRIB2 follower"


def test_memory_refusal_starts_no_follower():
    with tempfile.TemporaryDirectory() as t:
        (Path(t) / "cases" / "2024-05-06_12z").mkdir(parents=True)
        rc, started = _main(["--case", "2024-05-06_12z", "--source", "native", "--hours", "1", "--out",
                             str(Path(t) / "run"), "--products", "--no-maps", "--no-columns"], Path(t), memory_rc=3)
        assert rc == 3, rc
        assert not started, "a run refused by its memory check had started a GRIB2 follower"


if __name__ == "__main__":
    test_missing_case_folder_starts_no_follower()
    test_memory_refusal_starts_no_follower()
    print("test_run_follower_after_checks: 2 passed")
