"""CPU check of the column store's worker count (namgpu/io/columns.py auto_workers): one per frame, bounded by
cores, host memory and the scratch folder. Run: python tests/test_columns_workers.py (or pytest)."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.io.columns import auto_workers  # noqa: E402

GB = 1 << 30


def _frames(d: Path, n: int, size: int) -> list[Path]:
    out = []
    for i in range(n):
        f = d / f"wrfout_d01_2026-01-23_{i:02d}:00:00"
        with open(f, "wb") as fh:
            fh.truncate(size)                     # sparse: the size is what counts
        out.append(f)
    return out


def test_one_per_frame_on_a_big_host():
    with tempfile.TemporaryDirectory() as d:
        fr = _frames(Path(d), 13, 9 * GB)
        assert auto_workers(fr, 96, None, mem_available=900 * GB, scratch_free=450 * GB) == 13


def test_bounded_by_memory_scratch_and_cores():
    with tempfile.TemporaryDirectory() as d:
        fr = _frames(Path(d), 13, 9 * GB)
        # the rule of auto_workers since a2c155a: a worker counts one WRF view (9 GB here, given, since the sparse
        # frames have no header to size it from) of 60 % of the host memory and one view of 90 % of the scratch
        # folder's free room; the scratch bound needs a folder (a call with none has no room to count)
        v = 9 * GB
        assert auto_workers(fr, 96, None, mem_available=20 * GB, scratch_free=450 * GB, view=v) == 1
        assert auto_workers(fr, 96, None, mem_available=64 * GB, scratch_free=450 * GB, view=v) == 4
        assert auto_workers(fr, 96, d, mem_available=900 * GB, scratch_free=40 * GB, view=v) == 4
        assert auto_workers(fr, 96, None, mem_available=900 * GB, scratch_free=40 * GB, view=v) == 13
        assert auto_workers(fr, 8, None, mem_available=900 * GB, scratch_free=450 * GB, view=v) == 8
        assert auto_workers([], 8, None) == 1


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("test_columns_workers: PASS")
