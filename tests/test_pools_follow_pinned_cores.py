"""Worker pools are sized from the cores the process may use (namgpu.cores.usable: the affinity set, capped by the
cgroup quota), not from the host's core count.

Breakage it prevents: under `taskset -c 0-11` on a 24-core host the column store planned its workers and their threads
for 24 cores, the history writer pool and the GRIB2 follower's field pool were sized for 24 too, and a forecast sharing
the node with other jobs oversubscribed its 12 cores (2026-11-11). CPU only: the host is made to look like 24 cores
with 12 allowed."""
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import cores  # noqa: E402
from namgpu.io import columns, io_history, products  # noqa: E402


@pytest.fixture
def pinned(monkeypatch):
    """A 24-core host on which this process is pinned to 12 cores. Returns what cores.usable() then says: 12, or
    fewer where the test itself runs under a cgroup quota."""
    monkeypatch.setattr(os, "cpu_count", lambda: 24)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(12)), raising=False)
    cores.usable.cache_clear()
    n = cores.usable()
    assert 1 <= n <= 12, n
    yield n
    cores.usable.cache_clear()


def test_history_writers_count_the_pinned_cores(pinned, monkeypatch):
    monkeypatch.delenv("NAMGPU_HISTORY_WRITERS", raising=False)
    assert io_history.writers() == max(1, min(4, pinned // 4))     # 3 for 12 cores; the host count gave 4


def test_products_field_pool_counts_the_pinned_cores(pinned):
    assert products.field_workers() == max(2, min(16, pinned // 2))     # 6 for 12 cores; the host count gave 12


def test_column_store_plans_for_the_pinned_cores(pinned, tmp_path, monkeypatch):
    """columns.write hands auto_workers the usable cores and gives each worker its share of them."""
    seen = {}

    def auto_workers(frames, ncores, scratch, **kw):
        seen["cores"] = ncores
        raise RuntimeError("stop here: the plan is all this test reads")

    monkeypatch.setattr(columns, "auto_workers", auto_workers)
    monkeypatch.setattr(columns, "view_bytes", lambda frames: 1)
    frame = tmp_path / "wrfout_d01_2020-10-24_00:00:00"
    frame.write_bytes(b"x")
    with pytest.raises(RuntimeError, match="stop here"):
        columns.write([frame], tmp_path / "columns", scratch=str(tmp_path), exe="true", log=lambda m: None)
    assert seen["cores"] == pinned, seen


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
