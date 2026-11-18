"""The fit path's radiation memory, CPU only (no CUDA, no nvidia-smi): the door's pool cap and the step loop's batch
choice use one per-row-column figure, and that figure covers what a radiation call holds on the current code.

What it prevents: on 2026-11-13 the South Africa 4 km box (215 x 251, IFS 0.1 start) stopped with OutOfMemoryError at
its first radiation call on a 32 GB card. The door capped the pool at 4.68 GiB; the batch choice, sizing with 0.26 MB
per row-column, put 2 windows (62 rows) in it; a call holds 0.33 MB per row-column, so the batch needed about 0.9 GB
more than the cap left (a 4 km IFS box south of the equator)."""
from __future__ import annotations

import pytest

from namgpu.nest import domain
from namgpu.physics.rrtm import rrtm_call

BOX = dict(lat=-26.0, lon=28.5, size_km=1000.0, res_km=4, size_y_km=1000.0)
# measured on the current code (fit path, 1 h, eight radiation calls each): the largest rrtm_call:row_bytes and the
# pool in use when the first call began (used0), South Africa 4 km box, IFS 0.1 2026-11-12 12Z
ROW_BYTES_B1, ROW_BYTES_B2, USED0 = 71_367_136, 70_645_256, 851_211_264
# the forecast pool cap of the failed run (its OutOfMemoryError line: "limit set to: 5,029,080,064 bytes")
OOM_CEILING = 5_029_080_064
GIB = 2 ** 30


@pytest.fixture
def card32(monkeypatch):
    monkeypatch.setattr(domain, "card_total_gib", lambda index=None: 31.8)
    monkeypatch.setattr(domain, "sm_threads", lambda index=None: 170 * 1536)
    monkeypatch.setattr(domain, "card_name", lambda index=None: "test card 31.8 GiB")
    monkeypatch.delenv("NAMGPU_MEMORY_PATH", raising=False)
    monkeypatch.delenv("NAMGPU_SCRATCH_ARENA", raising=False)
    monkeypatch.delenv("NAMGPU_RRTM_POOL_GIB", raising=False)


def test_box_grid_is_the_canary_box():
    g = domain.nest_grid(**BOX)
    assert (g["im"], g["jm"], g["lm"]) == (215, 251, 60)


def test_one_figure_for_door_and_batch_choice():
    assert rrtm_call.ROW_COL_BYTES["fit"] == pytest.approx(domain.VRAM_PATH["fit"]["rrtm_mb_per_row_col"] * 1e6)
    assert rrtm_call.ROW_COL_BYTES["full"] == pytest.approx(domain.VRAM_PATH["full"]["rrtm_mb_per_row_col"] * 1e6)


def test_fit_figure_covers_the_measured_call():
    im = 215
    assert rrtm_call.ROW_COL_BYTES["fit"] >= max(ROW_BYTES_B1, ROW_BYTES_B2) / im


def _count(ceiling, used=USED0, measured=0.0):
    g = domain.nest_grid(**BOX)
    nw = len(rrtm_call.rrtm_chunk.windows(g["jm"], 32))
    return rrtm_call.batch_count(g["jm"], g["im"], 32, nw, True, used=used, held=used, free=20 * GIB,
                                 ceiling=ceiling, measured_row_bytes=measured)


def test_the_failed_run_now_takes_one_window_that_fits(monkeypatch):
    monkeypatch.delenv("NAMGPU_RRTM_POOL_GIB", raising=False)
    b = _count(OOM_CEILING)
    assert b == 1
    assert USED0 + 32 * ROW_BYTES_B1 <= OOM_CEILING             # 3.0 GiB of a 4.68 GiB cap
    # the old figure chose 2 windows, which the measured call cannot hold under that cap
    monkeypatch.setitem(rrtm_call.ROW_COL_BYTES, "fit", 0.26e6)
    assert _count(OOM_CEILING) == 2
    assert USED0 + 62 * ROW_BYTES_B2 > OOM_CEILING


def test_door_cap_holds_the_measured_box(card32):
    e = domain.vram_estimate(domain.nest_grid(**BOX), 32, index=None)
    assert e["memory_path"] == "fit"
    assert e["pool_limit_gib"] >= domain.MEASURED_POOL_GIB["south-4km-ifs-215x251-fit-r32-b1"]
    assert e["total_gib"] >= domain.MEASURED_POOL_GIB["south-4km-ifs-215x251-fit-r32-b1"] + e["lmem_gib"]


@pytest.mark.parametrize("grid", [BOX, dict(lat=38.6, lon=-98.9, size_km=999.0, res_km=3),
                                  dict(lat=38.6, lon=-98.9, size_km=300.0, res_km=3),
                                  dict(lat=24.5, lon=-87.5, size_km=3000.0, res_km=12, size_y_km=2600.0)])
def test_batch_chosen_under_the_door_cap_fits_it(card32, grid):
    """Whatever batch the step loop picks under the door's cap, the measured per-row-column cost of that batch
    fits under the cap (state as the pool in use)."""
    g = domain.nest_grid(**grid)
    e = domain.vram_estimate(g, 32, index=None)
    cap = e["pool_limit_bytes"]
    used = int(e["state_gib"] * GIB)
    nw = len(rrtm_call.rrtm_chunk.windows(g["jm"], 32))
    b = rrtm_call.batch_count(g["jm"], g["im"], 32, nw, True, used=used, held=used, free=20 * GIB, ceiling=cap)
    plan = rrtm_call.batches(g["jm"], 32, b)
    hrows = plan[0][1] - plan[0][0]
    measured_per_row_col = max(ROW_BYTES_B1, ROW_BYTES_B2) / 215
    assert used + hrows * g["im"] * measured_per_row_col <= cap, (grid, b, hrows, cap)
