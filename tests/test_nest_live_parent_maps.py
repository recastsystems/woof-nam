"""A live-parent run's domains get the receipt the maps tail reads (CPU only).

`woof-nam nest --concurrent` / `--two-way` runs both domains in one process (namgpu.nest.twoway), which writes one
receipt at <out> and none in <out>/parent or <out>/nest. The maps tail (run_forecast --maps-only) reads and finishes
<run>/forecast-receipt.json, so before the fix every live-parent run stopped at "maps rc 1" with no maps drawn.
"""
import json

import pytest

from namgpu import run_forecast
from namgpu.nest import twoway_door


def test_maps_tail_needs_a_domain_receipt(tmp_path):
    run = tmp_path / "nest"
    run.mkdir()
    with pytest.raises(FileNotFoundError):
        run_forecast.maps_only("2026-11-17_00z", run, tmp_path, columns=False, wait=False, maps=False)


def test_seeded_domain_receipt_is_read_by_the_maps_tail(tmp_path):
    for role in ("parent", "nest"):
        run = tmp_path / role
        run.mkdir()
        p = twoway_door.seed_domain_receipt(run, "woof-nam nest --concurrent", "twoway-receipt.json")
        rec = json.loads(p.read_text())
        assert rec["domain"] == role and rec["exit_code"] == 0
        assert run_forecast.maps_only("2026-11-17_00z", run, tmp_path, columns=False, wait=False, maps=False) == 0


def test_an_existing_receipt_is_kept(tmp_path):
    run = tmp_path / "nest"
    run.mkdir()
    (run / "forecast-receipt.json").write_text(json.dumps({"exit_code": 3, "mine": True}))
    twoway_door.seed_domain_receipt(run, "woof-nam nest --two-way", "twoway-receipt.json")
    assert json.loads((run / "forecast-receipt.json").read_text()) == {"exit_code": 3, "mine": True}
