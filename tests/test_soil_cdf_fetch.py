"""woof-nam fetch-soil-cdf: installs a pinned table, refuses one whose bytes differ, and NAMGPU_SOIL_CDF=auto finds it."""
import hashlib
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from namgpu.init import soil_cdf_fetch as f
from namgpu.init import soil_start


def _pins(td, blob):
    p = Path(td) / "pins.json"
    p.write_text(json.dumps({"repo": "x/y", "release": "v0", "files": {
        "cdf-03.npz": {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}}}))
    return p


def test_install_check_and_auto():
    with tempfile.TemporaryDirectory() as td:
        blob = b"table bytes" * 100
        src, home = Path(td) / "src", Path(td) / "home"
        src.mkdir()
        (src / "cdf-03.npz").write_bytes(blob)
        old_pins, old_env = f.PINS_FILE, {k: os.environ.get(k) for k in ("NAMGPU_SOIL_CDF_HOME", "NAMGPU_ROOT", "NAMGPU_SOIL_CDF")}
        f.PINS_FILE = _pins(td, blob)
        try:
            os.environ["NAMGPU_SOIL_CDF_HOME"] = str(home)
            os.environ["NAMGPU_ROOT"] = str(Path(td) / "noroot")
            assert f.main(["--months", "3", "--from", str(src)]) == 0
            assert (home / "cdf-03.npz").read_bytes() == blob
            # a second install is a no-op; a file with other bytes is refused and not installed
            assert f.main(["--months", "3", "--from", str(src)]) == 0
            (src / "cdf-03.npz").write_bytes(blob + b"x")
            (home / "cdf-03.npz").unlink()
            assert f.main(["--months", "3", "--from", str(src)]) == 1
            assert not (home / "cdf-03.npz").exists()
            # auto: $NAMGPU_ROOT/soil_cdf has no March table, so the fetched folder is read
            (home / "cdf-03.npz").write_bytes(blob)
            os.environ["NAMGPU_SOIL_CDF"] = "auto"
            ctx = SimpleNamespace(start=datetime(2026, 3, 1, 12))
            assert soil_start.cdf_table_path(ctx) == home / "cdf-03.npz"
        finally:
            f.PINS_FILE = old_pins
            for k, v in old_env.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


def test_shipped_pins_cover_every_month():
    pins = f.pins()["files"]
    assert sorted(pins) == [f"cdf-{m:02d}.npz" for m in range(1, 13)]
    assert all(v["bytes"] > 0 and len(v["sha256"]) == 64 for v in pins.values())


def test_auto_fetches_the_missing_month(monkeypatch, tmp_path):
    """NAMGPU_SOIL_CDF=auto with no table for the start month: the run fetches that month itself (from the release, or
    WOOF_NAM_SOIL_CDF_URL), checked against the pins; a fetch that fails stops with one line naming the command."""
    blob = b"march table" * 50
    src, home = tmp_path / "src", tmp_path / "home"
    src.mkdir()
    (src / "cdf-03.npz").write_bytes(blob)
    monkeypatch.setattr(f, "PINS_FILE", _pins(tmp_path, blob))
    monkeypatch.setenv("NAMGPU_SOIL_CDF_HOME", str(home))
    monkeypatch.setenv("NAMGPU_ROOT", str(tmp_path / "noroot"))
    monkeypatch.setenv("NAMGPU_SOIL_CDF", "auto")
    monkeypatch.setenv("WOOF_NAM_SOIL_CDF_URL", src.as_uri())
    ctx = SimpleNamespace(start=datetime(2026, 3, 1, 12))
    assert soil_start.cdf_table_path(ctx) == home / "cdf-03.npz"
    assert (home / "cdf-03.npz").read_bytes() == blob
    (home / "cdf-03.npz").unlink()
    monkeypatch.setenv("WOOF_NAM_SOIL_CDF_URL", (tmp_path / "nowhere").as_uri())
    import pytest
    with pytest.raises(FileNotFoundError) as e:
        soil_start.cdf_table_path(ctx)
    assert "could not fetch cdf-03.npz" in str(e.value) and "woof-nam fetch-soil-cdf --months 3" in str(e.value)
