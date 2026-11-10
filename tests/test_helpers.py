"""The pinned helper programs (rw_asos, rw_stage4, rw_hex_ugwp_oro): pins, lookup order, staging, refusals. CPU only."""
import hashlib
import json
import os
import re
from pathlib import Path

import pytest

from namgpu import exporter, helpers


def test_pins_cover_the_three_helpers():
    p = helpers.pins()
    assert p["release"] == exporter.ASSET_RELEASE and p["repo"] == exporter.REPO
    assert set(p["helpers"]) == {"rw_asos", "rw_stage4", "rw_hex_ugwp_oro"}
    for name, rows in p["helpers"].items():
        row = rows["linux-x86_64"]
        assert row["asset"] == f"{name}-linux-x86_64" and row["bytes"] > 100_000
        assert re.fullmatch(r"[0-9a-f]{64}", row["sha256"])


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A pin table for a small file in a local mirror folder, an empty home and no bridge folders."""
    mirror = tmp_path / "mirror"
    mirror.mkdir()
    body = b"#!/bin/sh\necho helper\n" * 50
    (mirror / "rw_asos-linux-x86_64").write_bytes(body)
    table = {"release": "v0.2.0", "repo": "recastsystems/woof-nam", "helpers": {"rw_asos": {"linux-x86_64": {
        "asset": "rw_asos-linux-x86_64", "bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}}}}
    monkeypatch.setattr(helpers, "pins", lambda: table)
    monkeypatch.setattr(exporter, "platform_key", lambda: "linux-x86_64")
    monkeypatch.setenv("WOOF_NAM_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("WOOF_NAM_BRIDGE_URL", str(mirror))
    monkeypatch.setenv("GPUWM_BRIDGES", "")
    monkeypatch.setenv("WOOF_NAM_NO_FETCH", "")
    monkeypatch.setenv("HOME", str(tmp_path / "nohome"))
    real = helpers.find

    def staged_only(name):          # this machine may hold a gpuwm bridges copy; the test is about the staged one
        p = helpers.staged_path(name)
        return p if p is not None and p.is_file() else None
    monkeypatch.setattr(helpers, "find", staged_only)
    return {"mirror": mirror, "table": table, "real_find": real, "tmp": tmp_path}


def test_first_use_stages_the_pinned_copy(fake):
    p = helpers.ensure("rw_asos")
    assert p == helpers.staged_path("rw_asos") and p.is_file() and os.access(p, os.X_OK)
    assert str(fake["tmp"] / "home") in str(p)
    assert helpers.ensure("rw_asos") == p                      # found, not fetched again


def test_a_file_that_fails_the_pin_is_not_installed(fake):
    (fake["mirror"] / "rw_asos-linux-x86_64").write_bytes(b"something else")
    with pytest.raises(exporter.ExporterMissing) as e:
        helpers.ensure("rw_asos")
    assert "not installed" in str(e.value) and "woof-nam fetch-bridges" in str(e.value)
    assert not helpers.staged_path("rw_asos").exists()
    assert not list(helpers.staged_path("rw_asos").parent.glob(".rw_asos.*"))


def test_no_fetch_is_one_line(fake, monkeypatch):
    monkeypatch.setenv("WOOF_NAM_NO_FETCH", "1")
    with pytest.raises(exporter.ExporterMissing) as e:
        helpers.ensure("rw_asos")
    assert "\n" not in str(e.value) and "woof-nam score" in str(e.value)


def test_bridge_folder_copy_comes_first(fake, monkeypatch):
    b = fake["tmp"] / "bridges"
    b.mkdir()
    (b / "rw_asos").write_text("x")
    monkeypatch.setenv("GPUWM_BRIDGES", str(b))
    assert fake["real_find"]("rw_asos") == b / "rw_asos"


def test_score_names_the_fix_when_a_reader_cannot_be_had(fake, monkeypatch):
    from namgpu.io import namscore
    monkeypatch.setenv("WOOF_NAM_NO_FETCH", "1")
    with pytest.raises(SystemExit) as e:
        namscore._tool("rw_asos")
    assert "woof-nam fetch-bridges" in str(e.value)


def test_fetch_bridges_from_a_file_on_disk(fake, capsys):
    assert exporter.main(["--from", str(fake["mirror"] / "rw_asos-linux-x86_64")]) == 0
    assert "rw_asos:" in capsys.readouterr().out and helpers.staged_path("rw_asos").is_file()
