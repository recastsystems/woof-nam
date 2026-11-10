"""RUC with HRRR static soil classes: a missing BNU soil set is fetched with gpuwm fetch-geog into the first geog root,
and a failed fetch stops with one line naming the command (CPU only; the fetch itself is replaced)."""
import subprocess
import sys

import pytest

from namgpu.init import ruc_hrrr_soilclass as r


def test_fetches_into_first_root(monkeypatch, tmp_path):
    seen = {}

    def fake(cmd, *a, **k):
        seen["cmd"] = cmd
        d = tmp_path / "geog" / "bnu_soiltype_top"
        d.mkdir(parents=True)
        (d / "index").write_text("type=categorical\n")
        return subprocess.CompletedProcess(cmd, 0)
    monkeypatch.setattr(subprocess, "run", fake)
    got = r.fetch_bnu([tmp_path / "geog", tmp_path / "other"])
    assert got == tmp_path / "geog" / "bnu_soiltype_top"
    assert seen["cmd"][:3] == [sys.executable, "-m", "gpuwm"]
    assert seen["cmd"][3:] == ["fetch-geog", "--datasets", "bnu_soiltype_top", "--root", str(tmp_path / "geog")]


def test_failed_fetch_names_the_command(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: subprocess.CompletedProcess(cmd, 1))
    with pytest.raises(SystemExit) as e:
        r.fetch_bnu([tmp_path / "geog"])
    assert "Run: gpuwm fetch-geog --datasets bnu_soiltype_top --root" in str(e.value)
