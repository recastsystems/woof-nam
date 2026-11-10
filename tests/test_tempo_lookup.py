"""TEMPO lookup files: a named folder must hold them; with none named, gpuwm-hex's rule is used (CPU only)."""
import pytest

from namgpu.physics.tempo import tables as T


def test_named_folder_reads_its_files_and_refuses_a_missing_one(tmp_path, monkeypatch):
    (tmp_path / "freeze_water_data_tempo_v3").write_bytes(b"x")
    monkeypatch.delenv("NAMGPU_TEMPO_LOOKUP", raising=False)
    get = T.lookup_resolver(tmp_path)
    assert get("freeze_water_data_tempo_v3") == str(tmp_path / "freeze_water_data_tempo_v3")
    with pytest.raises(SystemExit, match="NAMGPU_TEMPO_LOOKUP"):
        get("qr_acr_qs_data_tempo_v3")
    monkeypatch.setenv("NAMGPU_TEMPO_LOOKUP", str(tmp_path))
    assert T.lookup_resolver()("freeze_water_data_tempo_v3") == str(tmp_path / "freeze_water_data_tempo_v3")


def test_no_hard_coded_home_folder():
    # the old default ~/.gpuwm/tables/tempo-v3 was installed by nothing (two-wheel check 2026-11-09)
    assert T.LOOKUP_DIR_DEFAULT is None
