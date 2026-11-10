"""Version selection and the hex resource seam, without starting a forecast."""
import hashlib
from types import SimpleNamespace

import pytest

from namgpu import config
from namgpu.physics.driver import phys_tempo_driver as tempo


@pytest.fixture(autouse=True)
def clean_options(monkeypatch):
    for key in (tempo.ENV_VERSION, tempo.ENV_MELT, "NAMGPU_MP", "NAMGPU_PHYSICS_SUITE"):
        monkeypatch.delenv(key, raising=False)


def test_latest_option_and_explicit_legacy(monkeypatch):
    assert tempo._version() == "v3.2.5"
    assert tempo._version({"tempo_version": "v3.0.3"}) == "v3.0.3"
    monkeypatch.setenv(tempo.ENV_VERSION, "v3.2.5")
    assert tempo._version({"tempo_version": "v3.0.3"}) == "v3.2.5"
    monkeypatch.setenv(tempo.ENV_VERSION, "v3.2.4")
    with pytest.raises(ValueError, match="v3.2.4"):
        tempo._version()


def test_version_flag_abis_keep_melting_on(monkeypatch):
    assert tempo._flags("v3.0.3") == (0, 0, 0, 1)
    assert tempo._flags("v3.2.5") == (0, 0, 0, 1, 0)
    monkeypatch.setenv(tempo.ENV_MELT, "0")
    assert tempo._flags("v3.2.5") == (0, 0, 0, 0, 0)


@pytest.mark.parametrize("version,suffix", [("v3.2.5", "325"), ("v3.0.3", "303")])
def test_real_hex_column_and_warp_match_version(version, suffix):
    pytest.importorskip("hexcore")
    tv, ph, base, _ = tempo._hexcore(version)
    assert tv.__name__.endswith("tempo_v" + suffix)
    assert ph.__name__.endswith("tempo_phase_v" + suffix)
    assert callable(tv.launch_columns) and callable(ph.launch_steps)
    assert callable(base.fast_tables) and callable(base._set_scalars)
    assert tv.TABLE_LAYOUT == base.TABLE_LAYOUT


def test_installed_packet_header_is_verified(tmp_path):
    pytest.importorskip("hexcore")
    header = tmp_path / "maths.cuh"
    header.write_text("// certified device maths\n")
    packet = {"mp_tempo": {"device_math": {"path": str(header),
              "sha256": hashlib.sha256(header.read_bytes()).hexdigest()}}}
    seen = []
    rb = SimpleNamespace(resolve_resource=lambda key: seen.append(key) or tmp_path / "packet.json",
                         load_packet=lambda path: packet)
    assert tempo._maths(rb) == header.read_text()
    assert seen == ["physics-resources"]
    header.write_text("changed")
    with pytest.raises(ValueError, match="SHA256 differs"):
        tempo._maths(rb)


def test_tempo_version_does_not_select_tempo_for_frozen_nam(monkeypatch):
    before = config.load()
    monkeypatch.setenv(tempo.ENV_VERSION, "v3.0.3")
    after = config.load()
    assert before == after
    assert after.mp_column == "" and after.raw["physics_suite"] == "nam"
    monkeypatch.setenv("NAMGPU_MP", "tempo")
    option = config.load()
    assert option.mp_column == "tempo" and option.microphysics == "thompson"
