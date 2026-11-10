"""Without gpuwm-hex, a run that asks for TEMPO or MYNN v6.1 stops with one line before any work; every other choice
runs on gpuwm alone (CPU only, no card)."""
import importlib.metadata as md

import pytest

from namgpu import hexdep
from namgpu.physics import suites

KNOBS = ("NAMGPU_MP", "NAMGPU_PBL", "NAMGPU_PHYSICS_SUITE", "NAMGPU_HEXCORE_PATH")


@pytest.fixture
def no_hex(monkeypatch):
    for k in KNOBS:          # set, then removed: undone at the end even when the door under test sets it itself
        monkeypatch.setenv(k, "x")
        monkeypatch.delenv(k)
    real = hexdep.importlib.util.find_spec
    monkeypatch.setattr(hexdep.importlib.util, "find_spec", lambda name, *a: None if name == "hexcore" else real(name, *a))
    return monkeypatch


def test_what_needs_hex(no_hex):
    assert hexdep.wanted() == []                                   # the NAM default
    assert hexdep.wanted("nam") == []
    for s in ("hrrrv5", "rapv5", "woof-nam-next-12km", "woof-nam-next-7p6km"):
        assert hexdep.wanted(s) == [f"--physics-suite {s} (TEMPO microphysics)"], s
    assert set(suites.names()) == {"nam", "hrrrv5", "rapv5", "woof-nam-next-12km", "woof-nam-next-7p6km"}, \
        "a new suite row: say here whether it needs gpuwm-hex"
    no_hex.setenv("NAMGPU_MP", "tempo")
    assert hexdep.wanted() == ["NAMGPU_MP=tempo (TEMPO microphysics)"]
    no_hex.delenv("NAMGPU_MP")
    no_hex.setenv("NAMGPU_PBL", "mynn-v61")
    assert hexdep.wanted() == ["NAMGPU_PBL=mynn-v61 (MYNN v6.1)"]
    no_hex.setenv("NAMGPU_PBL", "mynn-edmf")                       # the older MYNN port is gpuwm's
    assert hexdep.wanted() == []


def test_one_line_when_missing(no_hex):
    hexdep.require()                                               # nothing asked for: no check, no message
    with pytest.raises(SystemExit) as e:
        hexdep.require("hrrrv5", prog="woof-nam run")
    msg = str(e.value)
    assert "\n" not in msg
    assert msg.startswith("woof-nam run: --physics-suite hrrrv5 (TEMPO microphysics) needs gpuwm-hex 0.5.0 or newer")
    assert "gpuwm-hex is not installed" in msg and 'pip install "recast-woof-nam[modern]"' in msg


def test_old_hex_is_refused_and_new_hex_passes(no_hex):
    no_hex.setattr(hexdep.importlib.util, "find_spec", lambda name, *a: object())
    no_hex.setattr(hexdep.md, "version", lambda name: "0.4.2")
    assert hexdep.missing() == "the installed gpuwm-hex is 0.4.2"
    with pytest.raises(SystemExit):
        hexdep.require("woof-nam-next-12km")
    no_hex.setattr(hexdep.md, "version", lambda name: "0.5.0")
    assert hexdep.missing() is None
    hexdep.require("woof-nam-next-12km")

    def none(name):
        raise md.PackageNotFoundError(name)
    no_hex.setattr(hexdep.md, "version", none)                     # a hex source tree on the path
    assert hexdep.missing() is None and hexdep.installed() == "source tree"


def test_the_front_doors_stop_before_any_work(no_hex, capsys):
    from namgpu import cli
    with pytest.raises(SystemExit) as e:
        cli.main(["run", "--case", "2026-01-23_12z", "--out", "/nonexistent/x", "--physics-suite", "rapv5"])
    assert str(e.value).startswith("woof-nam run: --physics-suite rapv5 (TEMPO microphysics) needs gpuwm-hex")
    no_hex.delenv("NAMGPU_PHYSICS_SUITE")          # `woof-nam run` set it for its children; a new command is a new process
    no_hex.setenv("NAMGPU_MP", "tempo")
    with pytest.raises(SystemExit) as e:
        cli.main(["suite", "--cycle", "2026012312", "--out", "/nonexistent/x"])
    assert str(e.value).startswith("woof-nam suite: NAMGPU_MP=tempo (TEMPO microphysics) needs gpuwm-hex")


def test_nest_and_ensemble_doors_stop_before_any_work(no_hex):
    """The two doors the test above leaves out: `woof-nam nest` and `woof-nam ensemble` read TEMPO / MYNN v6.1 from the
    environment (an ensemble member's configfile keys cannot select them: namgpu.ensemble.PORTED)."""
    from namgpu import cli
    no_hex.setenv("NAMGPU_PBL", "mynn-v61")
    with pytest.raises(SystemExit) as e:
        cli.main(["nest", "--ncep-nest", "firewxnest", "--source", "gfs", "--start", "2026012312", "--hours", "1",
                  "--out", "/nonexistent/x"])
    assert str(e.value).startswith("woof-nam nest: NAMGPU_PBL=mynn-v61 (MYNN v6.1) needs gpuwm-hex")
    no_hex.delenv("NAMGPU_PBL")
    no_hex.setenv("NAMGPU_MP", "tempo")
    with pytest.raises(SystemExit) as e:
        cli.main(["ensemble", "--roster", "/nonexistent/r.json", "--case", "2026-01-23_12z", "--hours", "1",
                  "--out", "/nonexistent/x"])
    assert str(e.value).startswith("woof-nam ensemble: NAMGPU_MP=tempo (TEMPO microphysics) needs gpuwm-hex")
    no_hex.delenv("NAMGPU_MP")
    no_hex.setenv("NAMGPU_PHYSICS_SUITE", "hrrrv5")
    with pytest.raises(SystemExit) as e:
        cli.main(["ensemble", "--roster", "/nonexistent/r.json", "--case", "2026-01-23_12z", "--hours", "1",
                  "--out", "/nonexistent/x"])
    assert str(e.value).startswith("woof-nam ensemble: --physics-suite hrrrv5 (TEMPO microphysics) needs gpuwm-hex")
