"""woof-nam doctor: a README install (terrain only) reads "ok" for the terrain every run needs and "optional" for the
nest land sets; it used to print "missing terrain data" after the README steps although the 12 km run had all it reads."""
import io
import os
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _geog(root: Path, names):
    for n in names:
        d = root / n
        d.mkdir(parents=True)
        (d / "index").write_text("type = continuous\n")


def _with_env(root: Path):
    keep = {k: os.environ.get(k) for k in ("NAMGPU_GEOG", "WPS_GEOG", "HOME")}
    os.environ["NAMGPU_GEOG"] = str(root)
    os.environ.pop("WPS_GEOG", None)
    os.environ["HOME"] = str(root.parent / "home")
    return keep


def _restore(keep):
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def test_terrain_only_is_ok_and_nest_land_optional():
    from namgpu import doctor
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "WPS_GEOG"
        _geog(root, ["topo_gmted2010_30s"])
        keep = _with_env(root)
        try:
            ok, msg = doctor.terrain()
            assert ok, msg
            assert "topo_gmted2010_30s" in msg
            ok2, msg2 = doctor.nest_land()
            assert not ok2
            assert "12 km run does not need it" in msg2 and "gpuwm fetch-geog" in msg2
        finally:
            _restore(keep)


def test_no_terrain_is_missing():
    from namgpu import doctor
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "WPS_GEOG"
        root.mkdir()
        keep = _with_env(root)
        try:
            ok, msg = doctor.terrain()
            assert not ok
            assert "--datasets topo_gmted2010_30s" in msg
        finally:
            _restore(keep)


def test_doctor_line_words():
    from namgpu import doctor
    buf = io.StringIO()
    with redirect_stdout(buf):
        doctor._line("optional", "nest land data", "x")
    assert buf.getvalue().split()[:4] == ["optional", "nest", "land", "data"]


if __name__ == "__main__":
    test_terrain_only_is_ok_and_nest_land_optional()
    test_no_terrain_is_missing()
    test_doctor_line_words()
    print("test_doctor_terrain: 3 passed")
