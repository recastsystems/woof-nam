"""The boundary builder gives the same file whoever calls it: the whole build runs on the run's own stream.

Breakage it prevents: `woof-nam run --resume` to a later end called gfs_bocos.build without entering the run's stream
(a non-blocking CUDA stream). The interpolation kernels ran on that stream, but the copy of the boundary strips, the
column stacks and the tendency arithmetic ran on the calling thread's default stream, which does not wait for it. The
rebuilt boco.0006 of the arctic-front case (2020-10-24 00Z, 2026-11-11) held the +6 h values at 114,853 points of the
+9 h strip, a pressure tendency of 2.25 Pa/s, differed on every rebuild, and the next run stopped at its first step.

CPU only. The device is replaced by numpy, the decoder and the interpolation by stubs that record whether the run's
stream was this thread's current stream when they ran, and which boundary time each file was paired from."""
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.init import gfs_bocos, sources  # noqa: E402

START = datetime(2020, 10, 24, 0)


class FakeStream:
    """Stands for cupy.cuda.Stream: entering it makes it the entering thread's current stream."""

    def __init__(self):
        self._tl = threading.local()
        self.syncs = 0

    def __enter__(self):
        self._tl.depth = getattr(self._tl, "depth", 0) + 1
        return self

    def __exit__(self, *exc):
        self._tl.depth -= 1

    def current(self) -> bool:
        return getattr(self._tl, "depth", 0) > 0

    def synchronize(self):
        self.syncs += 1


class FakeState:
    def __init__(self, grid, names=()):
        self.fields = {}
        self.hour = None

    def add(self, name, shape):
        self.fields[name] = np.zeros(shape, np.float32)
        return self.fields[name]

    def __getitem__(self, name):
        return self.fields[name]


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A case folder with empty GFS files to +36 h, and build()'s device work replaced by recording stubs."""
    from namgpu import config as configmod, devmem, state as statemod
    from namgpu.init import gfs_start, init_grid_geo

    monkeypatch.setenv("NAMGPU_BOCO_AGE_H", "0")
    monkeypatch.setitem(sys.modules, "cupy", np)
    case = tmp_path / "root" / "cases" / "2020-10-24_00z"
    cfg = SimpleNamespace(raw={}, lnsh=1, lnsv=1)
    for _, p in gfs_bocos.source_files(case, START, 36, sources.SOURCES["gfs"], cfg):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"g")
    stream = FakeStream()
    seen = {"decode": [], "model_state": [], "strips": [], "write": []}

    def lead(path):
        return int(Path(path).name.split(".f")[1][:3])

    def decode_source(bctx, path, tmpdir, **kw):
        seen["decode"].append((lead(path), stream.current()))
        return lead(path)

    def model_state(bctx, path, tmpdir, sfc_keys=None, rec=None, decoded=None, **kw):
        assert decoded == lead(path), "the interpolation got another boundary time's decode"
        seen["model_state"].append((decoded, stream.current()))
        bctx.st.hour = decoded

    def strips(st, lm, lnsh, lnsv):
        seen["strips"].append((st.hour, stream.current()))
        return {"T": (np.full(4, float(st.hour), np.float32),)}

    def write_boco(path, t0, old, new, tboco=10800.0):
        seen["write"].append((Path(path).name, int(old["T"][0][0]), int(new["T"][0][0]), stream.current()))
        Path(path).write_bytes(b"built")

    monkeypatch.setattr(statemod, "State", FakeState)
    monkeypatch.setattr(init_grid_geo, "run", lambda bctx: None)
    monkeypatch.setattr(gfs_start, "load_static", lambda bctx: ({"HGT_SFC": np.zeros((2, 2), np.float32)}, None))
    monkeypatch.setattr(gfs_start, "decode_source", decode_source)
    monkeypatch.setattr(gfs_start, "model_state", model_state)
    monkeypatch.setattr(gfs_bocos, "strips", strips)
    monkeypatch.setattr(gfs_bocos, "write_boco", write_boco)
    monkeypatch.setattr(configmod, "grid_stamp", lambda cfg: "grid")
    monkeypatch.setattr(devmem, "release", lambda stream=None: 0)
    ctx = SimpleNamespace(cfg=cfg, grid=SimpleNamespace(lm=2), stream=stream, case_dir=case, start=START,
                          init_source="gfs")
    return ctx, seen, tmp_path / "boco"


def _outside(seen):
    return {k: [x for x in v if not x[-1]] for k, v in seen.items() if any(not x[-1] for x in v)}


def test_build_runs_on_the_runs_stream_when_the_caller_has_not_entered_it(rig):
    """The resume door's call. Every decode, interpolation, strip copy and file write must see the run's stream as
    its thread's current stream, and the stream is synchronised before build returns."""
    ctx, seen, out = rig
    gfs_bocos.build(ctx, 9, out, log=lambda m: None, from_hour=6)
    print("outside the run's stream:", _outside(seen))
    assert not _outside(seen), ("these ran outside the run's stream, so they are not ordered after the kernels that "
                                f"fill the state: {_outside(seen)}")
    assert [w[:3] for w in seen["write"]] == [("boco.0006", 6, 9)]
    assert ctx.stream.syncs >= 1
    assert not ctx.stream.current()            # the caller's own current stream is put back


def test_full_build_pairs_each_file_with_its_two_times(rig):
    ctx, seen, out = rig
    with ctx.stream:
        gfs_bocos.build(ctx, 36, out, log=lambda m: None)
    assert [w[:3] for w in seen["write"]] == [(f"boco.{h:04d}", h, h + 3) for h in range(0, 36, 3)]
    assert sorted(d[0] for d in seen["decode"]) == list(range(0, 37, 3))
    assert not _outside(seen)
    assert (out / "BOCO-SOURCE.json").is_file()


def test_only_writes_the_files_asked_for_and_leaves_the_rest(rig):
    """A resumed run whose folder lacks boco.0006 and boco.0015: those two are built from their own two times each,
    nothing between them is written, and the files and the marker that are there are not touched."""
    ctx, seen, out = rig
    out.mkdir()
    for h in (0, 3, 9, 12, 18):
        (out / f"boco.{h:04d}").write_bytes(b"first leg")
    (out / "BOCO-SOURCE.json").write_text("first leg marker")
    gfs_bocos.build(ctx, 36, out, log=lambda m: None, from_hour=6, only=[6, 15])
    assert [w[:3] for w in seen["write"]] == [("boco.0006", 6, 9), ("boco.0015", 15, 18)], seen["write"]
    assert sorted(d[0] for d in seen["decode"]) == [6, 9, 15, 18]
    for h in (0, 3, 9, 12, 18):
        assert (out / f"boco.{h:04d}").read_bytes() == b"first leg"
    assert (out / "BOCO-SOURCE.json").read_text() == "first leg marker"
    assert not _outside(seen)


def test_only_refuses_an_hour_that_is_not_an_interval_start(rig):
    ctx, seen, out = rig
    with pytest.raises(ValueError, match="only="):
        gfs_bocos.build(ctx, 9, out, log=lambda m: None, from_hour=6, only=[7])
    assert seen["write"] == []


def test_parent_build_enters_the_runs_stream_too():
    """parent_bocos.build (a one-way nest's boundaries, also called by the resume door) has the same wrapper."""
    src = (Path(__file__).resolve().parents[1] / "namgpu" / "init" / "parent_bocos.py").read_text()
    body = src[src.index("def build(ctx"):src.index("def _build(ctx")]
    assert "with gfs_bocos.on_stream(ctx):" in body and "ctx.stream.synchronize()" in body


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
