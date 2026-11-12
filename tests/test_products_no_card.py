"""woof-nam products with no CUDA device stops before the first hour, in one line, whatever --post-device says.

Breakage it prevents (nam-differences-2, 2026-11-11): `woof-nam products --post-device cpu` with the card hidden built
the frame of hour 0 on the processor, then the pinned writer stopped with "rw_rrfs_upp exit 3: the RRFS extras need a
CUDA device". The writer computes the derived records of NOAA's files on a card only, with --post-device cpu too (it
then runs only post groups a-e on the processor), so no hour can be written without a card. CPU only: the card is
hidden (CUDA_VISIBLE_DEVICES empty), the writer is never run, and an hour that starts is recorded by a stub."""
import contextlib
import io
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import exporter  # noqa: E402
from namgpu.io import products  # noqa: E402


def _products(tmp: Path, device: str):
    rd = tmp / f"run-{device}"
    rd.mkdir()
    (rd / "wrfout_d01_2024-11-05_12:00:00").write_bytes(b"")
    started = []

    def fake_start(history, *a, **k):
        started.append(Path(history).name)
        raise RuntimeError("rw_rrfs_upp exit 3: the RRFS extras need a CUDA device")
    keep = (exporter.ensure, exporter.require, products._start_hour)
    exporter.ensure = lambda *a, **k: Path("/bin/true")
    exporter.require = lambda *a, **k: Path("/bin/true")
    products._start_hour = fake_start
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            rc = products.main(["--run", str(rd), "--set", "awphys", "--post-device", device])
    finally:
        exporter.ensure, exporter.require, products._start_hour = keep
    return rc, err.getvalue().strip(), started


def test_no_card_stops_before_hour_0():
    with tempfile.TemporaryDirectory() as t:
        for device in ("cpu", "auto", "gpu"):
            rc, err, started = _products(Path(t), device)
            assert not started, f"--post-device {device}: an hour was started with no card: {started}"
            assert rc == 2, f"--post-device {device}: exit {rc}, want 2 ({err!r})"
            assert "\n" not in err, f"--post-device {device}: want one line, got {err!r}"
            assert "no CUDA device" in err and "need a card" in err, err


def test_card_probe_reads_hidden_card():
    ok, why = products.card_visible()
    assert not ok and why, (ok, why)


if __name__ == "__main__":
    test_no_card_stops_before_hour_0()
    test_card_probe_reads_hidden_card()
    print("test_products_no_card: 2 passed")
