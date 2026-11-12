"""woof-nam run --products gives its card back before it waits for the GRIB2 follower, on every one-card exit.

Breakage it prevents (checker of door-products-3, 2026-11-11; shown on 7a5466d by the 3 h warm run of follow-4 with
nvidia-smi samples: the finished forecast held 15,900 MiB for its whole 36 s wait): on the --no-maps exit and on the
exit of a forecast that failed, the run went from its step loop to the follower's wait (products_finish) without
release_card, so the finished forecast kept its card memory (and on our box lanes its card lock) for the whole wait,
while the follower's writer needed that card. Only the maps exit gave the card back. Freeing the pool alone still left
4,082 MiB (the CUDA context, its modules, the local-memory reservation), so the wait runs in a re-exec of the process,
as the maps do. CPU only: the history drain, the writer shutdown, the card release, the receipt write, the re-exec and
the wait are stand-ins that record their order."""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu import run_forecast as rf  # noqa: E402
from namgpu.io import io_history  # noqa: E402


def _exit(no_maps: bool, error, waits: bool = True):
    calls = []
    keep = (io_history.drain, io_history.shutdown, rf.release_card, rf.products_finish, rf.os.execv,
            rf._FOLLOWER_PROC)
    with tempfile.TemporaryDirectory() as t:
        out = Path(t)

        def finish(o, rc, wait=True, stall_s=900.0):
            rec = json.loads((o / "forecast-receipt.json").read_text())
            calls.append(("wait", rec.get("card_released"), rc))
            return rc

        def execv(exe, argv):
            rec = json.loads((out / "forecast-receipt.json").read_text())
            calls.append(("exec-wait" if "--no-maps" in argv else "exec-maps", rec.get("card_released"),
                          rec.get("exit_code")))
            raise SystemExit(0)
        io_history.drain = lambda ctx: calls.append(("drain", None, None))
        io_history.shutdown = lambda: calls.append(("shutdown", None, None))
        rf.release_card = lambda ctx: calls.append(("release", None, None)) or {"card_used_gib_after": 0.4,
                                                                                "locks_dropped": []}
        rf.products_finish = finish
        rf.os.execv = execv
        rf._FOLLOWER_PROC = object() if waits else None
        a = SimpleNamespace(no_maps=no_maps, no_products_wait=False, products_stall_s=900.0, case="c", case_dir=None,
                            no_columns=True)
        rec = {"error": error}
        try:
            rc = rf.exit_one_card(a, SimpleNamespace(case_dir=out), out, rec, {})
        except SystemExit:
            rc = "exec"
        finally:
            (io_history.drain, io_history.shutdown, rf.release_card, rf.products_finish, rf.os.execv,
             rf._FOLLOWER_PROC) = keep
    return rc, calls


def _names(calls):
    return [c[0] for c in calls]


def test_no_maps_exit_releases_before_the_wait():
    rc, calls = _exit(no_maps=True, error=None)
    assert _names(calls) == ["drain", "shutdown", "release", "exec-wait"], calls
    assert calls[-1][1] and calls[-1][1]["card_used_gib_after"] == 0.4 and calls[-1][1]["at"], \
        "the receipt the follower reads must record the release"
    assert calls[-1][2] == 0 and rc == "exec"


def test_failed_forecast_exit_releases_before_the_wait():
    rc, calls = _exit(no_maps=False, error="OutOfMemoryError: stand-in")
    assert _names(calls) == ["drain", "shutdown", "release", "exec-wait"], calls
    assert calls[-1][2] == 1, "the wait-only tail must return the forecast's failure"


def test_maps_exit_still_releases_before_the_maps():
    rc, calls = _exit(no_maps=False, error=None)
    assert rc == "exec" and _names(calls) == ["drain", "shutdown", "release", "exec-maps"], calls


def test_no_follower_no_maps_writes_the_receipt_and_returns():
    rc, calls = _exit(no_maps=True, error=None, waits=False)
    assert _names(calls) == ["wait"] and rc == 0, calls


def test_wait_only_tail_waits_with_the_forecasts_code():
    calls = []
    keep = rf.products_finish
    rf.products_finish = lambda o, rc, wait=True, stall_s=900.0: calls.append(rc) or rc
    try:
        with tempfile.TemporaryDirectory() as t:
            out = Path(t)
            (out / "forecast-receipt.json").write_text(json.dumps({"exit_code": 1, "error": "x"}))
            assert rf.maps_only("2024-05-06_12z", out, None, True, maps=False) == 1 and calls == [1], calls
            # no maps, no columns: the receipt is the forecast's, unchanged
            assert json.loads((out / "forecast-receipt.json").read_text()) == {"exit_code": 1, "error": "x"}
    finally:
        rf.products_finish = keep


def test_history_write_error_is_the_runs_error():
    calls = []
    keep = (io_history.drain, io_history.shutdown, rf.release_card)

    def drain(ctx):
        raise OSError("disk full")
    io_history.drain, io_history.shutdown = drain, (lambda: calls.append("shutdown"))
    rf.release_card = lambda ctx: calls.append("release") or {}
    try:
        rec = {"error": None}
        rf.hand_back_card(SimpleNamespace(), rec)
    finally:
        io_history.drain, io_history.shutdown, rf.release_card = keep
    assert calls == ["shutdown", "release"] and "disk full" in rec["error"], (calls, rec)


if __name__ == "__main__":
    test_no_maps_exit_releases_before_the_wait()
    test_failed_forecast_exit_releases_before_the_wait()
    test_maps_exit_still_releases_before_the_maps()
    test_no_follower_no_maps_writes_the_receipt_and_returns()
    test_wait_only_tail_waits_with_the_forecasts_code()
    test_history_write_error_is_the_runs_error()
    print("test_run_card_before_wait: 6 passed")
