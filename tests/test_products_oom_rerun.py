"""woof-nam products on a full shared card: the hand-off frame and the writer wait and rerun, they do not fail the hour.
Breakage these prevent: the 0.2.1 candidate's 12 km arctic-front run with --products on a 32 GB card lost +8, +16,
+18 and +30 h: two frames met cupy OutOfMemoryError and failed at once (no rerun for the frame), and two writers
refused with their own "CUDA device has N bytes free, the frame needs about M", which the rerun did not read as a
full card."""
import os
import sys
from pathlib import Path

os.environ["NAMGPU_PRODUCTS_OOM_WAIT_S"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.io import products  # noqa: E402


class OutOfMemoryError(Exception):
    """Stands in for cupy.cuda.memory.OutOfMemoryError (matched by name, as products does)."""


def test_writer_refusal_reads_as_full_card():
    assert products._is_oom("Error: /x/wrfout: CUDA device has 182321152 bytes free, the frame needs about 465667424")
    assert products._is_oom("cudaErrorMemoryAllocation")
    assert not products._is_oom("rw_rrfs_upp exit 1: missing field PBLH")


def test_frame_oom_is_rerun_then_fits():
    calls, logs = [], []

    def start():
        calls.append(1)
        if len(calls) < 3:
            raise OutOfMemoryError("Out of memory allocating 83,886,080 bytes (allocated so far: 1,262,485,504 bytes).")
        return "hour"

    assert products._start_hour_fits(start, "wrfout_d01_x", log=logs.append) == "hour"
    assert len(calls) == 3
    assert len(logs) == 2 and "frame out of GPU memory, rerun 2" in logs[1]


def test_other_errors_fail_at_once():
    calls = []

    def start():
        calls.append(1)
        raise ValueError("no T2 in the history")

    try:
        products._start_hour_fits(start, "wrfout_d01_x", log=lambda m: None)
    except ValueError:
        pass
    else:
        raise AssertionError("a non-memory error must fail the hour")
    assert len(calls) == 1


def test_full_card_that_stays_full_fails():
    keep = products.OOM_RETRIES
    products.OOM_RETRIES = 3
    calls = []

    def start():
        calls.append(1)
        raise OutOfMemoryError("Out of memory")

    try:
        products._start_hour_fits(start, "wrfout_d01_x", log=lambda m: None)
    except OutOfMemoryError:
        pass
    else:
        raise AssertionError("a card that stays full fails the hour")
    finally:
        products.OOM_RETRIES = keep
    assert len(calls) == 4


if __name__ == "__main__":
    test_writer_refusal_reads_as_full_card()
    test_frame_oom_is_rerun_then_fits()
    test_other_errors_fail_at_once()
    test_full_card_that_stays_full_fails()
    print("test_products_oom_rerun: 4 passed")
