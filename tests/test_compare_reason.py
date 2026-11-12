"""CPU check that `woof-nam compare` names why it says DIFFERENT when no shared word differs (namgpu/io/hist_compare.py).
Run: python tests/test_compare_reason.py (or pytest)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

if __name__ == "__main__" and not os.environ.get("WOOF_NAM_TEST_INSTALLED"):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.io.hist_compare import _only_note, _why  # noqa: E402


def test_extra_variable_named():
    h = {"file": "wrfout_d01_x", "vars": 117, "only_a": [], "only_b": ["SNOWNC"], "diff": [], "words": 10, "words_diff": 0}
    rec = {"only_a": [], "only_b": [], "bitwise": False, "words_diff": 0}
    assert _only_note(h) == "; only in B: SNOWNC"
    assert _why(rec, [h]) == "; every shared value identical; variables only in B: SNOWNC"


def test_silent_when_identical_or_values_differ():
    h = {"file": "f", "vars": 1, "only_a": [], "only_b": [], "diff": [], "words": 10, "words_diff": 0}
    assert _only_note(h) == ""
    assert _why({"only_a": [], "only_b": [], "bitwise": True, "words_diff": 0}, [h]) == ""
    hd = dict(h, diff=["T"], words_diff=3, only_b=["SNOWNC"])
    assert _why({"only_a": [], "only_b": [], "bitwise": False, "words_diff": 3}, [hd]) == ""


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
    print("test_compare_reason: PASS")
