"""A start from a forecast hour of a run given with --cycle that has not happened, or has not posted the hours the
window reads yet, is refused in one line before any fetch (window._given_run_posted). Before, `woof-nam fetch --cycle
2026111900 --lead 6` the afternoon before printed its plan and ended with three bare "HTTP Error 404" lines. CPU only:
the archive question is a stub."""
from datetime import datetime

import pytest

from namgpu.init import window

NOW = datetime(2026, 11, 18, 19, 55)


def test_run_not_happened_is_refused():
    with pytest.raises(ValueError, match="has not happened yet"):
        window.resolve("gfs", hours=6, cycle=datetime(2026, 11, 19, 0), lead=6, now=NOW, posted=lambda c, h: True)


def test_recent_run_not_posted_is_refused_in_words():
    asked = []

    def posted(c, h):
        asked.append((c, h))
        return h <= 3          # the 12Z run has posted hours 0 to 3 so far
    with pytest.raises(ValueError, match=r"the 2026-11-18 12Z gfs run has not posted hours 6 to 12 yet; --at TIME"):
        window.resolve("gfs", hours=6, cycle=datetime(2026, 11, 18, 12), lead=6, now=NOW, posted=posted)
    assert asked and all(c == datetime(2026, 11, 18, 12) for c, _ in asked)


def test_recent_run_posted_plans_as_before():
    p = window.resolve("gfs", hours=6, cycle=datetime(2026, 11, 18, 6), lead=6, now=NOW, posted=lambda c, h: True)
    q = window.resolve("gfs", hours=6, cycle=datetime(2026, 11, 18, 6), lead=6, now=NOW, posted=None)
    assert p == q and p["lead_h"] == 6


def test_old_run_is_not_asked():
    def posted(c, h):
        raise AssertionError("an archived run is not asked")
    p = window.resolve("gfs", hours=6, cycle=datetime(2026, 11, 12, 12), lead=200, now=NOW, posted=posted)
    assert p["lead_h"] == 198


def test_archive_that_does_not_answer_is_left_to_the_fetch():
    def posted(c, h):
        raise OSError("timed out")
    p = window.resolve("gfs", hours=6, cycle=datetime(2026, 11, 18, 12), lead=6, now=NOW, posted=posted)
    assert p["lead_h"] == 6
