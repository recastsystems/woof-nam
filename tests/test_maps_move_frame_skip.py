"""A frame off the whole hour with no frame 1 h before it skips its 1 h rain map (CPU only).

A live moving nest (`woof-nam nest --concurrent --move-nest`) writes its window just before and after each move, off
the hour and outside any output list. The maps tail raised "unexpected SKIPPED products: qpf_1h" on the first such
frame, so the door ended "maps nest rc 1". A frame on the whole hour with no frame before it still raises.
"""
from namgpu.io import io_maps

FIRST = "wrfout_d01_2026-11-17_00:00:00"
PRESENT = {FIRST, "wrfout_d01_2026-11-17_01:00:00", "wrfout_d01_2026-11-17_01:45:00", "wrfout_d01_2026-11-17_02:00:00"}


def test_move_frame_without_a_frame_an_hour_before_is_expected():
    why = "wrfout_d01_2026-11-17_01:45:00: no history frame 1 h earlier to difference"
    assert io_maps._expected_skip("qpf_1h", why, FIRST, None, PRESENT)


def test_move_frame_skip_needs_the_present_set():
    why = "wrfout_d01_2026-11-17_01:45:00: no history frame 1 h earlier to difference"
    assert not io_maps._expected_skip("qpf_1h", why, FIRST, None, None)


def test_whole_hour_frame_without_a_frame_before_still_raises():
    why = "wrfout_d01_2026-11-17_02:00:00: no history frame 1 h earlier to difference"
    assert not io_maps._expected_skip("qpf_1h", why, FIRST, None, PRESENT - {"wrfout_d01_2026-11-17_01:00:00"})


def test_off_hour_frame_whose_hour_before_exists_is_not_excused():
    why = "wrfout_d01_2026-11-17_01:45:00: no history frame 1 h earlier to difference"
    assert not io_maps._expected_skip("qpf_1h", why, FIRST, None, PRESENT | {"wrfout_d01_2026-11-17_00:45:00"})
