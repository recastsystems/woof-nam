"""`woof-nam nest --ncep-nest firewxnest --concurrent` is refused in words (CPU only).

The fire-weather nest is placed inside NCEP's 3 km nest: its i/j_parent_start and its space ratio 2 count 3 km cells.
The live-parent NCEP door couples every nest to the 12 km parent, so it cropped and read the 12 km parent at those cell
numbers and the nest took its boundaries from parent cells that do not hold it.
"""
from types import SimpleNamespace

import pytest

from namgpu.nest import suite_door


def test_firewx_with_a_live_parent_is_refused(tmp_path):
    a = SimpleNamespace(two_way=False, source="gfs", ncep_nest="firewxnest", out=tmp_path / "out")
    with pytest.raises(SystemExit) as e:
        suite_door.main(a, [])
    assert "fire-weather nest" in str(e.value) and "woof-nam suite" in str(e.value)
    assert not (tmp_path / "out").exists()


def test_firewx_beside_another_nest_is_refused():
    assert suite_door.live_parent_refusal(["conusnest", "firewxnest"])


@pytest.mark.parametrize("name", ["conusnest", "alaskanest", "hawaiinest", "priconest"])
def test_the_3km_nests_are_not_refused(name):
    assert suite_door.live_parent_refusal([name]) is None
