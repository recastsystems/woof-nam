"""--ladder: each nest on the grid of the one before (CPU only)."""
import pytest

from namgpu.nest import domain, twoway_door


def _chain(spec):
    pg = domain.nest_grid(-9.0, -163.0, 1200, 12, 1200)
    ng = domain.nest_grid_associated(pg, -9.0, -163.0, 480, 3)
    return pg, ng, twoway_door.ladder_grids(spec, ng, -9.0, -163.0)


def test_no_ladder_is_empty():
    assert _chain(None)[2] == []


def test_12_3_15_ladder():
    pg, ng, more = _chain("1.5:240")
    assert ng["parent_child_space_ratio"] == 4
    (g,) = more
    assert g["res_km"] == 1.5 and g["parent_child_space_ratio"] == 2
    assert (g["tph0d"], g["tlm0d"]) == (pg["tph0d"], pg["tlm0d"])          # one pole for the whole ladder
    assert abs(g["dlmd"] - pg["dlmd"] / 8) < 1e-12
    # the 1.5 km nest sits inside the 3 km nest with room for the boundary rows
    assert all(v >= 0 for v in g["parent_room_cells"].values())


def test_ladder_refuses_a_bad_step():
    with pytest.raises(SystemExit):
        _chain("2:100")                       # 3 / 2 is not a whole ratio
    with pytest.raises(SystemExit):
        _chain("1.5")                         # RES:SIZE
    with pytest.raises(SystemExit):
        _chain("1.5:600")                     # wider than its 480 km parent


def test_ladder_step_takes_its_own_north_south_size():
    pg, ng, more = _chain("1.5:300x200")
    (g,) = more
    assert g["size_km"] == 300 and g["size_y_km"] == 200
    assert g["parent_cells"][0] > g["parent_cells"][1]          # wider than tall, in 3 km cells
    assert abs(g["extent_km"][1] - 200) < 3 and abs(g["extent_km"][0] - 300) < 5


def test_parent_spacing_is_a_row():
    assert twoway_door.parent_res(None) == 12 and twoway_door.parent_res(3.0) == 3
    with pytest.raises(SystemExit):
        twoway_door.parent_res(2.0)                                 # no RES_TABLE row
    # a 3 km root (IFS 0.1 deg -> 3 -> 1.5 km): the 1.5 km nest is a space ratio of 2 in it
    pg = domain.nest_grid(50.0, 8.0, 900, twoway_door.parent_res(3), 900)
    g = domain.nest_grid_associated(pg, 50.0, 8.0, 600, 1.5)
    assert pg["res_km"] == 3 and g["parent_child_space_ratio"] == 2


def test_the_combined_pool_counts_the_decode_floor_once():
    # three fit-path domains of the 18 Nov ladder shape: the floor (3.5 GiB) once, not three times
    def est(cols_im, cols_jm):
        g = {"im": cols_im, "jm": cols_jm}
        state = g["im"] * g["jm"] * 23.0 * 1024 / 2**30
        rrtm = 32 * g["im"] * 0.34e6 / 2**30
        return {"state_gib": state, "total_gib": state + rrtm + 1.3 + 0.5, "lmem_gib": 1.3, "fixed_gib": 0.5,
                "memory_path": "fit"}
    es = [est(87, 101), est(137, 161), est(137, 161)]
    work, cap = twoway_door._combined_pool(es)
    assert abs(work - max(sum(e["total_gib"] - 1.8 for e in es), sum(e["state_gib"] for e in es) + 3.5)) < 1e-9
    assert work < 6.0 and abs(cap - work - 0.25) < 1e-9
    # the old sum (each domain floored to state + 3.5) was over 11 GiB for the pool alone
    assert sum(max(e["total_gib"] - 1.8 + 0.25, e["state_gib"] + 3.5) for e in es) > 11.0
