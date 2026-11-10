"""namgpu/nest/coupled.py: the WindowCoupler host tables (numpy only, no card) on NCEP's CONUS nest geometry, and the
conusnest row of namgpu/nest/ncep_nests.py against nam.v4.2.11's own keys.

The breakage these catch: a nest boundary point that no rank fills (its strip keeps the start value for the whole run)
or that two ranks fill from different parent data, a parent point packed by no rank or by two (the window holds a stale
or a doubled value), and a conusnest grid or configfile that drifts from NCEP's (a product grid that no longer lands
where NOAA's did)."""
import numpy as np
import pytest

from namgpu.nest import coupled, ncep_nests
from namgpu import slab as slabmod

PARENT = (954, 835)           # parm/nam_configfile
CONUS = (1827, 1467, 295, 95, 4)


def _local_rows(jmg, n, r):
    if n == 1:
        return jmg
    a, b = slabmod.split_rows(jmg, n)[r]
    return min(b + slabmod.JHALO, jmg) - max(a - slabmod.JHALO, 1) + 1


@pytest.mark.parametrize("nranks", [1, 2, 3, 8])
def test_tables_partition(nranks):
    imp, jmp = PARENT
    imc, jmc, ips, jps, ratio = CONUS
    lns = 5
    seen_w = []
    strips = {"h": [], "v": []}
    for r in range(nranks):
        t = coupled.tables(imp, jmp, _local_rows(jmp, nranks, r), imc, jmc, _local_rows(jmc, nranks, r), ips, jps,
                           ratio, lns, lns, nranks, r)
        assert sum(t["counts"]) == t["npts"]
        seen_w.append(t["wpos"][r][: t["counts"][r]])
        assert t["ring"][-1] == len(t["ring"][0])            # the whole ring on every rank
        for k in ("h", "v"):
            ic, jc, _iw, _js, _w, sd, off, lstr, n = t[k]
            assert n == len(ic)
            we = sd >= 2
            # a W/E offset decodes to this rank's local row and the strip column
            assert np.all(off[we] // lns + t["joff_c"] == jc[we])
            assert np.all(off[we] < lstr[sd[we]])
            assert np.all(off[~we] < lstr[sd[~we]])
            if nranks > 1:
                assert (r == 0) == bool((sd == 0).any())
                assert (r == nranks - 1) == bool((sd == 1).any())
            strips[k].append(np.stack([sd, jc, ic, off % lns]).T)
    w = np.concatenate(seen_w)
    assert len(np.unique(w)) == len(w)                          # every window point packed by exactly one rank
    t1 = coupled.tables(imp, jmp, jmp, imc, jmc, jmc, ips, jps, ratio, lns, lns, 1, 0)
    assert len(w) == t1["npts"]
    for k in ("h", "v"):                                        # every global strip point filled on some rank
        g = {tuple(x) for x in np.stack([t1[k][5], t1[k][1], t1[k][0]]).T}
        got = {tuple(x[:3]) for a in strips[k] for x in a}
        assert got == g


def test_conusnest_grid_is_ncep():
    g = ncep_nests.nest_grid("conusnest")
    assert (g["im"], g["jm"], g["i_parent_start"], g["j_parent_start"]) == (1827, 1467, 295, 95)
    assert g["tph0d"] == 54.0 and g["tlm0d"] == -106.0
    assert abs(g["dlmd"] - 0.0315) < 1e-12 and abs(g["dphd"] - 0.027) < 1e-12
    assert abs(g["wbd"] + 22.995) < 1e-9 and abs(g["sbd"] + 34.884) < 1e-9     # parent wbd + 294 dlmd, sbd + 94 dphd
    lat, lon = g["center"]
    assert abs(lat - 38.61676) < 1e-4 and abs(lon + 98.87040) < 1e-4            # nam_namelist.nps_conusnest dom_cen


def test_conusnest_configfile_keys(tmp_path):
    from namgpu.nest import domain
    g = ncep_nests.nest_grid("conusnest")
    k = domain.config_keys(g, "conusnest", tmp_path / "s.npz", spec_adv=True)
    want = {"im": "1827", "jm": "1467", "dt_int": "6", "dt_num": "1", "dt_den": "4", "filt_dt_int": "2",
            "filt_dt_num": "0", "filt_dt_den": "4", "nrads": "192", "nradl": "192", "smag2": "0.3", "codamp": "12.0",
            "gwdflg": "false", "cdmb": "0.5", "sigfac": "0.00", "convection": "none", "rhgrd": "1.0", "fres": "0.25",
            "fsl": "0.75", "fss": "0.75", "nodeep": ".true.", "spec_adv": ".true.", "i_parent_start": "295",
            "j_parent_start": "95", "parent_child_space_ratio": "4"}   # nam.v4.2.11 parm/nam_configfile_conusnest
    assert {x: k[x] for x in want} == want
