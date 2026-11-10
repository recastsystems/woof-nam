"""Write the domain configfiles derived from NAM's own: parm/nam_configfile_4km, _5km, _7p6km and _band12.

    python tools/make_domain_configs.py

Each file is parm/nam_configfile (the 12 km parent, unmodified NAM copy) with the listed keys replaced in
place (comments kept), so every key a domain does not change stays NAM's. Rows are tagged with the NAM
source of the value.

4 km and 5 km: NAM's CONUS-nest footprint (parm/nam_configfile_conusnest: 1827 x 1467 at ratio 4,
i_parent_start 295, j_parent_start 95 = 456.5 x 366.5 parent cells), at ratio 3 (1371 x 1101) and 12/5
(1097 x 881). Each is its own rotated lat-lon grid centred on the footprint centre: parent rotated
(tlm, tph) = (5.7645, -15.093) deg = geographic 38.617 N, 98.870 W (grid.rotated_to_geographic on the
parent pole 54 N 106 W). wbd = -(IM-1)/2 dlmd, sbd = -(JM-1)/2 dphd (config.dlmd/dphd derive the spacing
back). Physics from the CONUS-nest configfile (convection none, gwdflg false, smag2 0.3, codamp 12,
rhgrd 1.0, fres 0.25, fsl/fss 0.75, nodeep true, cdmb 0.5, sigfac 0.0), except spec_adv stays false
(the proven path; the nest's true is a later knob). Steps: 4 km dt 25/3 s (the parent's dt / 3, 432 steps
per hour), filt_dt 25/9 s, nrads = nradl = 144 (20 min, as NAM); 5 km dt 10 s, filt_dt 10/3 s,
nrads = nradl = 120.

band4: the 4 km centre band (1371 x 201), the one-card smoke of the 4 km path.

band12: the 12 km centre band for the 1-card vs 2-card bitwise proof: rows 279..557 of the parent
(954 x 279), symmetric about the rotated equator (sbd -15.012), same pole and physics as the parent; its
static is cut from static/nam12_static.npz (static/nam12band_static.npz, made here when present).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
PARM = ROOT / "namgpu" / "parm"

from namgpu.nest.domain import NEST_PHYSICS, render  # noqa: E402  (the nest door's keys and renderer)

CENTRE = {"tph0d": "38.617", "tlm0d": "-98.870"}

DOMAINS = {
    "4km": dict(CENTRE, im="1371", jm="1101", wbd="-28.770", sbd="-19.800",
                dt_int="8", dt_num="1", dt_den="3", filt_dt_int="2", filt_dt_num="7", filt_dt_den="9",
                nrads="144", nradl="144", **NEST_PHYSICS,
                static_npz="static/nam4km_static.npz", domain_name="nam4km"),
    "5km": dict(CENTRE, im="1097", jm="881", wbd="-28.770", sbd="-19.800",
                dt_int="10", dt_num="0", dt_den="3", filt_dt_int="3", filt_dt_num="1", filt_dt_den="3",
                nrads="120", nradl="120", **NEST_PHYSICS,
                static_npz="static/nam5km_static.npz", domain_name="nam5km"),
    # 7.6 km: the same CONUS-nest footprint at 12/7.6 of the parent spacing (dlmd 0.0798, dphd 0.0684 deg;
    # 723 x 581, wbd = -361 dlmd, sbd = -290 dphd). Frozen-NAM 12 km physics (BMJ convection, GWD on, smag2 0.4,
    # codamp 9: the parent keys, not NEST_PHYSICS: 7.6 km does not resolve convection). dt 15 s (parent 25 s x 7.6/12
    # = 15.8, rounded down: dx/dt 507 m/s vs the parent 480), filt_dt 5 s (divides the 600 s TDFI window),
    # nrads = nradl = 80 (20 min, as NAM). Lane mesh-7p6km 2026-11-08.
    "7p6km": dict(CENTRE, im="723", jm="581", wbd="-28.807800", sbd="-19.836000",
                  dt_int="15", dt_num="0", dt_den="3", filt_dt_int="5", filt_dt_num="0", filt_dt_den="3",
                  nrads="80", nradl="80",
                  static_npz="static/nam7p6km_static.npz", domain_name="nam7p6km"),
    # na7p6: the NAM 12 km parent footprint (North America, pole 54 N 106 W, same wbd/sbd corners) at 7.6 km
    # (1506 x 1318: dlmd 0.079786, dphd 0.068392 deg = 12 km x 0.633). Parent physics keys (BMJ, GWD), dt 15 s,
    # filt_dt 5 s, radiation every 80 steps (20 min), as the 7p6km row. 2026-11-09.
    "na7p6": dict(im="1506", jm="1318",
                  dt_int="15", dt_num="0", dt_den="3", filt_dt_int="5", filt_dt_num="0", filt_dt_den="3",
                  nrads="80", nradl="80",
                  static_npz="static/namna7p6_static.npz", domain_name="namna7p6"),
    "band12": dict(jm="279", sbd="-15.012", static_npz="static/nam12band_static.npz", domain_name="nam12band"),
}
# band4: the 4 km grid's centre band (1371 x 201, rows 451..651, sbd -3.6), the one-card smoke of the 4 km
# path (nest physics, dt 25/3, own static and bocos) on a 32 GB card before the 8-card run.
DOMAINS["band4"] = dict(DOMAINS["4km"], jm="201", sbd="-3.600", static_npz="static/nam4kmband_static.npz",
                        domain_name="nam4kmband")
# band static = rows [start, stop) of the full domain's static
BANDS = {"band12": ("static/nam12_static.npz", 278, 557), "band4": ("static/nam4km_static.npz", 450, 651)}


def main():
    base = (PARM / "nam_configfile").read_text()
    for tag, repl in DOMAINS.items():
        p = PARM / f"nam_configfile_{tag}"
        p.write_text(render(base, repl, tag))
        print("wrote", p)
    for tag, (src, a, b) in BANDS.items():
        ref = ROOT / src
        if ref.is_file():
            z = np.load(ref)
            out = ROOT / DOMAINS[tag]["static_npz"]
            np.savez_compressed(out, **{k: (z[k][a:b] if z[k].ndim == 2 else z[k]) for k in z.files})
            print("wrote", out)
    from namgpu import config
    for tag in DOMAINS:
        c = config.load(PARM / f"nam_configfile_{tag}")
        print(tag, c.im, c.jm, "dlmd", round(c.dlmd, 6), "dphd", round(c.dphd, 6), "dt", c.dt, "filt_dt",
              c.filt_dt, "nrads", c.nrads, "conv", c.convection, "gwd", c.gwdflg, "spec_adv",
              c.raw.get("spec_adv"), "static", c.raw.get("static_npz"))


if __name__ == "__main__":
    main()
