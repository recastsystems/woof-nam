"""woof-nam nest --two-way | --concurrent: a 12 km parent and a nest inside it, run together in one process on one card.

    woof-nam nest --two-way --center 24N,83W --size-km 1900 --size-y-km 1400 --res-km 3 \
        --parent-size-km 3000 --parent-size-y-km 2600 --source gfs --start 2022092612 --hours 84 --out runs/ian-2way

A live parent (--concurrent, or any --parent-* option) runs one-way by default: the nest takes its boundaries from the
live parent every parent step (25 s), as the operational NAM ran its nests (nest_mode: 1-way). --two-way adds NMMB's
two-way feedback (the nest's T, Q, CW, U, V box means replace the parent's in the overlap, weight --2way-wgt, default
1.0 as NMMB's own two-way test). Without any of these `woof-nam nest` is the one-domain door, and --source parent
--parent-run is the one-way hand-off door, both unchanged.

Stages: domain (parent configfile, nest configfile on the parent's grid: domain.nest_grid_associated), preflight (both
domains in one process), renderer, warm, static (both), fetch (the parent's area holds the nest), wait-vram, init (each
domain's start saved once with namgpu.init.slab_init in its own process), forecast (namgpu.nest.twoway), maps (both),
sheets (the nest). Output: <out>/parent/ and <out>/nest/ (ordinary run folders: wrfout_d01_*, maps/), receipt
<out>/nest-receipt.json with the coupled run's twoway-receipt.json inside.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from namgpu.nest import cli, domain


def add_args(n) -> None:
    n.add_argument("--two-way", action="store_true", help="a 12 km parent and this nest run together in one "
                   "process, with NMMB's two-way feedback (the nest's T, Q, CW, U, V replace the parent's in the "
                   "overlap); off by default: a live parent runs one-way (--concurrent)")
    n.add_argument("--concurrent", action="store_true", help="a live parent with the feedback off: the nest takes "
                   "its boundaries from the live parent every parent step, as the operational NAM nests ran; the default "
                   "whenever a live parent is asked for (any --parent-* option)")
    n.add_argument("--parent-center", default=None, help="--two-way/--concurrent: the parent's centre (default "
                   "--center)")
    n.add_argument("--parent-size-km", type=float, default=None, help="--two-way/--concurrent: parent east-west "
                   "size (default --size-km + 800)")
    n.add_argument("--parent-size-y-km", type=float, default=None, help="parent north-south size (default the nest's "
                   "north-south size + 800)")
    n.add_argument("--parent-res-km", type=float, default=12.0, help="--two-way/--concurrent: the live parent's spacing, "
                   "km, a RES_TABLE row (default 12, the NAM parent; 3 runs a 3 km parent straight from the source, "
                   "e.g. IFS 0.1 deg -> 3 -> 1.5 km); each nest a whole space ratio of the domain before")
    n.add_argument("--2way-wgt", dest="twoway_wgt", type=float, default=1.0, help="weight of the nest in the parent's "
                   "overlap, 0-1 (NMMB 2way_wgt; default 1.0)")
    n.add_argument("--ladder", default=None, metavar="RES:SIZE[,RES:SIZE...]", help="with a live parent: more nests, "
                   "each inside the one before, on the same pole (e.g. --res-km 3 --ladder 1.5:240 runs 12 -> 3 -> "
                   "1.5 km, the 1.5 km nest 240 km wide); every domain steps together one-way (namgpu.nest.coupled), "
                   "each a whole space ratio of the one before; centred on --center. RES:SIZExSIZE_Y gives a step its own "
                   "north-south size (e.g. 1.5:600x400)")
    n.add_argument("--move-nest", action="store_true", help="--concurrent: the nest follows the storm inside the live "
                   "parent (namgpu/nest/live_move.py): it re-centres on the smoothed sea-level-pressure minimum by whole "
                   "parent cells, in the same process; off by default")
    n.add_argument("--move-every-min", type=float, default=15.0, help="--move-nest: minutes of model time between "
                   "storm fixes (default 15)")
    n.add_argument("--move-cells", type=int, default=3, help="--move-nest: move when the storm is more than this many "
                   "parent cells from the nest centre in either direction (default 3)")
    n.add_argument("--move-reach-km", default=None, metavar="X[,Y]", help="--move-nest: the area the nest may visit "
                   "(its fine static and surface fields are prepared once for it; default the parent less 2 x 150 km)")
    n.add_argument("--move-reach-center", default=None, help="--move-nest: centre of that area (default the parent's)")
    n.add_argument("--move-null-at-h", type=float, default=None, help="--move-nest proof: a forced move of 0 cells at "
                   "this hour (must change nothing)")
    n.add_argument("--move-smooth-km", type=float, default=60.0, help="--move-nest: box smoothing of sea-level pressure "
                   "before its minimum is taken as the storm centre (default 60 km: a broad low has several minima)")
    n.add_argument("--move-seam-cells", type=int, default=3, help="--move-nest: kept cells beside the new ground where "
                   "the nest-minus-parent difference is measured and carried into the new ground (0: none; default 3)")
    n.add_argument("--move-blend-cells", type=int, default=12, help="--move-nest: cells into the new ground over which "
                   "that difference fades to the parent's values (default 12)")


def ladder_grids(spec: str | None, ng: dict, lat: float, lon: float) -> list[dict]:
    """--ladder RES:SIZE[,RES:SIZE...]: each nest on the grid of the one before (domain.nest_grid_associated), so a
    ladder is rows of the same rule, not a new code path. [] without --ladder."""
    out, prev = [], ng
    for item in [s for s in (spec or "").split(",") if s.strip()]:
        try:
            res_s, size_s = item.split(":")
            sx, _, sy = size_s.lower().partition("x")
            res, size = float(res_s), float(sx)
            size_y = float(sy) if sy else None
        except ValueError:
            raise SystemExit(f"--ladder {spec}: each step is RES:SIZE or RES:SIZExSIZE_Y in km (e.g. 1.5:240, "
                             "1.5:600x400)") from None
        g = domain.nest_grid_associated(prev, lat, lon, size, res, size_y)
        g["size_km"] = size
        g["size_y_km"] = size_y or size
        out.append(g)
        prev = g
    return out


def parent_res(value) -> float | int:
    """--parent-res-km: the live parent's spacing, a RES_TABLE row (12 by default). Breakage it prevents: a spacing
    with no row would write no configfile keys and stop at the static build after the fetch."""
    r = float(value or 12.0)
    r = int(r) if r == int(r) else r
    if r not in domain.RES_TABLE:
        raise SystemExit(f"--parent-res-km {r:g}: no row in namgpu/nest/domain.py RES_TABLE "
                         f"({', '.join(f'{k:g}' for k in domain.RES_TABLE)})")
    return r


def _combined_pool(es: list) -> tuple[float, float]:
    """(working pool, pool cap) of every domain in one process, GiB: the domains' working pools add (state, the
    path's fixed part, the radiation rows), the GFS decode floor (domain.INIT_POOL_GIB: whole-globe records held while
    a start or a boundary time is decoded) counts once, as one decode runs at a time in the process (each domain's
    start is saved by its own process before the forecast; in the forecast only the outermost reads the source), and
    the cap's headroom once. Breakage it prevents: the floor was added for every domain, so a 12 -> 3 -> 1.5 km ladder
    of 53,000 columns was estimated at 13.15 GiB and held 4.80 GiB at its peak (18 Nov), and the 125,000-column
    hurricane ladder (16.18 GiB estimated) could not start on a shared 32 GB card or any 16 GB one."""
    work = sum(e["total_gib"] - e["lmem_gib"] - e["fixed_gib"]
               - (domain.VRAM_PATH[e["memory_path"] or "full"]["pool_headroom_gib"] if e["memory_path"] != "fit" else 0.0)
               for e in es)
    work = max(work, sum(e["state_gib"] for e in es) + domain.INIT_POOL_GIB)
    head = max(domain.VRAM_PATH[e["memory_path"] or "full"]["pool_headroom_gib"] for e in es)
    return work, work + head


def _combined_preflight(pg: dict, ng: dict, rows_given, index: str, more: tuple = ()) -> dict:
    """Every domain in one CUDA context: the working pools add, the GFS decode floor, the cap's headroom, the
    local-memory reservation and the context count once (domain.vram_estimate model; _combined_pool)."""
    free = domain.free_gib(index)
    tried = []
    for r in ((rows_given,) if rows_given else domain.RRTM_ROW_CHOICES):
        ep = domain.vram_estimate(pg, r, 1, index)
        ec = domain.vram_estimate(ng, r, 1, index)
        es = [ep, ec] + [domain.vram_estimate(g, r, 1, index) for g in more]
        work, cap = _combined_pool(es)
        tot = work + ec["lmem_gib"] + ec["fixed_gib"] + (cap - work if ec["memory_path"] != "fit" else 0.0)
        pool = cap
        e = dict(ec, total_gib=round(tot, 2), pool_limit_gib=round(pool, 2) if ec["pool_limit_bytes"] else None,
                 pool_limit_bytes=int(pool * 2**30) if ec["pool_limit_bytes"] else None, parent=ep, nest=ec)
        tried.append(e)
        if tot + domain.VRAM["margin_gib"] <= free:
            return {"ok": True, "free_gib": round(free, 2), "pick": e, "tried": tried, "card": domain.card_name(index)}
    return {"ok": False, "free_gib": round(free, 2), "tried": tried, "card": domain.card_name(index)}


def main(a, argv) -> int:
    if a.two_way and a.concurrent:
        raise SystemExit("--two-way and --concurrent: pick one (--two-way is --concurrent plus the feedback)")
    if a.cards != 1:
        raise SystemExit("--two-way/--concurrent run both domains on one card in this release (--cards 1)")
    if a.source == "parent":
        raise SystemExit("--two-way/--concurrent start both domains from a source analysis (gfs, ifs, ...); "
                         "--source parent is the one-way hand-off door")
    t_all = time.perf_counter()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lat, lon = cli.parse_center(a.center)
    plat, plon = cli.parse_center(a.parent_center) if a.parent_center else (lat, lon)
    cycle = datetime.strptime(a.start, "%Y%m%d%H")
    hours = float(a.hours)
    source, region = (domain.default_source(lat, lon, cycle) if a.source == "auto" else (a.source, "given"))
    boco_age, boco_why = cli.boco_age_for(source, cycle, a.boco_age_h)
    credit, credit_lines = cli.data_credits(source)
    sampler = cli.Sampler().start()
    mode = "two-way" if a.two_way else "concurrent"
    stages, rec = {}, {"door": f"woof-nam nest --{mode}", "argv": sys.argv[1:] if argv is None else list(argv),
                       "start": cycle.isoformat(), "hours": hours, "source": source, "source_region": region,
                       "boco_age_h": boco_age, "boco_age_why": boco_why, "data_credit": credit,
                       "data_licences": credit_lines, "host": os.uname().nodename, "mode": mode}

    def stage(name):
        sampler.mark(name)
        stages[name] = {"t0": time.perf_counter()}

    def done(name, **kw):
        s = stages[name]
        s["wall_s"] = round(time.perf_counter() - s.pop("t0"), 2)
        s.update(kw)
        print(f"woof-nam: {name} {s['wall_s']} s", flush=True)

    def finish(code):
        cli._write(out, rec, stages, sampler, t_all)
        return code

    # domain: the parent (own pole at its centre) and the nest on the parent's grid
    stage("domain")
    psx = a.parent_size_km or (a.size_km + 800.0)
    psy = a.parent_size_y_km or ((a.size_y_km or a.size_km) + 800.0)
    pres = parent_res(getattr(a, "parent_res_km", None))
    pg = domain.nest_grid(plat, plon, psx, pres, psy)
    ng = domain.nest_grid_associated(pg, lat, lon, a.size_km, a.res_km, a.size_y_km)
    pname = domain.domain_name(pg, psx, psy)
    name = domain.domain_name(ng, a.size_km, a.size_y_km) + f"_in{domain.grid_key(pg)[:6]}"
    from namgpu.init import static_build as _sb
    cli.STATIC_CACHE.mkdir(parents=True, exist_ok=True)
    pstatic = cli.STATIC_CACHE / f"{pname}-{domain.grid_key(pg)}-ds{_sb.datasets_key()}-{cycle:%Y%m%d}.npz"
    nstatic = cli.STATIC_CACHE / f"{name}-{domain.grid_key(ng)}-ds{_sb.datasets_key()}-{cycle:%Y%m%d}.npz"
    pcfg, ncfg = out / f"nam_configfile_{pname}", out / f"nam_configfile_{name}"
    pkeys = domain.write_configfile(pcfg, pg, pname, pstatic, False, handoff=False)
    nkeys = domain.write_configfile(ncfg, ng, name, nstatic, a.spec_adv, handoff=False)
    rec.update(parent={"domain": pname, "grid": pg, "configfile": str(pcfg), "keys": pkeys},
               nest={"domain": name, "grid": ng, "configfile": str(ncfg), "keys": nkeys}, domain=name, grid=ng,
               twoway_wgt=a.twoway_wgt if a.two_way else None)
    # --ladder: more nests, each on the grid of the one before; roles parent, nest, nest2, nest3, ...
    lgrids = ladder_grids(getattr(a, "ladder", None), ng, lat, lon)
    if lgrids and a.two_way:
        raise SystemExit("--ladder runs every domain one-way (namgpu.nest.coupled has no feedback): drop --two-way")
    moving = bool(getattr(a, "move_nest", False))
    env_role = None
    if moving:
        if a.two_way:
            raise SystemExit("--move-nest runs one-way (two-way feedback during moves is not built): drop --two-way")
        if lgrids:
            raise SystemExit("--move-nest with --ladder: nests that move with a moving nest are not built")
        if ng["jm"] == ng["im"]:
            raise SystemExit("--move-nest: give the nest a different north-south size (--size-y-km): a square window "
                             "cannot tell per-row from per-column arrays (namgpu/nest/move.py)")
        if a.move_reach_km:
            xy = [float(x) for x in a.move_reach_km.split(",")]
            rx, ry = (xy[0], xy[1]) if len(xy) > 1 else (xy[0], xy[0])
        else:
            rx, ry = psx - 300.0, psy - 300.0
        rlat, rlon = cli.parse_center(a.move_reach_center) if a.move_reach_center else (plat, plon)
        eg = domain.nest_grid_associated(pg, rlat, rlon, rx, a.res_km, ry)
        r_ = ng["parent_child_space_ratio"]
        if eg["im"] == ng["im"] or eg["jm"] == ng["jm"] or eg["im"] == eg["jm"]:
            eg = domain.nest_grid_associated(pg, rlat, rlon, rx + pg["dx_km"], a.res_km, ry + 2 * pg["dy_km"])
        ei0 = (eg["i_parent_start"] - ng["i_parent_start"]) * r_
        ej0 = (eg["j_parent_start"] - ng["j_parent_start"]) * r_
        if ei0 > 0 or ej0 > 0 or ei0 + eg["im"] < ng["im"] or ej0 + eg["jm"] < ng["jm"]:
            raise SystemExit(f"--move-nest: the reach ({eg['extent_km']} km at {rlat}, {rlon}) does not hold the "
                             "first nest: widen --move-reach-km or move --move-reach-center")
        from namgpu.nest import move as _mv
        ename = name + "-reach"
        estatic = cli.STATIC_CACHE / f"{ename}-{domain.grid_key(eg)}-ds{_sb.datasets_key()}-{cycle:%Y%m%d}.npz"
        ecfg = out / f"nam_configfile_{ename}"
        _mv.window_configfile(ncfg, ecfg, ei0, ej0, ename, estatic, extra={"im": eg["im"], "jm": eg["jm"]})
        env_role = ("envelope", ecfg, estatic, eg)
        rec["move"] = {"envelope": {"grid": eg, "i0": ei0, "j0": ej0, "configfile": str(ecfg)},
                       "every_min": a.move_every_min, "cells": a.move_cells}
        rec["door"] = "woof-nam nest --concurrent --move-nest"
        print(f"woof-nam: moving nest: reach {eg['im']} x {eg['jm']} ({eg['extent_km']} km) at offset ({ei0}, {ej0}) "
              f"nest cells; a fix every {a.move_every_min:g} min, a move past {a.move_cells} parent cells", flush=True)
    roles = [("parent", pcfg, pstatic, pg), ("nest", ncfg, nstatic, ng)]
    prev = ng
    for k, g in enumerate(lgrids, start=2):
        gname = domain.domain_name(g, g["size_km"], g["size_y_km"]) + f"_in{domain.grid_key(prev)[:6]}"
        gstatic = cli.STATIC_CACHE / f"{gname}-{domain.grid_key(g)}-ds{_sb.datasets_key()}-{cycle:%Y%m%d}.npz"
        gcfg = out / f"nam_configfile_{gname}"
        gkeys = domain.write_configfile(gcfg, g, gname, gstatic, a.spec_adv, handoff=False)
        roles.append((f"nest{k}", gcfg, gstatic, g))
        rec[f"nest{k}"] = {"domain": gname, "grid": g, "configfile": str(gcfg), "keys": gkeys}
        prev = g
    if lgrids:
        rec["ladder"] = [f"{g['res_km']} km" for _r, _c, _s, g in roles]
        rec["door"] = "woof-nam nest --concurrent --ladder"
    done("domain")
    print(f"woof-nam: parent {pname} {pg['im']} x {pg['jm']}; nest {name} {ng['im']} x {ng['jm']} x {ng['lm']} "
          f"({ng['extent_km']} km, ratio {ng['parent_child_space_ratio']}, SW corner on parent H "
          f"({ng['i_parent_start']}, {ng['j_parent_start']})), {mode}, source {source}", flush=True)
    for role, _c, _s, g in roles[2:]:
        print(f"woof-nam: {role} {g['im']} x {g['jm']} x {g['lm']} ({g['extent_km']} km, ratio "
              f"{g['parent_child_space_ratio']} in the domain before, SW corner on its H ({g['i_parent_start']}, "
              f"{g['j_parent_start']}))", flush=True)

    stage("preflight")
    index = domain._visible_index()
    pf = _combined_preflight(pg, ng, a.rrtm_rows, index, tuple(g for _r, _c, _s, g in roles[2:]))
    rec["preflight"] = pf
    done("preflight")
    if not pf["ok"] and not a.dry_run and a.wait_min <= 0:
        rec["refused"] = (f"woof-nam: refused: parent + nest need about {pf['tried'][-1]['total_gib']} GiB and the card "
                          f"has {pf['free_gib']} GiB free ({pf['card']})")
        print(rec["refused"], flush=True)
        return finish(3)
    rows = pf["pick"]["rrtm_rows"] if pf["ok"] else (a.rrtm_rows or 8)

    if not (a.no_maps and a.no_sheets):
        stage("renderer")
        from namgpu.io import maps as _maps
        try:
            rw = _maps.renderer()
        except SystemExit as e:
            done("renderer", ok=False)
            rec["failed_stage"] = f"renderer: {e}"
            return finish(6)
        done("renderer", ok=True, path=str(rw))
    if not a.no_warm:
        stage("warm")
        from namgpu import kernels
        w = kernels.warm()
        done("warm", **{k: w[k] for k in ("workers", "modules", "compile_s_sum", "failed", "slowest")})

    env = dict(os.environ, NAMGPU_RRTM_ROWS=str(rows), PYTHONUNBUFFERED="1", NAMGPU_DATA_CREDIT=credit,
               NAMGPU_DATA_LICENCES=json.dumps(credit_lines), NAMGPU_START_SOURCE=source,
               NAMGPU_BOCO_AGE_H=str(boco_age),
               PYTHONPATH=str(cli.ROOT) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    a.boco_age_h = boco_age

    stage("static")
    for _role, cfgfile, static, _g in roles + ([env_role] if env_role else []):
        stale = cli._static_stale(static) if static.is_file() else None
        if static.is_file() and not stale:
            continue
        code, _ = cli.run_logged([sys.executable, "-m", "namgpu.init.static_build", "--config", str(cfgfile), "--date",
                                  f"{cycle:%Y-%m-%d}", "--out", str(static)], dict(env, NAMGPU_CONFIGFILE=str(cfgfile)),
                                 out / "static.log", sampler)
        if code:
            done("static", rc=code)
            return finish(code)
    done("static", parent=str(pstatic), nest=str(nstatic))
    rec["static_meta"] = {role: cli._static_meta(static) for role, _c, static, _g in roles}
    if a.dry_run:
        return finish(0)

    stage("fetch")
    case = f"{cycle:%Y-%m-%d_%H}z"
    case_dir = (a.cases_root / case).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    fetch, lic = cli.resolve_fetch(source)
    from namgpu import config as configmod
    a.source_name = source
    fr = fetch(cycle, case_dir, max(6, int(hours + 0.999)), configmod.load(pcfg), a)   # the parent's area holds the nest
    rec["source_licence"] = lic
    done("fetch", **fr)
    if fr.get("failed"):
        return finish(4)

    stage("wait-vram")
    t_lim = time.perf_counter() + a.wait_min * 60
    while True:
        pf = _combined_preflight(pg, ng, a.rrtm_rows, index, tuple(g for _r, _c, _s, g in roles[2:]))
        if pf["ok"] or time.perf_counter() > t_lim:
            break
        time.sleep(30)
    done("wait-vram", free_gib=pf["free_gib"], ok=pf["ok"])
    rec["preflight_at_forecast"] = pf
    if not pf["ok"]:
        rec["refused"] = f"waited {a.wait_min} min: {pf['free_gib']} GiB free, need {pf['tried'][-1]['total_gib']} GiB"
        return finish(3)
    env["NAMGPU_RRTM_ROWS"] = str(pf["pick"]["rrtm_rows"])

    # init: each domain's start saved once, in its own process (its own configfile), then loaded by the coupled run
    shm = Path("/dev/shm")
    base = Path(os.environ.get("NAMGPU_INIT_DIR", str(shm if shm.is_dir() else out)))
    inits = {role: base / f"{Path(str(cfgfile)).name[len('nam_configfile_'):]}-{cycle:%Y%m%d%H}-init"
             for role, cfgfile, _s, _g in roles}
    if env_role:
        inits["envelope"] = base / f"{Path(str(env_role[1])).name[len('nam_configfile_'):]}-{cycle:%Y%m%d%H}-init"
    stage("init")
    for role, cfgfile, _s, _g in roles + ([env_role] if env_role else []):
        # the nest's own boundaries serve only its TDFI (boco.0000): 1 h asks for the 6 h minimum
        h = hours if role == "parent" else 1.0
        # the reach: held on the host after; may spill while built; its own TDFI, so the physics state the new ground
        # takes from it (surface-layer, radiation and PBL fields) is a stepped one, not the zeros of a cold start
        extra = ["--managed", "--tdfi"] if role == "envelope" else []
        code, _ = cli.run_logged([sys.executable, "-m", "namgpu.init.slab_init", "save", "--case", case, "--case-dir",
                                  str(case_dir), "--source", source, "--hours", str(h), "--out", str(inits[role])]
                                 + extra, dict(env, NAMGPU_CONFIGFILE=str(cfgfile)), out / f"init-{role}.log", sampler)
        if code:
            done("init", rc=code, failed=role)
            _rm(inits, rec)
            return finish(code)
    done("init", dirs={k: str(v) for k, v in inits.items()})

    stage("forecast")
    if pf.get("pick", {}).get("pool_limit_bytes"):
        env["CUPY_GPU_MEMORY_LIMIT"] = str(pf["pick"]["pool_limit_bytes"])
        rec["pool_limit_gib"] = pf["pick"]["pool_limit_gib"]
    if lgrids or moving:
        # a ladder: the coupled runner steps the tree (parent, then each nest RATIO times inside its parent's step)
        plan = {"domains": [{"name": role, "config": str(cfgfile), "init": str(inits[role]),
                             "parent": None if k == 0 else roles[k - 1][0]}
                            for k, (role, cfgfile, _s, _g) in enumerate(roles)]}
        if moving:
            plan["domains"][1]["move"] = {"base_config": str(ncfg), "envelope": str(inits["envelope"]),
                                          "env_i0": rec["move"]["envelope"]["i0"],
                                          "env_j0": rec["move"]["envelope"]["j0"],
                                          "every_min": a.move_every_min, "cells": a.move_cells,
                                          "null_move_at_h": a.move_null_at_h,
                                          "smooth_cells": 2 * int(round(a.move_smooth_km / ng["dx_km"] / 2.0)) + 1,
                                          "seam_cells": a.move_seam_cells, "blend_cells": a.move_blend_cells}
        (out / "ladder-plan.json").write_text(json.dumps(plan, indent=1))
        cmd = [sys.executable, "-m", "namgpu.nest.coupled", "--plan", str(out / "ladder-plan.json"), "--case", case,
               "--case-dir", str(case_dir), "--source", source, "--hours", str(hours), "--out", str(out)]
    else:
        cmd = [sys.executable, "-m", "namgpu.nest.twoway", "--parent-config", str(pcfg), "--nest-config", str(ncfg),
               "--parent-init", str(inits["parent"]), "--nest-init", str(inits["nest"]), "--case", case,
               "--case-dir", str(case_dir), "--source", source, "--hours", str(hours), "--out", str(out)]
    if a.two_way:
        cmd += ["--two-way", "--wgt", str(a.twoway_wgt)]
    code, _ = cli.run_logged(cmd, env, out / "run.log", sampler)
    _rm(inits, rec)
    tr = out / ("coupled-receipt.json" if (lgrids or moving) else "twoway-receipt.json")
    if tr.is_file():
        rec["forecast"] = json.loads(tr.read_text())
        rec["forecast"].pop("trace", None) if not rec["forecast"].get("error") else None
    done("forecast", rc=code)
    rec["forecast_rc"] = code
    if code:
        return finish(code)

    if not a.no_maps:
        stage("maps")
        for role, cfgfile, _s, _g in roles:
            seed_domain_receipt(out / role, rec["door"], tr.name)
            mc, _ = cli.run_logged([sys.executable, "-m", "namgpu.run_forecast", "--maps-only", "--case", case,
                                    "--case-dir", str(case_dir), "--out", str(out / role)],
                                   dict(env, NAMGPU_CONFIGFILE=str(cfgfile)), out / f"maps-{role}.log", sampler)
            if mc:
                rec["failed_stage"] = f"maps {role} rc {mc}"
        done("maps")
    if not a.no_sheets:
        stage("sheets")
        sheets_tool = cli.ROOT / "tools" / "nam_domain_sheets.py"
        run_dir = out / "nest"
        if sheets_tool.is_file():
            sc_cmd = [sys.executable, str(sheets_tool), "--run", str(run_dir), "--nest", source, "--out",
                      str(out / "sheets"), "--hours", f"1-{int(hours)}", "--start", f"{cycle:%Y-%m-%dT%H}",
                      "--cache", str(case_dir / "referee-cache")]
        else:
            sc_cmd = [sys.executable, "-m", "namgpu.io.io_referee_sheets", "nest", "--history",
                      *sorted(str(p) for p in run_dir.glob("wrfout_d01_*")), "--source", source, "--out",
                      str(out / "sheets"), "--leads", f"1-{int(hours)}", "--cache", str(case_dir / "referee-cache")]
        sc, _ = cli.run_logged(sc_cmd, dict(env, NAMGPU_CONFIGFILE=str(ncfg)), out / "sheets.log", sampler)
        done("sheets", rc=sc)
        if sc:
            rec["failed_stage"] = f"sheets rc {sc}"
    return finish(5 if rec.get("failed_stage") else 0)


def seed_domain_receipt(run_dir: Path, door: str, coupled_receipt: str) -> Path:
    """<run_dir>/forecast-receipt.json for the maps tail (run_forecast --maps-only), which reads and finishes it. A
    one-domain run writes it; the coupled runners (namgpu.nest.twoway, namgpu.nest.coupled) write one receipt at <out>
    and none per domain. An existing receipt is kept. Breakage it prevents: every live-parent run (--concurrent,
    --two-way, --ladder) stopped at "maps rc 1" (FileNotFoundError forecast-receipt.json) with no maps drawn."""
    frp = Path(run_dir) / "forecast-receipt.json"
    if not frp.is_file():
        frp.write_text(json.dumps({"door": door, "domain": Path(run_dir).name, "exit_code": 0,
                                   "coupled_receipt": str(coupled_receipt)}, indent=1))
    return frp


def _rm(inits: dict, rec: dict) -> None:
    for d in inits.values():
        shutil.rmtree(d, ignore_errors=True)
    rec["init_dirs_removed"] = [str(d) for d in inits.values()]
