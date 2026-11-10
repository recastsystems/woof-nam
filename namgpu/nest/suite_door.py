"""woof-nam nest --ncep-nest NAME[,NAME...] --concurrent: NCEP's own NAM nests run as NCEP ran them, one-way inside the
live NAM 12 km parent, on one card or on N cards (every domain split in rows over the same N ranks).

    woof-nam nest --ncep-nest conusnest --concurrent --source gfs --start 2026110500 --hours 84 --nest-hours 60 \
        --out runs/suite-2026110500 --cards 8
    (smoke on one card: --ncep-window 80x60 --parent-margin 30: a centred 80 x 60 parent-cell window of the nest, on
     the same pole, spacing, dt and physics, inside a crop of the 12 km grid 30 parent cells wider on every side)

The parent is the NAM 12 km grid itself (parm/nam_configfile: 954 x 835 x 60, the conus12 domain row, its NAM-analysis
static), so the run IS the 12 km forecast the public 12 km product comes from, with the nests inside it. Each nest is
a row of namgpu/nest/ncep_nests.py (NCEP's im/jm/i_parent_start/j_parent_start/ratio on the parent's pole), with NCEP's
conusnest physics and step (domain.RES_TABLE 3 km row, domain.NEST_PHYSICS) and spec_adv .true.; its static is built on
the GPU from WPS_GEOG with the permissive set (static_build). Every nest stops at --nest-hours (NCEP: 60) and the parent
runs on alone to --hours (NCEP: 84; exnam_fcst.sh LENGTH=60, then exnam_fcst_parentext.sh).

Stages: domain, preflight (every card), static (parent and nests), fetch (the parent's area holds the nests),
init (each domain's start saved once with namgpu.init.slab_init, the domains side by side on different cards when there
are several), forecast (namgpu.nest.coupled), maps (each domain, Rust renderer), receipt <out>/nest-receipt.json.
Output: <out>/parent/ and <out>/<nest>/ (ordinary run folders: wrfout_d01_*, maps/).
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from namgpu.nest import cli, domain

PARENT = "parent"


def parent_crop(nests: list[dict], margin: int) -> dict:
    """A crop of the NAM 12 km grid (same pole and spacing) holding every nest with `margin` parent cells on each side,
    and the nests' starts moved onto it. For a one-card smoke of the suite's geometry; never a product grid."""
    from namgpu.nest import ncep_nests
    p = ncep_nests.parent_grid()
    i0 = min(g["i_parent_start"] for g in nests) - margin
    j0 = min(g["j_parent_start"] for g in nests) - margin
    i1 = max(g["i_parent_start"] + math.ceil(g["parent_cells"][0]) for g in nests) + margin
    j1 = max(g["j_parent_start"] + math.ceil(g["parent_cells"][1]) for g in nests) + margin
    i0, j0, i1, j1 = max(1, i0), max(1, j0), min(p["im"], i1), min(p["jm"], j1)
    c = dict(p, im=i1 - i0 + 1, jm=j1 - j0 + 1, wbd=round(p["wbd"] + (i0 - 1) * p["dlmd"], 6),
             sbd=round(p["sbd"] + (j0 - 1) * p["dphd"], 6), crop=[i0, j0, i1, j1])
    for g in nests:
        g["i_parent_start"] -= i0 - 1
        g["j_parent_start"] -= j0 - 1
    return c


def _parent_configfile(path: Path, crop: dict | None, static: Path | None) -> tuple[Path, dict]:
    """The NAM 12 km configfile itself (parm/nam_configfile: the conus12 domain row, so domains.ensure_static gives it
    the NAM-analysis static); a crop gets a copy with its own size, corner, spacing (explicit: off centre on the NAM
    pole), static and boco folder name."""
    from namgpu import config
    base = (domain.PARM / "nam_configfile").read_text()
    if crop is None:
        path = domain.PARM / "nam_configfile"
    else:
        keys = {"im": str(crop["im"]), "jm": str(crop["jm"]), "wbd": f"{crop['wbd']:.6f}", "sbd": f"{crop['sbd']:.6f}",
                "dlmd": f"{crop['dlmd']:.9f}", "dphd": f"{crop['dphd']:.9f}", "static_npz": str(static),
                "domain_name": "nam12crop_" + "_".join(str(x) for x in crop["crop"])}
        path.write_text(domain.render(base, keys, "12 km crop"))
    c = config.load(path)
    return path, {"im": c.im, "jm": c.jm, "dlmd": round(c.dlmd, 9), "dphd": round(c.dphd, 9), "dt": float(c.dt)}


def _rows_ok(rows: int, ims: list[int]) -> bool:
    """RRTM chunk rows x IM must be a multiple of 8 (the taumol chunk; IM 1827 is odd, so rows 4 is refused at once)."""
    return all((rows * im) % 8 == 0 for im in ims)


def _preflight(grids: dict, cards: list[str], rows_given) -> dict:
    """Every domain's slab on every card, in one process per card: the pool parts add, the local-memory reservation
    and the context count once (domain.vram_estimate per domain with the card count)."""
    free = {c: domain.free_gib(c) for c in cards}
    ims = [g["im"] for g in grids.values()]
    tried = []
    for r in ((rows_given,) if rows_given else domain.RRTM_ROW_CHOICES):
        if not _rows_ok(r, ims):
            continue
        per = {}
        for c in cards:
            es = {n: domain.vram_estimate(g, r, len(cards), c) for n, g in grids.items()}
            e0 = next(iter(es.values()))
            # each domain's whole-run working pool (its peak estimate less the local memory and context, which count
            # once per process); the cap adds the row's headroom once. The per-domain cap's init floor
            # (domain.INIT_POOL_GIB) is not summed: the inits run in their own processes (stage init), each on ONE
            # card for its whole domain, and each one alone must fit (init_gib: the fit-path model on the whole
            # domain, its floor included; the 3 km CONUS init held a 41 GiB state, nam3km lane 2026-11-05)
            work = sum(e["total_gib"] - e["lmem_gib"] - e["fixed_gib"] for e in es.values())
            head = domain.VRAM_PATH[e0["memory_path"] or "full"]["pool_headroom_gib"]
            per[c] = {"total_gib": round(work + e0["lmem_gib"] + e0["fixed_gib"], 2), "rrtm_rows": r,
                      "memory_path": e0["memory_path"], "domains": es,
                      "init_gib": round(max((e["pool_limit_gib"] or 0) + e["lmem_gib"] + e["fixed_gib"]
                                            for e in (domain.vram_estimate(g, r, 1, c, path="fit")
                                                      for g in grids.values())), 2),
                      "pool_limit_gib": round(work + head, 2) if e0["pool_limit_bytes"] else None}
        worst = max(per.values(), key=lambda x: x["total_gib"])
        tried.append(worst)
        if all(max(per[c]["total_gib"], per[c]["init_gib"]) + domain.VRAM["margin_gib"] <= free[c] for c in cards):
            return {"ok": True, "free_by_card": {c: round(v, 2) for c, v in free.items()}, "pick": worst,
                    "tried": tried, "card": domain.card_name(cards[0])}
    return {"ok": False, "free_by_card": {c: round(v, 2) for c, v in free.items()}, "tried": tried,
            "card": domain.card_name(cards[0])}


def main(a, argv) -> int:
    from namgpu import config as configmod
    from namgpu.init import static_build as _sb
    from namgpu.nest import ncep_nests
    if a.two_way:
        raise SystemExit("--ncep-nest --two-way: NCEP ran every NAM nest one-way (nest_mode: 1-way); on several cards "
                         "the feedback's parent update points span ranks and no nest-to-parent gather is written. "
                         "Use --concurrent (one-way)")
    if a.source == "parent":
        raise SystemExit("--ncep-nest --concurrent starts the parent and the nests from a source analysis (gfs, ...)")
    t_all = time.perf_counter()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cycle = datetime.strptime(a.start, "%Y%m%d%H")
    names = [n.strip() for n in a.ncep_nest.split(",") if n.strip()]
    win = off = None
    if a.ncep_window:
        import re
        m = re.fullmatch(r"(\d+)[xX](\d+)(?:\+(\d+)\+(\d+))?", a.ncep_window.strip())
        if not m:
            raise SystemExit(f"--ncep-window {a.ncep_window!r}: want NXxNY or NXxNY+I+J parent cells")
        win = (int(m.group(1)), int(m.group(2)))
        off = (int(m.group(3)), int(m.group(4))) if m.group(3) is not None else None
    nests = {n: ncep_nests.nest_grid(n, win, off) for n in names}
    rows = {n: ncep_nests.row(n) for n in names}
    hours = float(a.hours if a.hours is not None else 84.0)
    nest_hours = float(a.nest_hours if a.nest_hours is not None else min(hours, max(r["hours"] for r in rows.values())))
    source = a.source if a.source != "auto" else "gfs"
    boco_age, boco_why = cli.boco_age_for(source, cycle, a.boco_age_h)
    credit, credit_lines = cli.data_credits(source)
    cards = ([c.strip() for c in a.card_list.split(",") if c.strip()] or [str(i) for i in range(a.cards)])[:a.cards] \
        if a.cards > 1 else [domain._visible_index()]
    sampler = cli.Sampler().start()
    stages = {}
    rec = {"door": "woof-nam nest --ncep-nest --concurrent", "argv": sys.argv[1:] if argv is None else list(argv),
           "start": cycle.isoformat(), "hours": hours, "nest_hours": nest_hours, "source": source,
           "boco_age_h": boco_age, "boco_age_why": boco_why, "data_credit": credit, "data_licences": credit_lines,
           "host": os.uname().nodename, "mode": "concurrent one-way", "cards": cards}

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

    stage("domain")
    cli.STATIC_CACHE.mkdir(parents=True, exist_ok=True)
    margin = a.parent_margin if a.parent_margin is not None else (30 if win else None)
    crop = parent_crop(list(nests.values()), int(margin)) if margin is not None else None
    pg = crop or ncep_nests.parent_grid()
    pstatic = None
    if crop is not None:
        pname = "nam12crop_" + "_".join(str(x) for x in crop["crop"])
        pstatic = cli.STATIC_CACHE / f"{pname}-{domain.grid_key(crop)}-ds{_sb.datasets_key()}-{cycle:%Y%m%d}.npz"
    pcfg, pchk = _parent_configfile(out / "nam_configfile_parent", crop, pstatic)
    dom_specs = [{"name": PARENT, "config": str(pcfg), "parent": None}]
    nstatic, ncfg, nchk = {}, {}, {}
    for n, g in nests.items():
        dname = ncep_nests.domain_name(g)
        nstatic[n] = cli.STATIC_CACHE / f"{dname}-{domain.grid_key(g)}-ds{_sb.datasets_key()}-{cycle:%Y%m%d}.npz"
        ncfg[n] = out / f"nam_configfile_{n}"
        nchk[n] = domain.write_configfile(ncfg[n], g, dname, nstatic[n], bool(rows[n].get("spec_adv")), handoff=False)
        dom_specs.append({"name": n, "config": str(ncfg[n]), "parent": rows[n].get("parent", PARENT)})
    rec.update(parent={"grid": pg, "configfile": str(pcfg), "check": pchk, "crop": crop is not None},
               nests={n: {"grid": nests[n], "configfile": str(ncfg[n]), "keys": nchk[n], "src": rows[n]["src"],
                          "hiresf": rows[n].get("hiresf", {}).get("file")} for n in names})
    done("domain")
    for n, g in nests.items():
        print(f"woof-nam: {n} {g['im']} x {g['jm']} x {g['lm']} on the NAM 12 km grid ({g['dx_km']} km, ratio "
              f"{g['parent_child_space_ratio']}, SW corner on parent H ({g['i_parent_start']}, {g['j_parent_start']})"
              f"{', window ' + a.ncep_window if win else ''}); parent {pg['im']} x {pg['jm']}"
              f"{' (crop ' + str(crop['crop']) + ')' if crop else ''}; {nest_hours:g} h nests, {hours:g} h parent, "
              f"{len(cards)} card{'s' if len(cards) > 1 else ''}", flush=True)

    stage("preflight")
    grids = {PARENT: pg, **nests}
    pf = _preflight(grids, cards, a.rrtm_rows)
    rec["preflight"] = pf
    done("preflight")
    est = pf["tried"][-1]["total_gib"] if pf["tried"] else None
    print(f"woof-nam: estimate {pf.get('pick', pf['tried'][-1] if pf['tried'] else {}).get('total_gib')} GiB per card "
          f"({pf['card']}, free {pf['free_by_card']})", flush=True)
    if not pf["ok"] and not a.dry_run and a.wait_min <= 0:
        rec["refused"] = (f"woof-nam: refused: the parent and {', '.join(names)} need about {est} GiB per card on "
                          f"{len(cards)} card(s) and the cards have {pf['free_by_card']} GiB free ({pf['card']}); "
                          f"use more cards")
        print(rec["refused"], flush=True)
        return finish(3)
    rrtm = pf["pick"]["rrtm_rows"] if pf["ok"] else (a.rrtm_rows or 8)

    if not a.no_warm:
        stage("warm")
        from namgpu import kernels
        w = kernels.warm()
        done("warm", **{k: w[k] for k in ("workers", "modules", "compile_s_sum", "failed", "slowest")})
    env = dict(os.environ, NAMGPU_RRTM_ROWS=str(rrtm), PYTHONUNBUFFERED="1", NAMGPU_DATA_CREDIT=credit,
               NAMGPU_DATA_LICENCES=json.dumps(credit_lines), NAMGPU_START_SOURCE=source,
               NAMGPU_BOCO_AGE_H=str(boco_age),
               PYTHONPATH=str(cli.ROOT) + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""))
    a.boco_age_h = boco_age

    stage("static")
    case = f"{cycle:%Y-%m-%d_%H}z"
    case_dir = (a.cases_root / case).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    builds = [(ncfg[n], nstatic[n]) for n in names] + ([(pcfg, pstatic)] if pstatic is not None else [])
    for cfgfile, static in builds:
        stale = cli._static_stale(static) if static.is_file() else None
        if static.is_file() and not stale:
            continue
        code, _ = cli.run_logged([sys.executable, "-m", "namgpu.init.static_build", "--config", str(cfgfile), "--date",
                                  f"{cycle:%Y-%m-%d}", "--out", str(static)],
                                 dict(env, NAMGPU_CONFIGFILE=str(cfgfile), CUDA_VISIBLE_DEVICES=cards[0]),
                                 out / "static.log", sampler)
        if code:
            done("static", rc=code)
            return finish(code)
    if pstatic is None:
        from namgpu import domains
        os.environ[configmod.CONFIG_ENV] = str(pcfg)
        prow = domains.ensure_static(case_dir, cycle, source, card=cards[0])
    else:
        prow = {"file": str(pstatic), "action": "built or cached"}
    done("static", parent=prow, nests={n: str(nstatic[n]) for n in names})
    rec["static_meta"] = {n: cli._static_meta(nstatic[n]) for n in names}
    if a.dry_run:
        return finish(0)

    stage("fetch")
    fetch, lic = cli.resolve_fetch(source)
    a.source_name = source
    fr = fetch(cycle, case_dir, max(6, int(hours + 0.999)), configmod.load(pcfg), a)   # the parent's area holds the nests
    rec["source_licence"] = lic
    done("fetch", **fr)
    if fr.get("failed"):
        return finish(4)

    stage("wait-vram")
    t_lim = time.perf_counter() + a.wait_min * 60
    while not pf["ok"] and time.perf_counter() < t_lim:
        time.sleep(30)
        pf = _preflight(grids, cards, a.rrtm_rows)
    done("wait-vram", ok=pf["ok"], free_by_card=pf["free_by_card"])
    rec["preflight_at_forecast"] = pf
    if not pf["ok"]:
        rec["refused"] = f"waited {a.wait_min} min: {pf['free_by_card']} GiB free, need {est} GiB per card"
        return finish(3)
    env["NAMGPU_RRTM_ROWS"] = str(pf["pick"]["rrtm_rows"])

    # init: each domain's start saved once (its own process and card; side by side when there are several cards)
    shm = Path("/dev/shm")
    base = Path(os.environ.get("NAMGPU_INIT_DIR", str(shm if shm.is_dir() else out)))
    inits = {s["name"]: base / f"suite-{os.getpid()}-{s['name']}-{cycle:%Y%m%d%H}-init" for s in dom_specs}
    stage("init")
    procs = []
    for k, s in enumerate(dom_specs):
        # a nest's own source boundaries serve only its TDFI (boco.0000): 1 h asks for the 6 h minimum
        h = hours if s["parent"] is None else 1.0
        card = cards[k % len(cards)]
        cmd = [sys.executable, "-m", "namgpu.init.slab_init", "save", "--case", case, "--case-dir", str(case_dir),
               "--source", source, "--hours", str(h), "--out", str(inits[s["name"]])]
        e = dict(env, NAMGPU_CONFIGFILE=s["config"], CUDA_VISIBLE_DEVICES=card)
        e.pop("CUPY_GPU_MEMORY_LIMIT", None)
        log = open(out / f"init-{s['name']}.log", "w")
        procs.append((s["name"], card, subprocess.Popen(cmd, env=e, stdout=log, stderr=subprocess.STDOUT, cwd=str(cli.ROOT)),
                      log))
        if len(cards) == 1:                         # one card: one init at a time
            procs[-1][2].wait()
    codes = {}
    for nm, card, p, log in procs:
        codes[nm] = p.wait()
        log.close()
    done("init", dirs={k: str(v) for k, v in inits.items()}, rc=codes,
         cards={nm: card for nm, card, _p, _l in procs})
    if any(codes.values()):
        _rm(inits, rec)
        return finish(1)
    for s in dom_specs:
        s["init"] = str(inits[s["name"]])
    plan = out / "coupled-plan.json"
    plan.write_text(json.dumps({"domains": dom_specs}, indent=1))

    stage("forecast")
    if len(cards) == 1 and pf["pick"].get("pool_limit_gib"):
        env["CUPY_GPU_MEMORY_LIMIT"] = str(int(pf["pick"]["pool_limit_gib"] * 2**30))
        rec["pool_limit_gib"] = pf["pick"]["pool_limit_gib"]
    cmd = [sys.executable, "-m", "namgpu.nest.coupled", "--plan", str(plan), "--case", case, "--case-dir", str(case_dir),
           "--source", source, "--hours", str(hours), "--nest-hours", str(nest_hours), "--out", str(out),
           "--cards", str(len(cards))]
    if len(cards) > 1:
        cmd += ["--card-list", ",".join(cards), "--hist-workers", str(a.hist_workers)]
    code, _ = cli.run_logged(cmd, env, out / "run.log", sampler)
    if not a.keep_init:
        _rm(inits, rec)
    cr = out / "coupled-receipt.json"
    if cr.is_file():
        rec["forecast"] = json.loads(cr.read_text())
    done("forecast", rc=code)
    rec["forecast_rc"] = code
    if code:
        return finish(code)

    if not a.no_maps:
        stage("maps")
        for s in dom_specs:
            mc, _ = cli.run_logged([sys.executable, "-m", "namgpu.run_forecast", "--maps-only", "--case", case,
                                    "--case-dir", str(case_dir), "--out", str(out / s["name"])],
                                   dict(env, NAMGPU_CONFIGFILE=s["config"]), out / f"maps-{s['name']}.log", sampler)
            if mc:
                rec["failed_stage"] = f"maps {s['name']} rc {mc}"
        done("maps")
    return finish(5 if rec.get("failed_stage") else 0)


def _rm(inits: dict, rec: dict) -> None:
    for d in inits.values():
        shutil.rmtree(d, ignore_errors=True)
    rec["init_dirs_removed"] = [str(d) for d in inits.values()]
