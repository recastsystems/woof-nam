"""Run one WOOF-NAM forecast through the driver entry (Driver.run): hourly history, then maps
(io-maps, rw_wrfbatch).

    woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z
    (same as: python -m namgpu.run_forecast ...; python tools/run_forecast.py ... in a checkout)
Writes <out>/forecast-receipt.json (init s, per-hour wall, ms/step, nonfinite census at the end,
peak VRAM, units left out) plus the driver's timing_receipt.json, history files and maps/.

Restarts (namgpu/restart.py): --restart-every 1h writes a restart set every hour under <out>/restart; the
receipt's "restart_latest" is the newest complete set. `woof-nam run --resume <out>/restart` (or a set) continues
the run to its original end; case, source, hours and output folder come from the set.
History size (namgpu/io/history_select.py): --history-fields products|full|state|<list>, --full-every 3h.

Spin-up start (namgpu/spinup.py): --start spinup starts from the source analysis --spinup-hours (6) before the cycle,
runs the model that long with the source's boundaries, brings the state to the cycle's analysis without dropping the
spun-up fields (--spinup-update iau, centred incremental analysis update, or nudge, 1 h nudging) and forecasts from
the cycle, so hour 0 carries hydrometeors, cold pools and mesoscale circulations. The IAU run re-executes itself once
(woof-nam run --resume <out>/spinup/rewind) to rewind to the start of its window. The receipt's "spinup" names the
variant and its timings. The default (--start cold) is the unchanged cold start.

Our own box lanes source env-box.sh, which sets WOOF_NAM_BOX=1: only then does the run hand back the
card lock it inherited (CARD3.lock / NAMGPU_CARD_LOCK) before the maps. Without it no lock is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from namgpu import paths

PKG_PARENT = str(paths.PKG.parent)      # put back on sys.path for the maps-only tail


def release_card(ctx) -> dict:
    """Free this run's device memory. With WOOF_NAM_BOX=1 (our own box lanes) also unlock the card lock
    this process inherited (rl_run.sh and the lane runners hold it on an inherited fd; LOCK_UN on that fd
    releases it for the parent too). The lock file is NAMGPU_CARD_LOCK (default: any open file named
    CARD3.lock). Without WOOF_NAM_BOX no lock is touched. Returns what was done."""
    import fcntl
    import gc
    import os

    import cupy as cp
    ctx.stream.synchronize()
    ctx.st.fields.clear()
    ctx.tables.clear()
    gc.collect()
    from namgpu import devmem
    devmem.release()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    used = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    want = os.environ.get("NAMGPU_CARD_LOCK", "")
    dropped = []
    for fd in (os.listdir("/proc/self/fd") if paths.box_mode() else ()):
        try:
            tgt = os.readlink(f"/proc/self/fd/{fd}")
        except OSError:
            continue
        if (want and os.path.realpath(want) == tgt) or (not want and os.path.basename(tgt) == "CARD3.lock"):
            fcntl.flock(int(fd), fcntl.LOCK_UN)
            os.close(int(fd))
            dropped.append(tgt)
    print(f"card released: pool freed, card used {used} GiB, lock dropped {dropped or 'none held'}", flush=True)
    return {"card_used_gib_after": used, "locks_dropped": dropped}


def _domain_names():
    from namgpu import domains
    return domains.names()


def _stage(rec, name, seconds, parallelism):
    """One row of the receipt's stage table: wall seconds and how the stage runs (GPU, N cards, threads)."""
    rows = [r for r in rec.get("stages", []) if r["stage"] != name]      # a rerun tail replaces its own row
    rec["stages"] = rows + [{"stage": name, "s": None if seconds is None else round(seconds, 2),
                             "parallelism": parallelism}]


def _reserve_step_lmem(rec) -> None:
    """Step-loop local-memory reservation, set while the pool is empty, for a run that skips the init chain (a saved init
    or a restart). The init chain sets it at its end (init_case_driver); without this the reservation grew at a step-loop
    launch after the pool had filled (the OOM mode kernels.trim_local_memory exists to prevent; the VRAM audit saw
    9,216 B/thread at the peak, the fer_hires frame, not step_frame_bytes() 10,016). Fit path only, as in the chain."""
    from namgpu import kernels as _k
    from namgpu import state as statemod
    if not statemod.fit_on():
        return
    rec["local_memory"] = lm = _k.trim_local_memory("load")
    print(f"local-memory reservation set for the step loop: {lm['bytes_per_thread']} B/thread = "
          f"{lm['reserved_gib']} GiB ({lm['delta_gib']:+} GiB device used)", flush=True)


def memory_check(a) -> int:
    """Pre-flight of `woof-nam run`, the nest door's estimate (nest/domain.py vram_estimate) on the run's grid and
    rows: refuse in one line when the card cannot hold the run, before any allocation. 0.1.2 had none, so the 12 km
    grid on an 8 GB card died in State() with a raw OutOfMemoryError. The estimate is the fit memory path's (the
    smallest working set; a card of 64 GiB or more takes the full path, which holds more but fits there: 58-76 GB for
    12 km on a 96 GB card), so this refuses only a run that cannot fit on any path. When the card would take the full
    path by its size and cannot hold this grid on it, the run takes the fit path (below). No margin either: this door has
    no smaller setting to fall back to, and column bands shrink a tight run.
    Skipped when the nest door already checked (NAMGPU_PREFLIGHT_DONE) or NAMGPU_PREFLIGHT=0."""
    if os.environ.get("NAMGPU_PREFLIGHT_DONE") or os.environ.get("NAMGPU_PREFLIGHT") == "0":
        return 0
    from namgpu import config, state as statemod
    from namgpu.nest import domain
    from namgpu.physics.rrtm import rrtm_chunk
    forced = os.environ.get("NAMGPU_RRTM_ROWS", "").strip()
    if forced and int(forced) < rrtm_chunk.MIN_ROWS:
        print(f"woof-nam run: refused: NAMGPU_RRTM_ROWS {forced}: radiation runs in row chunks of at least "
              f"{rrtm_chunk.MIN_ROWS} rows; set 4 or more", file=sys.stderr)
        return 3
    cfg = config.load()
    g = {"im": int(cfg.im), "jm": int(cfg.jm)}
    if a.cards > 1:
        cards = [c.strip() for c in a.card_list.split(",") if c.strip()] or [str(i) for i in range(a.cards)]
        cards = cards[:a.cards]
    else:
        cards = [domain._visible_index()]
    # Radiation rows on the fit path: NAMGPU_RRTM_ROWS when set, else the domain row's "fit_rrtm_rows" when the card
    # has room for it, else FIT_RRTM_ROWS (8). Before this, `run --domain` always took 8 rows on the fit path, so the
    # 7.6 km grid (about 16.6 GiB peak at 32 rows, measured on a 5090) spent 46 % of its device time in 8-row
    # radiation chunks on cards with room for 32 (lane mesh-7p6km, 2026-11-08). The row is data, measured per grid.
    from namgpu import domains as _domains
    want_rows = int(_domains.current().get("fit_rrtm_rows") or 0)
    rows_try = ([int(forced)] if forced else
                sorted({statemod.FIT_RRTM_ROWS, max(want_rows, statemod.FIT_RRTM_ROWS)}, reverse=True))
    pick = rows_try[0]
    frees = {}
    for c in cards:
        try:
            free = domain.free_gib(c)
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return 0                      # no nvidia-smi reading: nothing to check against (the run says so if it dies)
        fits = None
        for rows in rows_try:
            e = domain.vram_estimate(g, min(rows, g["jm"]), len(cards), c, path="fit")
            if e["total_gib"] <= free:
                fits = rows
                break
        if fits is None:
            total = domain.card_total_gib(c)
            print(f"woof-nam run: refused: the {g['im']} x {g['jm']} grid needs about {e['total_gib']} GiB per card "
                  f"({len(cards)} card{'s' if len(cards) > 1 else ''}, NAMGPU_RRTM_ROWS {e['rrtm_rows']}, the "
                  f"smallest memory path) and card {c} ({domain.card_name(c)}, {total:.1f} GiB) has "
                  f"{free:.2f} GiB free; use more cards (--cards N) or a smaller domain (woof-nam nest)", file=sys.stderr)
            return 3
        pick = min(pick, fits)
        frees[c] = free
    # A card of 64 GiB or more takes the full memory path by its size alone. A grid whose full path the card cannot
    # hold now (domain.full_run_gib, measured) runs the fit path instead: the same history, bit for bit. Breakage it
    # prevents: the 7.6 km North America grid on a 96 GB card died with OutOfMemoryError at 99 GB in the start filter
    # (2026-11-09). A path the caller set is never changed.
    if (os.environ.get("NAMGPU_MEMORY_PATH", "").strip().lower() in ("", "auto")
            and os.environ.get("NAMGPU_SCRATCH_ARENA", "").strip() not in ("0", "1")):
        need = domain.full_run_gib(g, len(cards))
        for c in cards:
            total = domain.card_total_gib(c)
            if statemod.resolve_memory_path("", total)["path"] == "full" and need > frees[c]:
                why = (f"the full path needs about {need:.0f} GiB per card for the {g['im']} x {g['jm']} grid and "
                       f"card {c} ({total:.1f} GiB) has {frees[c]:.1f} GiB free")
                os.environ["NAMGPU_MEMORY_PATH"] = "fit"          # the forecast and its ranks read it
                os.environ["NAMGPU_MEMORY_PATH_WHY"] = why
                print(f"woof-nam run: fit memory path: {why}; the history is the same on either path", flush=True)
                break
    if not forced and pick > statemod.FIT_RRTM_ROWS:
        path = statemod.resolve_memory_path(os.environ.get("NAMGPU_MEMORY_PATH", ""), domain.card_total_gib(cards[0]))
        if path["path"] == "fit":
            os.environ["NAMGPU_RRTM_ROWS"] = str(pick)   # rrtm_call and the ranks read it
            print(f"woof-nam run: fit memory path, radiation in {pick}-row chunks (domain row fit_rrtm_rows, "
                  f"the card has room)", flush=True)
    return 0


def write_step_file(out: Path, cfg) -> Path:
    """<out>/nam_configfile_step: the run's physics step (dt and nphs as config.load() resolved them, overrides
    included) in configfile form, so `woof-nam products` (namgpu.io.products._run_ctx) can form step rates from PREC:
    the precipitation-type trace mask (PREC / (DT NPHS) >= 0.05 mm/h, namgpu.products.ptype). Breakage it prevents:
    a 12 km or 7.6 km parent run folder carried no configfile, so the awphys CSNOW/CICEP/CFRZR/CRAIN records had no
    physics step and were left out (ptype-winter lane, 2026-11-09). Not a forecast input; the run never reads it."""
    dst = Path(out) / "nam_configfile_step"
    dt = float(cfg.dt)
    whole = int(dt)
    num = int(round((dt - whole) * 1000.0))
    txt = ("# woof-nam run: this run's physics step, for woof-nam products (PRATE, the ptype trace rate)\n"
           f"dt_int: {whole}\ndt_num: {num}\ndt_den: 1000\nnphs: {int(cfg.nphs)}\n")
    if not dst.is_file() or dst.read_text() != txt:
        tmp = dst.with_name(dst.name + f".tmp{os.getpid()}")
        tmp.write_text(txt)
        tmp.replace(dst)
    return dst


def main(argv=None):
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(
        description="Run one WOOF-NAM forecast on the NAM 12 km CONUS grid (954 x 835 x 60, all physics) on "
                    "one NVIDIA card, or on several cards of one machine with --cards N. Writes hourly history "
                    "(wrfout_d01_<valid time>), maps in <out>/maps/ and <out>/forecast-receipt.json. Memory: one "
                    "card with 24 GB or more. On a card under 64 GB, or one that cannot hold the grid's full working set "
                    "(the 7.6 km North America grid on a 96 GB card), the run takes the fit path by itself (every "
                    "physics scheme stays on; NAMGPU_MEMORY_PATH=full or fit picks one); the log and the receipt "
                    "name the path that ran.",
        epilog="example (after woof-nam fetch --cycle 2025031412 --hours 24 --case-dir cases/2025-03-14_12z):\n"
               "  woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z\n"
               "On 2 cards (the same history, bit for bit; check with woof-nam compare):\n"
               "  woof-nam run --case 2025-03-14_12z --source native --hours 24 --out runs/2025-03-14_12z-2c --cards 2\n"
               "A GFS start (--source gfs) also needs the static fields once: woof-nam static --help.",
        formatter_class=HelpFormatter)
    ap.add_argument("--case", default=None,
                    help="the case folder woof-nam fetch wrote (cases/2025-03-14_12z), or its bare name, looked up "
                         "under <work root>/cases/; the name gives the start time (YYYY-MM-DD_HHz)")
    ap.add_argument("--source", default="native",
                    help="start from the archived NAM analysis (native, cycles before 2026-11-03 12Z), GFS (gfs), "
                         "ECMWF open data IFS (ifs from 2026-10-20 12Z, ifs-0p25 from 2025-03-14 12Z; start and "
                         "boundaries from the same IFS cycle, CC BY 4.0) or the ERA5 reanalysis (era5, from 1940 to "
                         "about 6 days ago; start and boundaries are ERA5 analyses; for hindcasts, CC BY 4.0) or NOAA's "
                         "RAP analysis (rap, from 2021-02-22: 13 km, radar-cycled, with its own clouds; GFS boundaries "
                         "and the GFS analysis outside RAP's grid and above 50 hPa) or NOAA's HRRR analysis (hrrr: 3 km CONUS, "
                         "radar-cycled, the same way: GFS outside CONUS and above 50 hPa, GFS boundaries) or NOAA's RRFS v1 "
                         "analysis (rrfs, from 2026-08-12: 3 km, the same way as hrrr) or a GEFS v12 member (gefs, with --member)")
    ap.add_argument("--soil-from", default=None, choices=("source", "gfs", "gfs-moisture"),
                    help="where the start's soil temperature and moisture come from: the start source's own (default) "
                         "or the same cycle's GFS analysis (rap and hrrr: their RUC soil moisture is read at Noah's depths "
                         "without rescaling between the two soil tables; gfs gives Noah-native soil instead; gfs-moisture keeps the source soil "
                         "temperature and takes GFS soil moisture and canopy water)")
    ap.add_argument("--member", default=None,
                    help="the source member, for a source with members (gefs: c00 control, p01-p30); default the "
                         "configfile key source_member, else the control. Sets NAMGPU_SOURCE_MEMBER for the run")
    ap.add_argument("--case-dir", type=Path, default=None,
                    help="the case folder (raw/, boco-<domain>/) when --case is only its name; the nest door passes a "
                         "cache on a data disk shared by every domain of that cycle")
    ap.add_argument("--parent-run", type=Path, default=None,
                    help="--source parent: the parent run directory (its hourly wrfout with the hand-off set)")
    ap.add_argument("--domain", default=None,
                    help="the grid, a row of parm/domains.json: %s (default: NAMGPU_CONFIGFILE if set, else conus12, "
                         "the NAM 12 km parent). conus4 and conus5 start from GFS and build their static surface from "
                         "WPS_GEOG (permissive soil set) on the first run of each month" % ", ".join(_domain_names()))
    ap.add_argument("--hours", type=float, default=None, help="forecast length in hours (default 1)")
    ap.add_argument("--physics-suite", default=None, metavar="NAME",
                    help="physics suite (parm/physics_suites.json): nam (default, NAM v4.2.11 physics), hrrrv5 or rapv5 "
                         "(TEMPO, MYNN-EDMF, MYNN surface layer, RUC, RRTMG, GSL drag on the NMMB dycore), or the "
                         "presets woof-nam-next-12km and woof-nam-next-7p6km. Every suite but nam uses TEMPO, which "
                         "needs gpuwm-hex 0.5.0 or newer (pip install \"recast-woof-nam[modern]\")")
    ap.add_argument("--out", type=Path, default=None, help="output folder (relative paths are under the work root)")
    ap.add_argument("--skip-init", default="init-fetch-case",
                    help="comma list of init units to skip (default: the fetch, already done by woof-nam fetch)")
    ap.add_argument("--leave-out", default="", help="comma list of plan units set to mode skip (receipt lists them)")
    ap.add_argument("--filter", choices=("auto", "on", "off"), default="auto",
                    help="the start's digital filter (TDFI). auto (default): off for --source native, whose NAM "
                         "analysis is already the NAM's filtered 0 h state; on for every other source. on / off force it")
    ap.add_argument("--no-maps", action="store_true", help="write history only, no maps")
    ap.add_argument("--pbl", choices=("myj", "ysu"), default=None,
                    help="boundary layer: myj (NAM: MYJ PBL + Janjic surface layer, default) or ysu (WRF YSU PBL + "
                         "revised MM5 surface layer); sets NAMGPU_TURBULENCE / NAMGPU_SFC_LAYER for every process")
    ap.add_argument("--topo-wind", type=int, choices=(0, 1), default=None,
                    help="with --pbl ysu: WRF topo_wind (0 off, WRF default; 1 Jimenez-Dudhia subgrid-topography "
                         "surface drag from the 30-arcsec terrain spread); sets NAMGPU_TOPO_WIND")
    ap.add_argument("--products", nargs="?", const="auto", default=None, metavar="SET",
                    help="also write NOAA's NAM GRIB2 files while the forecast runs (<out>/products/): the awphys "
                         "files on grid 218 for the 12 km parent, a nest's hiresf files on its NOAA grid (SET: "
                         "conusnest, alaskanest, hawaiinest, priconest, firewxnest, hiresf; default auto). A "
                         "`woof-nam products --follow` process writes each hour as its history file lands "
                         "(log <out>/products.log, <out>/products/DONE at the end)")
    ap.add_argument("--no-columns", action="store_true",
                    help="no sounding column store after the run (<out>/columns/d01/*.wcol; woof-nam columns)")
    ap.add_argument("--maps-only", action="store_true", help=argparse.SUPPRESS)   # internal: the post-run maps tail (no CUDA)
    ap.add_argument("--cards", type=int, default=1,
                    help="number of cards (default 1). N > 1 runs one process per card, each owning a band of whole "
                         "rows of the grid, with halo rows exchanged over NCCL; the history files are the same as "
                         "one card's, bit for bit")
    ap.add_argument("--card-list", default=os.environ.get("NAM_CARDS", ""),
                    help="comma list of card indices for --cards N (default 0..N-1; env NAM_CARDS)")
    ap.add_argument("--hist-workers", type=int, default=4, help="history writer processes for --cards N (default 4)")
    ap.add_argument("--init-from", type=Path, default=None,
                    help="a saved initial state (python -m namgpu.init.slab_init save ...) to load instead of running "
                         "the init; every card keeps its own rows. With --cards N and no --init-from, the run saves "
                         "one first, on the first card, in <out>/init (deleted at the end unless --keep-init)")
    ap.add_argument("--keep-init", action="store_true",
                    help="keep the saved initial state <out>/init (--cards N); on one card, save it there after the "
                         "init (a later run with --init-from <out>/init starts from the same bits)")
    ap.add_argument("--restart-every", default="0",
                    help="write a restart set every N hours of forecast (1h, 90m, 3; default 0 = none) under "
                         "--restart-dir; woof-nam run --resume continues from the newest one")
    ap.add_argument("--restart-dir", type=Path, default=None, help="restart root (default <out>/restart)")
    ap.add_argument("--restart-keep", type=int, default=1, help="restart sets kept (newest; default 1)")
    ap.add_argument("--resume", type=Path, default=None,
                    help="continue a run from a restart set, or from a restart root (its newest set), to the run's "
                         "original end (or --hours). Case, source and output folder come from the set")
    ap.add_argument("--restart-at", default="",
                    help="also write a restart set at these forecast hours (comma list, e.g. 3,6): the landing hours "
                         "of a moving nest (namgpu/nest/move.py), right after that hour's history")
    ap.add_argument("--move-from", type=Path, default=None,
                    help="moving nest: lay this restart set (the previous window, or a restart root: its newest set) "
                         "over the --init-from fresh start of this configfile's window, shifted by the windows' "
                         "offset, and step on from the set's step count (no TDFI). namgpu/nest/move.py")
    ap.add_argument("--follow-control", type=Path, default=None,
                    help="moving nest segment (woof-nam nest --follow): read the controller's control.json at every "
                         "landable time and stop there (restart set written) when it says stop. namgpu/nest/segment.py")
    ap.add_argument("--follow-wait-s", type=float, default=900.0,
                    help="--follow-control: the longest wait for a decision at a landable time before continuing")
    ap.add_argument("--move-check", action="store_true",
                    help="--move-from: read every carried array back from the card and compare its overlap with the "
                         "set bit for bit, and the two windows' static files on their shared cells (receipt "
                         "move.check)")
    ap.add_argument("--history-fields", default=None,
                    help="fields in each hourly history file: products (default: what the maps, side-by-side, the "
                         "column store and woof-nam products read), full (every field), state (3-D model state), "
                         "or a comma list of field names")
    ap.add_argument("--full-every", default=None,
                    help="every Nth output hour carries every field whatever --history-fields says (default 3h; "
                         "0 = never)")
    ap.add_argument("--managed-init", action="store_true",
                    help="--cards N: build the saved initial state in CUDA managed memory (a domain whose init does "
                         "not fit one card)")
    ap.add_argument("--gwd", choices=("nam", "ugwp", "off"), default=None,
                    help="orographic drag: nam (NAM's GWD, default), ugwp (NOAA-GSL UGWP drag suite as in RRFS: "
                         "large-scale GWD and blocking, small-scale GWD, turbulent form drag), off")
    ap.add_argument("--start", default="cold",
                    help="cold (default): the forecast starts from the --source analysis at the cycle. spinup: start "
                         "from the source analysis --spinup-hours before the cycle, run the model that long with the "
                         "source's boundaries, bring the state to the cycle's analysis (--spinup-update) and forecast "
                         "from the cycle: hour 0 then carries hydrometeors, cold pools and mesoscale circulations, as "
                         "the operational NAM's 6 h catch-up gave it. A source name (gfs, ifs, era5, ...) is a cold "
                         "start from that source")
    ap.add_argument("--spinup-hours", type=int, default=6,
                    help="--start spinup: hours of model before the cycle (default 6; a multiple of 6)")
    ap.add_argument("--spinup-update", default="iau", choices=("iau", "nudge"),
                    help="--start spinup: how the spun-up state is brought to the cycle's analysis. iau (default): "
                         "the increment analysis minus model at the cycle is added in equal parts over a window "
                         "centred on the cycle (the run rewinds to the window's start once). nudge: the state is "
                         "relaxed toward the analysis over the last hour before the cycle. Both change PD, T, Q, U, V "
                         "only, never the hydrometeors")
    ap.add_argument("--spinup-iau-hours", type=float, default=None,
                    help="--spinup-update iau: length of the centred window (default --spinup-hours: T0-3 h to T0+3 h)")
    ap.add_argument("--spinup-nudge-hours", type=float, default=1.0,
                    help="--spinup-update nudge: nudging window ending at the cycle, hours (default 1)")
    ap.add_argument("--spinup-nudge-tau", default="1h",
                    help="--spinup-update nudge: relaxation time (default 1h)")
    a = ap.parse_args(argv)
    if a.start not in ("cold", "spinup"):
        from namgpu.init import sources as _srcmod
        if a.start not in _srcmod.SOURCES:
            ap.error(f"--start {a.start!r}: cold, spinup, or a source ({', '.join(sorted(_srcmod.SOURCES))})")
        a.source, a.start = a.start, "cold"          # --start gfs: a cold start from GFS
    if a.start == "spinup" and not a.maps_only:
        from namgpu.init import sources as _srcmod
        _row = _srcmod.SOURCES.get(a.source, {})
        if _row.get("kind") != "isobaric" or "boundary" not in _row:
            ap.error(f"--start spinup --source {a.source}: the spin-up needs the source's analysis 6 h before the "
                     "cycle and its boundaries over that window, which only the isobaric sources with a boundary "
                     "row have (gfs, ifs, ifs-0p25, era5); the native row is one analysis file per case "
                     "(raw/nam/bgrd3d00.grib2) and a parent run starts at its own hour 0")
        if a.cards > 1:
            ap.error("--start spinup --cards N: the N-card run builds its initial state in a separate process that "
                     "every rank loads, and the spin-up's re-base and analysis update are not written for that "
                     "layout yet; run the spin-up on one card")
        if a.resume is not None or a.init_from is not None:
            ap.error("--start spinup with --resume/--init-from: a spin-up run resumes from its own restart sets "
                     "(woof-nam run --resume <set>, the spin-up is read from the set)")
    if a.gwd:
        os.environ["NAMGPU_GWD"] = a.gwd       # read by config.load(); rank children inherit it
    if a.pbl:
        os.environ["NAMGPU_TURBULENCE"] = a.pbl
        os.environ["NAMGPU_SFC_LAYER"] = {"myj": "myj", "ysu": "sfclayrev"}[a.pbl]
    if a.topo_wind is not None:
        os.environ["NAMGPU_TOPO_WIND"] = str(a.topo_wind)
    if getattr(a, "member", None):
        os.environ["NAMGPU_SOURCE_MEMBER"] = str(a.member)
    from namgpu import devmem as _devmem
    _devmem.blocking_sync()          # waits block instead of spinning a core (devmem.blocking_sync)
    resume_meta = None
    if a.resume is not None and not a.maps_only:
        from namgpu import restart as rstmod
        a.resume = rstmod.resolve(a.resume if a.resume.is_absolute() else paths.work_root() / a.resume)
        resume_meta = rstmod.read_meta(a.resume)
        a.case = a.case or (resume_meta["case_dir"] if Path(resume_meta["case_dir"]).is_dir() else resume_meta["case"])
        a.out = a.out or Path(resume_meta["out"])
        a.source = resume_meta["source"]
        a.hours = a.hours if a.hours is not None else float(resume_meta["hours"])
        hist = resume_meta.get("history") or {}
        a.history_fields = a.history_fields or hist.get("fields")
        a.full_every = a.full_every if a.full_every is not None else hist.get("full_every")
        if a.restart_every == "0" and resume_meta.get("restart_every"):
            a.restart_every = resume_meta["restart_every"]
        if a.restart_dir is None and resume_meta.get("restart_dir"):
            a.restart_dir = Path(resume_meta["restart_dir"])
        if a.init_from is not None:
            raise SystemExit("--resume and --init-from: a restart set already holds the whole state; give one")
    if a.move_from is not None and (a.init_from is None or a.resume is not None):
        raise SystemExit("--move-from needs --init-from <fresh start of the new window> (and no --resume)")
    if a.case is None or a.out is None:
        ap.error("--case and --out are required (or --resume <restart>)")
    a.hours = 1.0 if a.hours is None else a.hours
    if a.physics_suite:
        from namgpu.physics import suites as _suites
        os.environ[_suites.ENV] = a.physics_suite      # config.load, the ranks and the children read it
        _suites.selected()                            # an unknown name stops here, before the init
        # a preset row may name its grid (woof-nam-next-7p6km -> na7p6); an explicit --domain or NAMGPU_CONFIGFILE wins
        if a.domain is None and not os.environ.get("NAMGPU_CONFIGFILE") and _suites.row().get("domain"):
            a.domain = _suites.row()["domain"]
            print(f"woof-nam run: --physics-suite {a.physics_suite} runs on its own grid --domain {a.domain}", flush=True)
    if "NAMGPU_RANK" not in os.environ:
        # TEMPO or MYNN v6.1 without gpuwm-hex: one line here, before the fetch and the start, not a traceback at
        # the physics start (namgpu/hexdep.py says which options need it)
        from namgpu import hexdep
        hexdep.require(a.physics_suite, prog="woof-nam run")
    from namgpu import restart as _rst
    from namgpu.io import history_select as _hs
    a.history_fields = a.history_fields or _hs.DEFAULT_PRESET
    a.full_every = _hs.DEFAULT_FULL_EVERY if a.full_every is None else a.full_every
    try:
        a.restart_every_h = _rst.parse_hours(a.restart_every)
        a.full_every_h = _rst.parse_hours(a.full_every)
    except ValueError as e:
        ap.error(str(e))
    _hs.parse(a.history_fields)                          # an unknown field name stops here, before the init
    try:
        a.restart_at_h = [float(x) for x in str(a.restart_at).split(",") if x.strip()]
    except ValueError:
        ap.error(f"--restart-at {a.restart_at!r}: want hours, e.g. 3,6")
    if a.cards > 1 and (a.restart_every_h > 0 or a.resume is not None or a.restart_at_h or a.move_from is not None):
        raise SystemExit("--restart-every / --resume run on one card in this release: with --cards N every rank "
                         "would write its own rows and the resume would have to load them on the same N cards, "
                         "which is not written yet")
    if a.maps_only:
        return maps_only(a.case, a.out if a.out.is_absolute() else paths.work_root() / a.out, a.case_dir,
                         not a.no_columns)
    if a.products and "NAMGPU_RANK" not in os.environ:
        from namgpu import exporter
        try:
            exporter.ensure()
            if a.products not in ("auto", "awphys"):
                exporter.require(("pair_next", "mask_carrier"), f"the {a.products} nest files")
        except exporter.ExporterMissing as e:
            raise SystemExit(f"woof-nam run --products: {e}") from None
        _start_products(a.out if a.out.is_absolute() else paths.work_root() / a.out, a.products)
    from namgpu import domains
    domains.select(a.domain, a.source)
    if "NAMGPU_RANK" not in os.environ:
        start0 = domains.start_of(Path(a.case) if a.case_dir is not None else paths.case_dir(a.case))   # the name, in one line
        missing_case_folder(a, start0)
        rc = memory_check(a)
        if rc:
            return rc
    static_rec = None
    if "NAMGPU_RANK" not in os.environ:
        # once per run, before the init (and before any rank): in a child pinned to the first card
        cards0 = [c.strip() for c in a.card_list.split(",") if c.strip()]
        first = cards0[0] if cards0 else (None if a.cards == 1 else "0")
        cdir0 = a.case_dir.resolve() if a.case_dir is not None else paths.case_dir(a.case)
        static_rec = domains.ensure_static(cdir0, domains.start_of(Path(a.case) if a.case_dir is not None else cdir0),
                                           a.source, card=first)
        print(f"static: {static_rec['action']} {static_rec['file']} ({static_rec['seconds']} s)", flush=True)
    if a.cards > 1 and "NAMGPU_RANK" not in os.environ:
        if domains.current().get("managed_init") and not a.managed_init and a.init_from is None:
            a.managed_init = True       # the domain row: its initial state is bigger than one card
        return launch(a, list(argv) if argv is not None else sys.argv[1:], static_rec)
    if "NAMGPU_RANK" in os.environ:
        # kill -USR1 <rank pid> prints every thread stack to the rank log (no debugger needed)
        import faulthandler
        import signal
        faulthandler.register(signal.SIGUSR1, all_threads=True)

    import cupy as cp
    from namgpu import config, grid as gridmod, state as statemod
    from namgpu.driver import Ctx, Driver
    from namgpu.init import init_case_driver

    out = a.out if a.out.is_absolute() else paths.work_root() / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = config.load()
    config.start_filter(cfg, a.source, a.filter)          # TDFI off for a native NAM start unless --filter on
    write_step_file(out, cfg)
    g = gridmod.build(cfg)
    multi = rank_setup(g) if "NAMGPU_RANK" in os.environ else None   # several cards: this process is one rank
    if multi is not None and a.init_from is None:
        raise SystemExit("--cards N needs --init-from <saved init>: a rank holds only its own rows, and the init "
                         "chain builds the whole domain (whole-domain fits, the GWD terrain window, the boundaries), "
                         "so the init runs once (namgpu.init.slab_init save) and every rank loads its rows")
    if multi is not None:
        g = multi["grid"]                                     # the rank's grid (slab.local_grid)
    if a.init_from is not None or a.resume is not None:
        # the saved init replaces every field: allocate only the scalars here (no 2-D/3-D/4-D arrays twice)
        st = statemod.State(g, names={k for k, v in statemod.all_specs().items() if v.rank == 0})
    else:
        st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    if multi is not None:
        ctx.slab, ctx.comm, ctx.hist_geo_global = multi["slab"], multi["comm"], multi["geo"]
        ctx.decomp = (ctx.slab.rank, 1, ctx.slab.nranks)
        print(f"rank {ctx.slab.rank}/{ctx.slab.nranks}: rows {ctx.slab.jts}..{ctx.slab.jte} of {ctx.slab.jmg}, "
              f"memory {ctx.slab.jms}..{ctx.slab.jme} ({g.jm} rows), card {os.environ.get('CUDA_VISIBLE_DEVICES')}",
              flush=True)
    # --case-dir (nest door): the case cache on a data disk shared by every domain of that cycle; --case names it
    ctx.case_dir = a.case_dir.resolve() if a.case_dir is not None else paths.case_dir(a.case)
    ctx.start = datetime.strptime((Path(a.case).name if a.case_dir is not None else ctx.case_dir.name)[:14],
                                  "%Y-%m-%d_%Hz")
    ctx.start_time = ctx.start
    ctx.init_source = a.source
    if a.soil_from:
        from namgpu.init import gfs_start as _gs
        ctx.soil_from = a.soil_from
        os.environ[_gs.SOIL_FROM_ENV] = a.soil_from   # card processes
    ctx.parent_run = a.parent_run
    ctx.out_dir = out if multi is None else out / f"rank{ctx.slab.rank}"   # history goes via NAMGPU_HIST_SHM
    Path(ctx.out_dir).mkdir(parents=True, exist_ok=True)
    ctx.forecast_hours = max(6, int(a.hours + 0.999))
    drv = Driver(ctx)
    ctx.driver = drv
    ctx.dt = cfg.dt
    drv.check_ready()
    skip = [s for s in a.skip_init.split(",") if s]
    chain = [u for u in init_case_driver.CHAIN if u not in skip]
    rec = {"case": a.case, "source": a.source, "hours": a.hours, "dt": cfg.dt, "skipped_init": skip,
           "domain": domains.current()["name"], "grid": [int(cfg.im), int(cfg.jm), int(cfg.lm)],
           "cards": int(os.environ.get("NAMGPU_NRANKS", "1")), "woof_nam": _version()}
    if static_rec is not None:
        rec["static"] = static_rec
        _stage(rec, "static", static_rec["seconds"], "GPU, 1 card" if static_rec["action"] == "built" else "none (cached)")
    spin = None
    if a.start == "spinup" or (resume_meta or {}).get("spinup"):
        from namgpu import spinup
        from namgpu.init import sources as srcmod
        if resume_meta is not None:
            spin = spinup.Plan.from_meta(resume_meta["spinup"])
        else:
            spin = spinup.make_plan(ctx.start, a.spinup_hours, a.spinup_update, a.spinup_iau_hours,
                                    a.spinup_nudge_hours, _rst.parse_hours(a.spinup_nudge_tau) * 3600.0, a.source,
                                    srcmod.boco_age_h(srcmod.SOURCES[a.source], cfg))
        rec["start"] = "spinup"
        rec["spinup"] = {"plan": spin.meta()}
        pa = out / "spinup" / "phase-a-receipt.json"
        if resume_meta is not None and pa.is_file():
            rec["spinup"]["phase_a_receipt"] = json.loads(pa.read_text())
        if resume_meta is None:
            ts = time.perf_counter()
            rec["spinup"]["inputs"] = spinup.ensure_inputs(ctx.case_dir, spin, cfg)
            _stage(rec, "spinup inputs", time.perf_counter() - ts, "I/O, 16 threads")
        print(f"spinup: start {spin.spin_start:%Y-%m-%d %HZ}, {spin.hours} h to the cycle {spin.cycle:%Y-%m-%d %HZ}, "
              f"update {spin.update}", flush=True)
    mp = statemod.memory_path()
    rec["memory_path"] = mp
    # the stdout line names the path only; the card size that chose it stays in the receipt
    print(f"memory path: {mp['path']} ({mp['how']})", flush=True)
    t0 = time.perf_counter()
    if a.resume is not None:
        from namgpu import restart as rstmod
        from namgpu.init import slab_init
        rec["resumed_from"] = str(a.resume)
        rec["resumed_at_fhr"] = resume_meta["fhr"]
        _reserve_step_lmem(rec)
        rec["init"] = {"resumed_from": str(a.resume), "load": slab_init.load(ctx, a.resume)}
        rec["skipped_init"] = "all (resumed)"
        ctx.slab = ctx.comm = ctx.decomp = None
        ctx.case_dir = paths.case_dir(a.case)              # this machine's case folder, not the writer's
        ctx.out_dir = out
        rstmod.resume_boco(ctx, a.resume)
        from namgpu.steps import drv_loop_limits
        with ctx.stream:
            drv_loop_limits.run(ctx)
    elif a.init_from is not None:
        from namgpu.init import slab_init
        _reserve_step_lmem(rec)
        rec["init"] = {"loaded_from": str(a.init_from), "load": slab_init.load(ctx, a.init_from)}
        rec["skipped_init"] = "all (loaded)"
        if a.move_from is not None:
            # moving nest: the fresh start of this window is loaded; lay the previous window's restart set over it
            from namgpu import restart as rstmod
            from namgpu.nest import move as movemod
            mset = rstmod.resolve(a.move_from if a.move_from.is_absolute() else paths.work_root() / a.move_from)
            mmeta = rstmod.read_meta(mset)
            if not mmeta.get("window"):
                raise SystemExit(f"--move-from {mset}: the set records no window (written before namgpu/nest/move.py)")
            dj, di = movemod.shift_between(mmeta["window"], cfg)
            fresh_attrs = {k: getattr(ctx, k, None) for k in movemod.FRESH_ATTRS}
            rec["move"] = {"from": str(mset), "from_window": mmeta["window"], "window": movemod.window_meta(cfg),
                           "shift_dj_di": [dj, di]}
            mrec = movemod.overlay(ctx, mset, dj, di, out_dir=out, log=lambda m: print(m, flush=True))
            for k, v in fresh_attrs.items():
                if v is not None:
                    setattr(ctx, k, v)
            ctx.case_dir = a.case_dir.resolve() if a.case_dir is not None else paths.case_dir(a.case)
            rec["move"]["overlay"] = {k: v for k, v in mrec.items() if k not in ("carried",)}
            rec["resumed_at_fhr"] = round(int(ctx.ntimestep) * float(cfg.dt) / 3600.0, 4)
            ctx.move_landing = (int(ctx.ntimestep), dj, di, mset, mrec)
        # The saved ctx attributes are the init process's (one card: comm/slab/decomp None); this rank's win.
        if multi is not None:
            ctx.slab, ctx.comm, ctx.hist_geo_global = multi["slab"], multi["comm"], multi["geo"]
            ctx.decomp = (ctx.slab.rank, 1, ctx.slab.nranks)
        else:
            ctx.slab = ctx.comm = ctx.decomp = None
        # Per-rank tables are dropped by the load (slab_init.PER_RANK_TABLES); drv-loop-limits rebuilds this rank's
        # NMMB_LIMITS (consumers such as dyn-pgforce-divcor read it and do not build it).
        from namgpu.steps import drv_loop_limits
        with ctx.stream:
            drv_loop_limits.run(ctx)
    elif spin is not None:
        with ctx.stream, spinup.InitScope(ctx, spin, ctx.forecast_hours):
            rec["init"] = init_case_driver.run(ctx, only=chain)       # the start at T0-S, its own boundaries
    else:
        with ctx.stream:
            rec["init"] = init_case_driver.run(ctx, only=chain)
    ctx.stream.synchronize()
    if a.keep_init and a.init_from is None and a.resume is None and multi is None:
        # one card: the initial state, saved for later runs that must start from the same bits (--init-from)
        from namgpu.init import slab_init
        ts = time.perf_counter()
        rec["init_saved"] = {"path": str(out / "init"), **slab_init.save(ctx, out / "init")}
        rec["init_saved"]["save_s"] = round(time.perf_counter() - ts, 2)
        print(f"initial state saved to {out / 'init'} ({rec['init_saved']['gib']} GiB)", flush=True)
    if getattr(ctx, "move_landing", None) is not None and a.move_check:
        from namgpu.nest import move as movemod
        n_, dj_, di_, mset_, mrec_ = ctx.move_landing
        chk = movemod.carried_check(ctx, mset_, dj_, di_, mrec_["carried"])
        old_static = (mmeta.get("static_npz") or "")
        from namgpu import config as _cfgm
        new_static = _cfgm.static_npz(cfg)
        if old_static and new_static is not None and Path(old_static).is_file() and Path(new_static).is_file():
            chk["static"] = movemod.static_alignment(Path(old_static), Path(new_static), dj_, di_)
        rec["move"]["check"] = chk
        print(f"move check: {chk['arrays']} carried arrays, {chk['mismatched_arrays']} with mismatches; static "
              f"{ {k: v for k, v in (chk.get('static') or {}).items() if v['interior'] or v['band']} }", flush=True)
    rec["init_s"] = round(time.perf_counter() - t0, 2)
    # the start source row as staged (file, static, levels, attribution and licence: ECMWF open data is CC BY 4.0)
    rec["init_source"] = ctx.tables.get("grid_source")
    _stage(rec, "init" if a.init_from is None else "init load", rec["init_s"], "GPU, 1 card")
    print(f"init done {rec['init_s']} s, state {st.nbytes() / 2**30:.2f} GiB", flush=True)
    # Init temporaries (GWD terrain, table builds) stay cached in the CuPy pool and showed up as 40-55 GB on
    # card 3 for a forecast whose step loop needs ~17 GB. Return them.
    import gc
    gc.collect()
    from namgpu import devmem
    devmem.release(ctx.stream)
    cp.get_default_pinned_memory_pool().free_all_blocks()
    rec["cuda_wait"] = devmem.CUDA_WAIT          # blocking or spin (devmem.blocking_sync)
    rec["after_init_used_gib"] = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    rec["pool_after_init_gib"] = round(devmem.pool().total_bytes() / 2**30, 2)

    spin_boco = None
    if spin is not None:
        if a.resume is not None and Path((resume_meta or {}).get("case_dir", "")).is_dir():
            ctx.case_dir = Path(resume_meta["case_dir"])
        spin_boco, s_b = spinup.forecast_bocos(ctx, spin, a.hours)
        _stage(rec, "forecast boundaries", s_b, "GPU, 1 card; decode threads")
    leave = [u for u in a.leave_out.split(",") if u]
    for u in leave:
        drv.units[u]["mode"] = "skip"
    rec["left_out"] = leave

    from namgpu import restart as rstmod
    from namgpu.io import history_select
    ctx.history_select = history_select.Selection(a.history_fields, rstmod.steps_for(a.full_every_h, cfg.dt))
    rec["history"] = dict(ctx.history_select.receipt(), full_every=a.full_every)
    ctx.restart_writer = None
    if a.restart_every_h > 0 or a.restart_at_h or a.follow_control is not None:
        rroot = a.restart_dir or (out / "restart")
        rroot = rroot if rroot.is_absolute() else paths.work_root() / rroot
        ctx.restart_writer = rstmod.Writer(rroot, a.restart_every_h, cfg.dt, keep=a.restart_keep, meta={
            "case": str(a.case), "case_dir": str(ctx.case_dir), "source": a.source, "hours": a.hours,
            "out": str(out), "start": f"{(ctx.start if spin is None else spin.cycle):%Y-%m-%d_%H:%M:%S}",
            "dt": float(cfg.dt),
            "restart_every": a.restart_every, "restart_dir": str(rroot), "memory_path": mp["path"],
            "history": {"fields": a.history_fields, "full_every": a.full_every},
            "woof_nam": __import__("namgpu").__version__,
            "window": __import__("namgpu.nest.move", fromlist=["window_meta"]).window_meta(cfg),
            "configfile": str(config.configfile_path()), "static_npz": str(config.static_npz(cfg) or "")},
            at_hours=a.restart_at_h)
        if spin is not None:
            ctx.restart_writer.meta["spinup"] = dict(spin.meta(), phase="forecast",
                                                     increment=str(out / "spinup" / "increment"))
        print(f"restart sets every {a.restart_every_h:g} h" + (f" and at {a.restart_at_h} h" if a.restart_at_h else "")
              + f" -> {rroot}", flush=True)
    # moving nest shock numbers (namgpu/nest/move.py Monitor): after a landing, and before the segment's last restart
    ctx.move_monitors = []
    if getattr(ctx, "move_landing", None) is not None:
        from namgpu.nest import move as movemod
        n_, dj_, di_, _s, _r = ctx.move_landing
        per_min = 60.0 / float(cfg.dt)
        frames = [n_ + int(round(m * per_min)) for m in movemod.EXTRA_FRAMES_MIN]
        ctx.move_monitors.append(movemod.Monitor(ctx, n_ + 1, movemod.MONITOR_STEPS,
                                                 movemod.masks(int(g.jm), int(g.im), dj_, di_), "after move",
                                                 frames=frames))
    if a.restart_at_h:
        # before every landing hour (30 min), and after it when this run goes on through it (the same numbers with no
        # move, the reference for a moved run's first hour)
        from namgpu.nest import move as movemod
        pre = int(round(30 * 60.0 / float(cfg.dt)))
        n_stop = rstmod.steps_for(a.hours, cfg.dt)
        for h_ in a.restart_at_h:
            n_end = rstmod.steps_for(h_, cfg.dt)
            if n_end - pre > int(ctx.ntimestep):
                ctx.move_monitors.append(movemod.Monitor(ctx, n_end - pre + 1, pre, None, f"before {h_:g} h"))
            if n_end + movemod.MONITOR_STEPS <= n_stop:
                ctx.move_monitors.append(movemod.Monitor(ctx, n_end + 1, movemod.MONITOR_STEPS, None,
                                                         f"after {h_:g} h, no move"))

    ctx.follow_control = None
    if a.follow_control is not None:
        # a moving-nest segment: the controller's decision at every landable time, and the shock numbers of the
        # 30 min before each one (the reference for the first hour after a move), as --restart-at does
        from namgpu.nest import move as movemod, segment as segmod
        ctx.follow_control = segmod.FollowStop(a.follow_control, float(cfg.dt), wait_s=a.follow_wait_s)
        pre = int(round(30 * 60.0 / float(cfg.dt)))
        for n_end in ctx.follow_control.landing_steps(int(ctx.ntimestep), rstmod.steps_for(a.hours, cfg.dt)):
            if n_end - pre > int(ctx.ntimestep):
                ctx.move_monitors.append(movemod.Monitor(ctx, n_end - pre + 1, pre, None,
                                                         f"before {n_end * float(cfg.dt) / 3600.0:g} h"))
    from namgpu.steps.drv_physics_initialize import in_run_dir
    rd = in_run_dir(ctx)
    rd.__enter__()

    hours_log = []

    def log(msg):
        hours_log.append(msg)
        print(msg, flush=True)

    t1 = time.perf_counter()
    spin_exec = None
    try:
        # TDFI first, as the NAM forecast does (filter_method 3, exnam_fcst.sh), then the forecast loop: the same
        # order as drv-time-loop / tests/smoke_full.py. Driver.run alone skipped the filter (every run_forecast
        # run before 2026-11-04 19:00Z was a no-DFI start).
        from types import SimpleNamespace
        from namgpu.steps import drv_dfi_tdfi_driver, drv_time_loop

        def _tdfi(c):
            tf = time.perf_counter()
            out = drv_dfi_tdfi_driver.run(c)
            c.stream.synchronize()
            rec["tdfi_s"] = round(time.perf_counter() - tf, 2)
            print(f"tdfi done {rec['tdfi_s']} s {out}", flush=True)
            return out
        rec["filter_method"] = int(cfg.filter_method)
        with ctx.stream:
            if spin is not None and a.resume is not None:
                ctx.filter_active = False             # the filter ran in the first process
                rec["spinup"]["run"] = spinup.run_resumed(ctx, drv, spin, resume_meta, a.hours, spin_boco, log)
            elif spin is not None and spin.update == "iau":
                wmeta = {"case": str(a.case), "case_dir": str(ctx.case_dir), "source": a.source, "hours": a.hours,
                         "out": str(out), "start": f"{spin.cycle:%Y-%m-%d_%H:%M:%S}", "dt": float(cfg.dt),
                         "restart_every": a.restart_every,
                         "restart_dir": None if a.restart_dir is None else str(a.restart_dir),
                         "memory_path": mp["path"], "history": {"fields": a.history_fields, "full_every": a.full_every},
                         "woof_nam": __import__("namgpu").__version__}
                rec["spinup"]["phase_a"] = spinup.phase_a_iau(ctx, drv, spin, _tdfi, out, wmeta, log)
                spin_exec = rec["spinup"]["phase_a"]["rewind_set"]
            elif spin is not None:
                rec["spinup"]["run"] = spinup.run_nudge(ctx, drv, spin, _tdfi, a.hours, spin_boco, log)
            elif a.resume is not None or getattr(ctx, "move_landing", None) is not None:
                ctx.filter_active = False             # the filter ran before the first restart set / the move
                n = drv.run(a.hours, log=log)
            else:
                if int(cfg.filter_method) == 0:
                    # no filter: the 0 h reflectivity from the start state (CALMICT), as the filter's end writes it
                    from namgpu.steps import drv_dfi_refl_restore
                    drv_dfi_refl_restore.run(ctx)
                n = drv_time_loop._run(ctx, a.hours, SimpleNamespace(run=_tdfi), log=log)
        rec["error"] = None
    except Exception as e:  # noqa: BLE001
        n = ctx.ntimestep
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-3000:]
        print("RUN ERROR", rec["error"], flush=True)
    ctx.stream.synchronize()
    wall = time.perf_counter() - t1
    if spin_exec is not None and rec["error"] is None:
        rec["spinup_phase_a_wall_s"] = round(wall, 2)
        rec["exit_code"] = None
        (out / "spinup" / "phase-a-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
        rd.__exit__(None, None, None)
        print(f"spinup: phase A done in {wall:.1f} s; rewinding to {spin_exec}", flush=True)
        spinup.exec_phase_b(a, spin_exec, ctx.case_dir, out)    # does not return
    if getattr(ctx, "restart_writer", None) is not None:
        try:
            ctx.restart_writer.drain()
        except Exception as e:  # noqa: BLE001
            rec["error"] = rec["error"] or f"restart write: {type(e).__name__}: {e}"
        rec["restart"] = ctx.restart_writer.receipt()
        rec["restart_latest"] = rec["restart"]["latest"]     # the set a worker resumes from (woof-nam run --resume)
    if getattr(ctx, "follow_control", None) is not None:
        rec["follow_segment"] = ctx.follow_control.receipt()
    n0 = int(rec.get("resumed_at_fhr", 0) * 3600.0 / cfg.dt + 0.5) if "resumed_at_fhr" in rec else 0
    if getattr(ctx, "move_monitors", None):
        for m in ctx.move_monitors:
            m.drain()
        rec.setdefault("move", {})["monitors"] = [m.receipt() for m in ctx.move_monitors]
    rec["steps_done"] = int(ctx.ntimestep)
    rec["run_s"] = round(wall, 2)
    # per forecast step: the filter's 300 half-steps are timed apart (tdfi_s), not spread over the forecast steps
    rec["ms_per_step"] = round((wall - (rec.get("tdfi_s") or 0.0)) / max(1, ctx.ntimestep - n0) * 1e3, 1)
    if rec.get("tdfi_s") is not None:
        _stage(rec, "tdfi", rec["tdfi_s"], "GPU, 1 card")
    _stage(rec, "forecast", wall - (rec.get("tdfi_s") or 0.0), "GPU, 1 card; history on a writer thread")
    if spin is not None and isinstance(rec["spinup"].get("run"), dict) and "forecast_s" in rec["spinup"]["run"]:
        fs = rec["spinup"]["run"]["forecast_s"]
        nf = int(ctx.ntimestep) - (n0 if rec["spinup"]["run"].get("resumed_phase") == "forecast" else 0)
        rec["ms_per_step"] = round(fs / max(1, nf) * 1e3, 1)
        _stage(rec, "spinup", wall - fs - (rec.get("tdfi_s") or 0.0),
               "GPU, 1 card" + ("; resumed from the rewind set" if a.resume is not None else ""))
        _stage(rec, "forecast", fs, "GPU, 1 card; history on a writer thread")
        if rec["error"] is None:
            rec["spinup"]["cleaned"] = spinup.cleanup(out)
    rec["hour_lines"] = hours_log
    bad = init_case_driver.census(ctx)
    if multi is not None:
        # Several cards: the answer is the owned rows (history, maps, sums). Halo rows of fields NMMB never
        # exchanges (physics scratch, accumulators) are not kept consistent and may hold NaN; they are never read for
        # an owned value (the 1 vs N card history is bitwise). The pass/fail census is on the owned rows; the halo
        # count is kept beside it.
        rec["nonfinite_end_incl_halo"] = bad
        bad = owned_census(ctx)
    rec["nonfinite_end"] = bad
    SANE = ("T", "U", "V", "W", "Q", "CW", "Q2", "PD", "T2", "TSKIN", "ACPREC", "PREC", "CUPREC")
    sx = {}
    for nm in SANE:
        v = st.fields.get(nm)
        if v is not None and multi is not None and v.ndim >= 2 and v.shape[-2] == int(g.jm):
            v = v[..., ctx.slab.own0:ctx.slab.own0 + ctx.slab.nown, :]   # owned rows (halo rows of physics scratch hold NaN)
        if v is not None:
            sx[nm] = [round(float(v.min()), 5), round(float(v.max()), 5), round(float(v.mean()), 6)]
    rec["stats_end"] = sx
    rec["pool_end_gib"] = round(devmem.pool().total_bytes() / 2**30, 2)
    rec["pool"] = devmem.stats()            # allocator, live/reserved bytes and their high-water marks
    rec["card_used_gib"] = round((lambda m: (m[1] - m[0]) / 2**30)(cp.cuda.Device().mem_info), 1)
    print("end stats", json.dumps(sx), "nonfinite", bad, flush=True)
    rd.__exit__(None, None, None)
    if not a.no_maps and rec["error"] is None and multi is None:
        # Maps are host work. Before them: drop the card lock and give back ALL of the card, so the next
        # run starts while the frames render (run lead 11:05Z: the old post-run maps held CARD3.lock and
        # 62 GB for ~50 min on one core). Freeing the pool still left 10.7 GB in unit caches, so this process
        # re-executes itself as the maps-only tail (same PID for the runner, no CUDA context left).
        from namgpu.io.io_history import drain, shutdown as history_shutdown
        drain(ctx)                                   # the last history frame is on disk
        # The exec below replaces this process; without this the history writer processes, their forkserver and its
        # resource tracker stayed behind after every run with maps (1.3-1.8 GB of host memory each, still there an
        # hour after the run; release pass 2026-11-09).
        history_shutdown()
        rec["card_released"] = release_card(ctx)
        rec["exit_code"] = 1 if (rec["error"] or bad) else 0
        (out / "forecast-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
        sys.stdout.flush()
        sys.stderr.flush()
        os.execv(sys.executable, [sys.executable, "-c",
                                  "import sys; sys.path.insert(0, %r); from namgpu.run_forecast import main; "
                                  "sys.exit(main(sys.argv[1:]))" % PKG_PARENT,
                                  "--maps-only", "--case", a.case if a.case_dir is not None else str(ctx.case_dir),
                                  "--out", str(out)]
                 + (["--case-dir", str(ctx.case_dir)] if a.case_dir is not None else [])
                 + (["--no-columns"] if a.no_columns else []))
    rp = out / "forecast-receipt.json" if multi is None else out / f"forecast-receipt.r{ctx.slab.rank}.json"
    if multi is not None:
        rec["rank"] = {"rank": ctx.slab.rank, "nranks": ctx.slab.nranks, "jts": ctx.slab.jts, "jte": ctx.slab.jte,
                       "jms": ctx.slab.jms, "jme": ctx.slab.jme, "card": os.environ.get("CUDA_VISIBLE_DEVICES")}
        from namgpu.io.io_history import drain as _drain
        _drain(ctx)
    rp.write_text(json.dumps(rec, indent=1, default=str))
    print(f"receipt {rp}", flush=True)
    return 1 if (rec["error"] or bad) else 0


def _start_products(out: Path, set_name: str = "auto") -> None:
    """The GRIB2 follower (awphys, or a nest's hiresf set): its own process (no CUDA context of this run; the
    writer's GPU post is a small separate context), started once by the top process, alive across the maps-only
    re-exec, done when it has written every hour after the run's receipt appears."""
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "products.log", "ab")
    subprocess.Popen([sys.executable, "-c",
                      "import sys; sys.path.insert(0, %r); from namgpu.io.products import main; "
                      "sys.exit(main(sys.argv[1:]))" % PKG_PARENT, "--run", str(out), "--follow", "--set", set_name],
                     stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    print(f"products: GRIB2 follower ({set_name}) started -> {out / 'products'} (log {out / 'products.log'})",
          flush=True)


def owned_census(ctx) -> dict:
    """Non-finite count per float field over this rank's OWNED rows (axis -2 of [.., JM, X] fields). One host sync."""
    import cupy as cp
    s = ctx.slab
    jm = int(ctx.grid.jm)
    out, seen = {}, set()
    for name, a in ctx.st.fields.items():
        if not isinstance(a, cp.ndarray) or a.dtype.kind != "f" or a.ndim < 2 or a.shape[-2] != jm:
            continue
        if a.data.ptr in seen and a.ndim == 2:
            continue
        seen.add(a.data.ptr)
        n = int(cp.count_nonzero(~cp.isfinite(a[..., s.own0:s.own0 + s.nown, :])))
        if n:
            out[name] = n
    return out


def rank_setup(g) -> dict:
    """This process is rank NAMGPU_RANK of NAMGPU_NRANKS (set by launch()). Keeps the global history geometry, then
    gives the rank's grid (slab.local_grid: jm = the slab's memory rows, per-row arrays cut). Opens the NCCL
    communicator."""
    import numpy as np

    from namgpu.parallel import make_comm
    from namgpu.slab import Slab, local_grid
    r, n = int(os.environ["NAMGPU_RANK"]), int(os.environ["NAMGPU_NRANKS"])
    jmg, im = int(g.jm), int(g.im)
    cen = (jmg // 2, im // 2)
    geo = {"xlat": np.degrees(np.asarray(g.glat_h)).astype(np.float32),
           "xlon": np.degrees(np.asarray(g.glon_h)).astype(np.float32),
           "cen_lat": float(np.degrees(g.glat_h[cen])), "cen_lon": float(np.degrees(g.glon_h[cen])),
           "dx": float(np.asarray(g.dxh)[jmg // 2])}
    slab = Slab.for_rank(r, n, jmg)
    comm = make_comm(r, n, os.environ["NAMGPU_NCCL_ID"])
    return {"slab": slab, "comm": comm, "geo": geo, "grid": local_grid(g, slab)}


def _py_module(mod: str) -> list:
    """argv that runs `mod`'s main() with this package importable (a source checkout or an installed wheel)."""
    return [sys.executable, "-c",
            "import sys; sys.path.insert(0, %r); from %s import main; sys.exit(main(sys.argv[1:]))" % (PKG_PARENT, mod)]


def _drop_opt(argv: list, opt: str) -> list:
    return [x for x in argv if x != opt]


def _version():
    from namgpu import __version__
    return __version__


def launch(a, argv, static_rec=None) -> int:
    """--cards N: one process per card (CUDA_VISIBLE_DEVICES pinned; one Python thread driving N cards would be a
    single-core stage on the step path), plus the history writer pool. Every rank runs this same module with
    NAMGPU_RANK set. If one rank fails the others are stopped (NCCL would wait forever). Without --init-from the
    initial state is built once on the first card and saved to <out>/init first."""
    import shutil
    import signal
    import subprocess

    out = a.out if a.out.is_absolute() else paths.work_root() / a.out
    out.mkdir(parents=True, exist_ok=True)
    cards = [c.strip() for c in a.card_list.split(",") if c.strip()] or [str(i) for i in range(a.cards)]
    if len(cards) < a.cards:
        raise SystemExit(f"--cards {a.cards} but --card-list has {len(cards)} cards: {cards}")
    cards = cards[:a.cards]
    t0 = time.perf_counter()
    child_argv = _drop_opt(_drop_opt(list(argv), "--keep-init"), "--managed-init")
    from namgpu import domains
    rec = {"cards": cards, "ncards": len(cards), "case": a.case, "source": a.source, "hours": a.hours,
           "domain": domains.current()["name"], "woof_nam": _version()}
    if static_rec is not None:
        rec["static"] = static_rec
        _stage(rec, "static", static_rec["seconds"], "GPU, 1 card" if static_rec["action"] == "built" else "none (cached)")
    made_init = None
    if a.init_from is None:
        made_init = out / "init"
        cmd = _py_module("namgpu.init.slab_init") + [
            "save", "--case", str(a.case), "--source", a.source, "--hours", str(a.hours), "--out", str(made_init),
            "--skip-init", a.skip_init] + (["--managed"] if a.managed_init else [])
        print(f"saving the initial state once on card {cards[0]} -> {made_init} (log {out / 'init.log'})", flush=True)
        with open(out / "init.log", "w") as logf:
            rc = subprocess.call(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=cards[0]), stdout=logf,
                                 stderr=subprocess.STDOUT)
        rec["init_save_s"] = round(time.perf_counter() - t0, 1)
        _stage(rec, "init save", rec["init_save_s"], "GPU, 1 card" + (" (managed memory)" if a.managed_init else ""))
        if rc != 0:
            rec["exit_code"] = 1
            rec["error"] = f"initial state save failed (exit {rc}); see {out / 'init.log'}"
            (out / "forecast-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
            print(rec["error"], flush=True)
            return 1
        child_argv += ["--init-from", str(made_init)]
    idfile = out / "nccl.id"
    idfile.unlink(missing_ok=True)
    shm = Path("/dev/shm") / f"namgpu-hist-{os.getpid()}"
    shutil.rmtree(shm, ignore_errors=True)
    shm.mkdir(parents=True)
    writer = subprocess.Popen(_py_module("namgpu.io.history_merge") + [
        "--shm", str(shm), "--out", str(out), "--nranks", str(a.cards), "--workers", str(a.hist_workers)])
    if "--no-maps" not in child_argv:
        child_argv.append("--no-maps")
    procs = []
    t_run = time.perf_counter()
    for r, card in enumerate(cards):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=card, NAMGPU_RANK=str(r), NAMGPU_NRANKS=str(a.cards),
                   NAMGPU_NCCL_ID=str(idfile), NAMGPU_HIST_SHM=str(shm))
        logf = open(out / f"rank{r}.log", "w")
        procs.append((subprocess.Popen(_py_module("namgpu.run_forecast") + child_argv, env=env,
                                       stdout=logf, stderr=subprocess.STDOUT), logf))
    print(f"launched {a.cards} ranks on cards {cards}; logs {out}/rank*.log; history pool {a.hist_workers}",
          flush=True)
    codes = [None] * len(procs)
    failed = False
    while any(c is None for c in codes):
        for i, (p, _) in enumerate(procs):
            if codes[i] is None and p.poll() is not None:
                codes[i] = p.returncode
                if p.returncode != 0 and not failed:
                    failed = True
                    print(f"rank {i} exited {p.returncode}: stopping the other ranks", flush=True)
                    for q, _ in procs:
                        if q.poll() is None:
                            q.send_signal(signal.SIGTERM)
        time.sleep(0.5)
    for _, f in procs:
        f.close()
    (shm / "END").write_text("1")
    wcode = writer.wait()
    shutil.rmtree(shm, ignore_errors=True)
    rec.update({"rank_exit": codes, "history_pool_exit": wcode, "run_wall_s": round(time.perf_counter() - t_run, 1),
                "wall_s": round(time.perf_counter() - t0, 1),
                "ranks": [json.loads(p.read_text()) if p.exists() else None
                          for p in (out / f"forecast-receipt.r{r}.json" for r in range(a.cards))]})
    _stage(rec, "init load + tdfi + forecast", rec["run_wall_s"],
           f"GPU, {a.cards} cards (one process each); history on a {a.hist_workers}-process writer pool")
    # integer counters read at the end (nonfinite census per rank on the owned rows): one list, no step-path work
    rec["nonfinite_end_by_rank"] = [(r or {}).get("nonfinite_end") for r in rec["ranks"]]
    rec["ms_per_step_by_rank"] = [(r or {}).get("ms_per_step") for r in rec["ranks"]]
    rec["card_used_gib_by_rank"] = [(r or {}).get("card_used_gib") for r in rec["ranks"]]
    ok = all(c == 0 for c in codes) and wcode == 0 and not any(rec["nonfinite_end_by_rank"])
    rec["exit_code"] = 0 if ok else 1
    if made_init is not None and not a.keep_init:
        shutil.rmtree(made_init, ignore_errors=True)
        rec["init_deleted"] = str(made_init)
    (out / "forecast-receipt.json").write_text(json.dumps(rec, indent=1, default=str))
    print(f"ranks exit {codes}, history pool {wcode}, wall {rec['wall_s']} s -> {out / 'forecast-receipt.json'}",
          flush=True)
    if ok and not a.no_maps:
        return maps_only(str(a.case), out, a.case_dir, not a.no_columns)
    return 0 if ok else 1


def missing_case_folder(a, start) -> None:
    """A case folder that does not exist stops in one line naming the fetch, before any card or init work (it
    stopped deep in the init with a traceback). The nest and suite doors make their case folder before they call run;
    --resume and --source parent take theirs from the run they continue."""
    if a.resume is not None or a.source == "parent":
        return
    cdir = a.case_dir.resolve() if a.case_dir is not None else paths.case_dir(a.case)
    if cdir.is_dir():
        return
    src = "" if a.source in ("native", None) else f" --source {a.source}"
    raise SystemExit(f"woof-nam run: no case folder {cdir}; fetch it first: woof-nam fetch --cycle "
                     f"{start:%Y%m%d%H} --hours {a.hours:g} --case-dir {cdir}{src}")


def maps_only(case: str, out: Path, case_dir: Path | None = None, columns: bool = True) -> int:
    """Post-run maps in a process with no CUDA context: render every history frame of <out> not yet drawn
    (frames an hourly watcher rendered are skipped) across all cores, then finish the receipt."""
    from types import SimpleNamespace

    from namgpu.io import io_maps
    rp = out / "forecast-receipt.json"
    rec = json.loads(rp.read_text())
    cdir = Path(case_dir).resolve() if case_dir else paths.case_dir(case)
    ctx = SimpleNamespace(out_dir=out, case_dir=cdir,
                          start=datetime.strptime((Path(case).name if case_dir else cdir.name)[:14], "%Y-%m-%d_%Hz"))
    t2 = time.perf_counter()
    try:
        rec["maps"] = str(io_maps.run(ctx))[:2000]
    except (Exception, SystemExit) as e:  # noqa: BLE001
        # SystemExit too: maps.renderer() raises it when the renderer is missing, and before this the
        # receipt was never finished (no maps_error, no maps_s; review round 3, #3)
        rec["maps_error"] = f"{type(e).__name__}: {e}"
        print("MAPS ERROR", rec["maps_error"], flush=True)
    rec["maps_s"] = round(time.perf_counter() - t2, 2)
    _stage(rec, "maps", rec["maps_s"], "all cores (rw_wrfbatch)")
    if columns:
        from namgpu.io import columns as colmod
        try:
            res = colmod.write(colmod.frames_of([out]), out / "columns")
            rec["columns"] = {"files": len(res["files"]) - res["failed"], "failed": res["failed"],
                              "seconds": res["seconds"], "dir": str(out / "columns"),
                              "errors": [r.get("error") for r in res["files"] if r["rc"]][:3]}
            _stage(rec, "columns", res["seconds"], res["parallelism"])
            print(f"columns {rec['columns']['files']} files in {res['seconds']} s -> {out / 'columns'}", flush=True)
        except Exception as e:  # noqa: BLE001
            rec["columns_error"] = f"{type(e).__name__}: {e}"
            print("COLUMNS ERROR", rec["columns_error"], flush=True)
    print(f"maps {rec.get('maps', '')} in {rec['maps_s']} s", flush=True)
    rp.write_text(json.dumps(rec, indent=1, default=str))
    print(f"receipt {rp}", flush=True)
    return int(rec.get("exit_code", 0)) or (1 if ("maps_error" in rec or "columns_error" in rec
                                                  or (rec.get("columns") or {}).get("failed")) else 0)


if __name__ == "__main__":
    _rc = main()
    from namgpu.io import io_history as _ioh
    _ioh.shutdown()             # writer processes hold stdout; os._exit skips their exit hooks
    sys.stdout.flush()
    sys.stderr.flush()
    # hard exit: after a device OOM the CUDA/CuPy teardown hung for minutes holding the card and its lock
    # (seen on a 32 GB card, 2026-11-04)
    os._exit(int(_rc or 0))
