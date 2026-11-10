"""Spin-up start (woof-nam run --start spinup): carry a spun-up state into the forecast, as the operational NAM did.

The operational NAM v4 started every cycle from the GDAS analysis 6 h before it and ran a 6 h catch-up with its
own hourly analysis before the free forecast, so its hour 0 already held hydrometeors, cold pools and mesoscale
circulations. We have no hourly analysis to offer; this keeps the catch-up. T0 is the cycle (the forecast's hour
0), S the spin-up length (--spinup-hours, default 6; a multiple of 6, because the start is a source cycle's
analysis, published every 6 h).

  1. Start from the source analysis at T0-S through the same isobaric start path as a cold start (init chain,
     static, TDFI), then integrate S hours with model physics on and the source's boundaries over [T0-S, T0],
     history off. The boundaries come from the same cycle the forecast takes its boundaries from (age 6 for GFS
     = the T0-6 cycle: leads 0..6 here, 6..90 for the forecast), so they run on without a seam at T0.
  2. Bring the state to the source analysis at T0, A(T0), without throwing the spun-up fields away
     (--spinup-update):
       iau    centred incremental analysis update, as GDAS does. At T0 the increment is A(T0) - X(T0) for PD, T,
              Q, U, V (X = the spun-up state; A through the start path's own horizontal and vertical
              interpolation, gfs_start.model_state, so the increment is analysis minus model on the same B grid
              and hybrid levels). The run then rewinds to T0-W/2 (a restart set written there on the way) and
              integrates again, adding increment/N after each of the N steps in (T0-W/2, T0+W/2]
              (W = --spinup-iau-hours, default S). The forecast's hour 0 is T0, so its 0 h history holds about
              half the increment, as GFS's IAU forecasts do; the rest goes in over the first W/2 hours.
              The rewind is a fresh process (os.execv into woof-nam run --resume <rewind set>): the tested resume
              path, with every unit cache rebuilt.
       nudge  analysis nudging over (T0-H, T0] (H = --spinup-nudge-hours, default 1): after every step
              X += (dt/tau) (A(T0) - X) for the same fields (tau = --spinup-nudge-tau, default 1 h), then free.
     Neither touches the hydrometeors (CW and the Ferrier fractions), W, TKE, soil, snow or the surface: they keep
     what S hours of model physics made. PINT moves with PD by its hydrostatic part (sg2 * dPD), so the
     nonhydrostatic pressure departure is kept; TP/UP/VP (the Adams-Bashforth history of T/U/V) move with T/U/V.
     Q is floored at EPSQ after each update.
  3. At T0 the run is re-based: clock start T0, step count 0, the boundaries switch to the forecast's own set (the
     very files and folder a cold start of that cycle uses), the run totals, the bucket accumulators and the
     hourly max fields are zeroed (a cold start's 0 h has none), and the 0 h history is written. Then the normal
     forecast.

Nothing here runs unless --start spinup is given: a cold start takes none of these paths.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

FIELDS = ("PD", "T", "Q", "U", "V")              # what the analysis update changes; never hydrometeors
PREV = {"T": "TP", "U": "UP", "V": "VP"}         # Adams-Bashforth history (dyn-adv1 sets them to T/U/V each step)
EPSQ = np.float32(1.0e-12)                       # NMMB EPSQ (module_CONSTANTS)
UPDATES = ("iau", "nudge")
# Run totals and averaging counters a cold start's 0 h history has at zero (beside drv-bucket-reset's tables).
_TOTALS = ("ACPREC_TOT", "CUPREC_TOT", "ACPREC", "CUPREC", "AVRAIN", "AVCNVC", "ACUTIM", "APHTIM")


# ------------------------------------------------------------------------------------------ timeline

def spinup_age(main_age_h: int, hours: int) -> int:
    """Boundary age of the spin-up window: the cycle the forecast takes its boundaries from (main_age_h older than
    T0) when that cycle is valid at T0-S, else the spin-up start cycle itself (age 0)."""
    return int(main_age_h) - int(hours) if int(main_age_h) >= int(hours) else 0


def window_steps(hours: float, dt: float) -> int:
    n = int(round(float(hours) * 3600.0 / float(dt)))
    if n <= 0:
        raise ValueError(f"spin-up window {hours} h is shorter than one step of {dt} s")
    return n


def iau_weight(k: int, n: int) -> float:
    """Weight of the increment added after the step that ends k steps after T0 (k <= 0: at or before T0): 1/n for the
    n steps of the centred window (-(n//2), n - n//2], else 0. Constant weights, as GDAS IAU without its filter."""
    lo = -(n // 2)
    return 1.0 / n if lo < k <= n + lo else 0.0


def nudge_coef(k: int, n: int, dt: float, tau_s: float) -> float:
    """Relaxation coefficient dt/tau for the n steps that end in (T0 - n dt, T0], else 0."""
    return float(dt) / float(tau_s) if -n < k <= 0 else 0.0


def step_k(start: datetime, ntimestep: int, t0: datetime, dt: float) -> int:
    """Steps from T0 to the end of the step just taken (post-advance count), negative before T0."""
    return int(round((start - t0).total_seconds() / float(dt))) + int(ntimestep)


@dataclass
class Plan:
    t0: str                  # the cycle, %Y-%m-%dT%H (forecast hour 0)
    hours: int               # spin-up length S
    update: str              # iau | nudge
    iau_hours: float         # W, the centred IAU window
    nudge_hours: float       # H, the nudging window ending at T0
    nudge_tau_s: float
    main_age_h: int          # the forecast's boundary age
    spin_age_h: int          # the spin-up window's boundary age
    source: str

    @property
    def cycle(self) -> datetime:
        return datetime.strptime(self.t0, "%Y-%m-%dT%H")

    @property
    def spin_start(self) -> datetime:
        return self.cycle - timedelta(hours=self.hours)

    def meta(self) -> dict:
        return asdict(self)

    @classmethod
    def from_meta(cls, d: dict) -> "Plan":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})


def make_plan(t0: datetime, hours: int, update: str, iau_hours, nudge_hours: float, nudge_tau_s: float,
              source: str, main_age_h: int) -> Plan:
    """Checks the knobs against what the run can do; each refusal names what would break."""
    hours = int(hours)
    if hours <= 0 or hours % 6:
        raise SystemExit(f"--spinup-hours {hours}: must be a positive multiple of 6 (the spin-up starts from a "
                         "source cycle's analysis, and GFS/IFS/ERA5 starts are 00/06/12/18Z)")
    if update not in UPDATES:
        raise SystemExit(f"--spinup-update {update!r}: one of {', '.join(UPDATES)}")
    iau_hours = float(hours if iau_hours is None else iau_hours)
    if not 0.0 < iau_hours / 2.0 <= hours:
        raise SystemExit(f"--spinup-iau-hours {iau_hours:g}: the centred window starts at T0-W/2, which must lie "
                         f"inside the {hours} h spin-up (0 < W <= {2 * hours})")
    if not 0.0 < float(nudge_hours) <= hours:
        raise SystemExit(f"--spinup-nudge-hours {nudge_hours:g}: the nudging window ends at T0 and must lie inside "
                         f"the {hours} h spin-up")
    if float(nudge_tau_s) <= 0:
        raise SystemExit("--spinup-nudge-tau must be > 0")
    return Plan(t0=f"{t0:%Y-%m-%dT%H}", hours=hours, update=update, iau_hours=iau_hours,
                nudge_hours=float(nudge_hours), nudge_tau_s=float(nudge_tau_s), main_age_h=int(main_age_h),
                spin_age_h=spinup_age(main_age_h, hours), source=source)


# ------------------------------------------------------------------------------------------ the update itself

def increment(an: dict, x: dict, xp=np) -> dict:
    """A - X for FIELDS, float32, on the arrays' own device."""
    return {n: (an[n] - x[n]).astype(xp.float32) for n in FIELDS}


def apply_increment(st: dict, inc: dict, w: float, sg2, xp=np) -> None:
    """In place: X += w * inc for T, Q, U, V (and their Adams-Bashforth history), PD += w dPD and PINT += sg2 w dPD
    (its hydrostatic part; sg2 [LM+1] is the grid's interface coefficient in PINT's order, index 0 = model top);
    Q floored at EPSQ. st: name -> array (ctx.st.fields)."""
    w = xp.float32(w)
    for n in ("T", "Q", "U", "V"):
        d = w * inc[n]
        st[n] += d
        p = PREV.get(n)
        if p is not None and p in st and st[p].shape == st[n].shape:
            st[p] += d
    dpd = w * inc["PD"]
    st["PD"] += dpd
    if "PINT" in st:
        st["PINT"] += xp.asarray(sg2, dtype=xp.float32).reshape(-1, 1, 1) * dpd[None]
    xp.maximum(st["Q"], EPSQ, out=st["Q"])


def apply_nudge(st: dict, an: dict, c: float, sg2, xp=np) -> None:
    """In place: X += c (A - X) for FIELDS, through apply_increment (same companions, same Q floor)."""
    apply_increment(st, {n: an[n] - st[n] for n in FIELDS}, c, sg2, xp)


def rms(inc: dict, xp=np) -> dict:
    return {n: round(float(xp.sqrt(xp.mean(inc[n].astype(xp.float64) ** 2))), 5) for n in FIELDS}


class Forcing:
    """The per-step hook (driver.Driver._run calls ctx.step_hook(ctx) after every step). Host arithmetic on the
    step count picks the weight; the update is a few element-wise launches on ctx.stream, no host sync."""

    def __init__(self, plan: Plan, dt: float, sg2, inc: dict | None = None, an: dict | None = None):
        import cupy as cp
        self.plan, self.dt = plan, float(dt)
        self.sg2 = cp.asarray(np.asarray(sg2, np.float32))
        self.inc, self.an = inc, an
        self.n_iau = window_steps(plan.iau_hours, dt)
        self.n_nudge = window_steps(plan.nudge_hours, dt)
        self.applied = 0
        self.last_k = None

    def __call__(self, ctx) -> None:
        import cupy as cp
        k = step_k(ctx.start, ctx.ntimestep, self.plan.cycle, self.dt)
        if self.inc is not None:
            w = iau_weight(k, self.n_iau)
            if w:
                apply_increment(ctx.st.fields, self.inc, w, self.sg2, cp)
                self.applied += 1
                self.last_k = k
        if self.an is not None:
            c = nudge_coef(k, self.n_nudge, self.dt, self.plan.nudge_tau_s)
            if c:
                apply_nudge(ctx.st.fields, self.an, c, self.sg2, cp)
                self.applied += 1
                self.last_k = k


# ------------------------------------------------------------------------------------------ inputs

def boundary_files(case_dir: Path, plan: Plan, row: dict) -> list:
    """[(hour, path)] the spin-up window's boundaries read: the spin_age_h cycle, every interval_h, bc_hours(S)."""
    from namgpu.domains import bc_hours
    b = row["boundary"]
    cyc = plan.spin_start - timedelta(hours=plan.spin_age_h)
    return [(fh, Path(case_dir) / b["file"].format(bcyc=cyc.hour, lead=fh + plan.spin_age_h))
            for fh in range(0, bc_hours(plan.hours) + 1, int(b["interval_h"]))]


def input_files(case_dir: Path, plan: Plan, row: dict) -> list:
    """Every file the spin-up reads: the analysis at T0-S and the window's boundaries, plus the analysis at T0 (a cold
    start's own start file, checked because the update reads it)."""
    out = [Path(case_dir) / row["file"].format(cyc=plan.spin_start.hour),
           Path(case_dir) / row["file"].format(cyc=plan.cycle.hour)]
    return out + [p for _, p in boundary_files(case_dir, plan, row)]


def ensure_inputs(case_dir: Path, plan: Plan, cfg=None, log=print) -> dict:
    """Fetch what is missing (fetch_case.fetch_spinup: the source's start row at T0-S and its boundary leads for the
    window, and the T0 analysis if absent). Returns a receipt row."""
    from namgpu.init import sources as srcmod
    row = srcmod.SOURCES[plan.source]
    t = time.perf_counter()
    missing = [p for p in input_files(case_dir, plan, row) if not p.is_file()]
    rec = {"missing_before": [str(p) for p in missing], "fetched": False}
    if missing:
        from namgpu.init import fetch_case
        log(f"spinup: fetching {len(missing)} missing inputs ({', '.join(p.name for p in missing[:4])}"
            f"{' ...' if len(missing) > 4 else ''})")
        rec["fetch_failed"] = fetch_case.fetch_spinup(plan.cycle, Path(case_dir), plan.hours, cfg=cfg,
                                                      source=plan.source, age_h=plan.main_age_h, log=log)
        rec["fetched"] = True
        still = [str(p) for p in input_files(case_dir, plan, row) if not p.is_file()]
        if still:
            raise FileNotFoundError(f"spinup: inputs still missing after the fetch: {still[:4]}")
    rec["seconds"] = round(time.perf_counter() - t, 2)
    return rec


# ------------------------------------------------------------------------------------------ run pieces

class InitScope:
    """The init chain at T0-S: ctx.start, the boundary folder tag and the boundary age of the spin-up window, set for
    the chain and put back after it."""

    def __init__(self, ctx, plan: Plan, forecast_hours: int):
        self.ctx, self.plan, self.fh = ctx, plan, forecast_hours

    def __enter__(self):
        c = self.ctx
        self.env = os.environ.get("NAMGPU_BOCO_AGE_H")
        os.environ["NAMGPU_BOCO_AGE_H"] = str(self.plan.spin_age_h)
        c.start = c.start_time = self.plan.spin_start
        c.boco_tag = "spinup"
        c.forecast_hours = self.plan.hours
        return self

    def __exit__(self, *exc):
        if self.env is None:
            os.environ.pop("NAMGPU_BOCO_AGE_H", None)
        else:
            os.environ["NAMGPU_BOCO_AGE_H"] = self.env
        self.ctx.boco_tag = None
        self.ctx.forecast_hours = self.fh
        return False


def _private_ctx(ctx, start):
    import copy
    from types import SimpleNamespace
    # a copy of the grid: gfs_start.ensure_vcoord sets the vertical coordinate on the grid it is given, and the
    # running model's grid keeps the one init-vcoord-recover gave it
    return SimpleNamespace(cfg=ctx.cfg, grid=copy.copy(ctx.grid), stream=ctx.stream, case_dir=ctx.case_dir,
                           start=start, start_time=start, init_source=getattr(ctx, "init_source", None),
                           tables={}, first_step=True, analysis=None, parent_run=None)


def forecast_bocos(ctx, plan: Plan, hours: float, log=print) -> tuple:
    """The forecast's boundary folder for T0 (gfs_bocos.run: built if absent or stale, reused when a cold start of this
    cycle already made it). Returns (folder, seconds)."""
    import cupy as cp
    from namgpu.init import gfs_bocos
    t = time.perf_counter()
    ns = _private_ctx(ctx, plan.cycle)
    ns.forecast_hours = max(6, int(hours + 0.999))
    with ctx.stream:
        out = gfs_bocos.run(ns)
    ctx.stream.synchronize()
    cp.get_default_memory_pool().free_all_blocks()
    s = round(time.perf_counter() - t, 2)
    log(f"spinup: forecast boundaries for {plan.cycle:%Y-%m-%d %HZ} in {out} ({s} s)")
    return Path(out), s


def analysis_on_grid(ctx, plan: Plan, log=print) -> tuple:
    """A(T0) on the model grid: the source's T0 start file through gfs_start.model_state (decode, horizontal
    interpolation onto the B grid, vertical interpolation onto the hybrid levels, wind rotation), the start path's
    own interpolation. Returns ({PD, T, Q, U, V} device copies, path, seconds)."""
    import cupy as cp
    from namgpu import state as statemod
    from namgpu.init import gfs_bocos, gfs_start, init_grid_geo, source_records as srec, sources as srcmod
    t = time.perf_counter()
    name, row = srcmod.source_row(ctx)
    rec = srec.table(row)
    ns = _private_ctx(ctx, plan.cycle)
    ns.st = statemod.State(ns.grid, names=gfs_bocos.BC_FIELDS)
    path = srcmod.source_path(ns, row)
    if not path.is_file():
        raise FileNotFoundError(f"spinup: the T0 analysis {path} is missing")
    tmpdir = str(Path(ctx.case_dir).resolve().parents[1] / "scratch")
    Path(tmpdir).mkdir(exist_ok=True)
    with ctx.stream:
        init_grid_geo.run(ns)
        static, _ = gfs_start.load_static(ns)
        ns.st.add("HGT_M", static["HGT_SFC"].shape)[...] = static["HGT_SFC"]
        gfs_start.model_state(ns, path, tmpdir, sfc_keys=rec["sfc_columns"], rec=rec)
        an = {n: cp.array(ns.st[n], copy=True) for n in FIELDS}
    ctx.stream.synchronize()
    del ns
    cp.get_default_memory_pool().free_all_blocks()
    s = round(time.perf_counter() - t, 2)
    log(f"spinup: analysis {path.name} on the model grid ({s} s)")
    return an, str(path), s


def zero_accumulators(ctx) -> list:
    """What a cold start's 0 h has at zero: run totals, the bucket accumulators, the hourly max fields."""
    from namgpu.io import io_max_fields
    from namgpu.steps import drv_bucket_reset
    st = ctx.st
    names = list(dict.fromkeys(_TOTALS + drv_bucket_reset.ARRAY_ORDER + drv_bucket_reset.POST_ARRAY_ORDER))
    done = []
    with ctx.stream:
        for n in names:
            a = st.fields.get(n)
            if a is not None:
                a.fill(0)
                done.append(n)
        if all(n in st.fields for n in io_max_fields.RESET_VALUES):
            io_max_fields._reset(st)
            done.append("max fields")
    ctx.tables["MAX_FIELDS_NCOUNT"] = 0
    return done


def rebase(ctx, plan: Plan, boco_dir, log=print) -> dict:
    """Make T0 the run's start: clock, step count, the forecast's boundary folder, zeroed accumulators, history back
    on with the 0 h history due."""
    ctx.start = ctx.start_time = plan.cycle
    ctx.ntimestep = 0
    ctx.boco_dir = str(boco_dir)
    ctx.history_off = False
    ctx.wrote_first_hist = False
    zeroed = zero_accumulators(ctx)
    log(f"spinup: re-based at {plan.cycle:%Y-%m-%d %HZ} (forecast hour 0); boundaries {boco_dir}; "
        f"{len(zeroed)} accumulators zeroed")
    return {"zeroed": zeroed}


def _save_increment(inc: dict, d: Path) -> dict:
    import cupy as cp
    d.mkdir(parents=True, exist_ok=True)
    for n, a in inc.items():
        np.save(d / f"{n}.npy", cp.asnumpy(a), allow_pickle=False)
    return {"dir": str(d), "gib": round(sum(a.nbytes for a in inc.values()) / 2**30, 2)}


def _load_increment(d: Path) -> dict:
    import cupy as cp
    return {n: cp.asarray(np.load(Path(d) / f"{n}.npy")) for n in FIELDS}


def _snapshot(ctx) -> dict:
    return {n: ctx.st.fields[n] for n in FIELDS}


def phase_a_iau(ctx, drv, plan: Plan, tdfi, out: Path, writer_meta: dict, log=print) -> dict:
    """IAU, first process: TDFI, integrate to T0-W/2, write the rewind set there (its files are written on a thread
    while the model steps on), integrate to T0, form the increment against A(T0) and save it. The caller then execs
    phase B (woof-nam run --resume <rewind set>)."""
    import cupy as cp
    from types import SimpleNamespace
    from namgpu import restart as rstmod
    from namgpu.steps import drv_time_loop
    rec = {"update": "iau"}
    writer, ctx.restart_writer = getattr(ctx, "restart_writer", None), None
    ctx.history_off = True
    t = time.perf_counter()
    rewind_h = plan.hours - plan.iau_hours / 2.0
    with ctx.stream:
        drv_time_loop._run(ctx, rewind_h, SimpleNamespace(run=tdfi), log=log)
    rec["start_to_rewind_s"] = round(time.perf_counter() - t, 2)
    meta = dict(writer_meta, spinup=dict(plan.meta(), phase="rewind", increment=str(out / "spinup" / "increment")))
    w = rstmod.Writer(out / "spinup" / "rewind", 0, float(ctx.cfg.dt), meta=meta, keep=1)
    ts = time.perf_counter()
    with ctx.stream:
        rec["rewind_entry"] = w.write(ctx)
    t2 = time.perf_counter()
    with ctx.stream:
        drv.run(float(plan.hours), log=log)
    rec["rewind_to_t0_s"] = round(time.perf_counter() - t2, 2)
    w.drain()
    rec["rewind_set"] = w.receipt()["latest"]
    rec["rewind_write_wall_s"] = round(time.perf_counter() - ts, 2)
    an, rec["analysis_file"], rec["analysis_s"] = analysis_on_grid(ctx, plan, log)
    with ctx.stream:
        inc = increment(an, _snapshot(ctx), cp)
    del an
    rec["increment_rms"] = rms(inc, cp)
    rec["increment"] = _save_increment(inc, out / "spinup" / "increment")
    del inc
    log(f"spinup: increment A - X at T0, rms {rec['increment_rms']}")
    rec["phase_a_s"] = round(time.perf_counter() - t, 2)
    ctx.restart_writer = writer
    (out / "spinup").mkdir(parents=True, exist_ok=True)
    (out / "spinup" / "phase-a.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


def exec_phase_b(a, rewind_set: str, case_dir: Path, out: Path) -> None:
    """Re-execute as woof-nam run --resume <rewind set> (same PID; this process's CUDA context goes with it)."""
    from namgpu.run_forecast import PKG_PARENT
    argv = ["--resume", str(rewind_set), "--case", str(case_dir), "--out", str(out)]
    if a.domain:
        argv += ["--domain", a.domain]
    for flag, on in (("--no-maps", a.no_maps), ("--no-columns", a.no_columns)):
        if on:
            argv.append(flag)
    if a.leave_out:
        argv += ["--leave-out", a.leave_out]
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv(sys.executable, [sys.executable, "-c",
                              "import sys; sys.path.insert(0, %r); from namgpu.run_forecast import main; "
                              "sys.exit(main(sys.argv[1:]))" % PKG_PARENT] + argv)


def run_nudge(ctx, drv, plan: Plan, tdfi, hours: float, boco_dir, log=print) -> dict:
    """Nudging, one process: TDFI, integrate to T0-H, A(T0) on the grid, nudge over (T0-H, T0], re-base, forecast."""
    import cupy as cp
    from types import SimpleNamespace
    from namgpu.steps import drv_time_loop
    rec = {"update": "nudge"}
    writer, ctx.restart_writer = getattr(ctx, "restart_writer", None), None
    ctx.history_off = True
    t = time.perf_counter()
    with ctx.stream:
        drv_time_loop._run(ctx, plan.hours - plan.nudge_hours, SimpleNamespace(run=tdfi), log=log)
    rec["start_to_window_s"] = round(time.perf_counter() - t, 2)
    an, rec["analysis_file"], rec["analysis_s"] = analysis_on_grid(ctx, plan, log)
    hook = Forcing(plan, ctx.cfg.dt, ctx.grid.sg2, an=an)
    with ctx.stream:
        d = increment(an, _snapshot(ctx), cp)
    rec["departure_rms_window_start"] = rms(d, cp)
    del d
    ctx.step_hook = hook
    t2 = time.perf_counter()
    with ctx.stream:
        drv.run(float(plan.hours), log=log)
    rec["window_s"] = round(time.perf_counter() - t2, 2)
    with ctx.stream:
        d = increment(an, _snapshot(ctx), cp)
    rec["departure_rms_t0"] = rms(d, cp)
    del d
    rec["nudge_steps"] = hook.applied
    ctx.step_hook = None
    del an, hook
    cp.get_default_memory_pool().free_all_blocks()
    rec["spinup_s"] = round(time.perf_counter() - t, 2)
    rec["rebase"] = rebase(ctx, plan, boco_dir, log)
    ctx.restart_writer = writer
    t3 = time.perf_counter()
    with ctx.stream:
        drv.run(hours, log=log)
    rec["forecast_s"] = round(time.perf_counter() - t3, 2)
    return rec


def run_resumed(ctx, drv, plan: Plan, meta: dict, hours: float, boco_dir, log=print) -> dict:
    """A resumed spin-up run: phase B of the IAU (from the rewind set: IAU to T0, re-base, forecast with the rest of
    the increment), or a forecast restart set of a spin-up run (re-attaches the IAU while its window is open)."""
    sp = meta.get("spinup") or {}
    rec = {"update": plan.update, "resumed_phase": sp.get("phase", "forecast")}
    hook = None
    if plan.update == "iau":
        k = step_k(ctx.start, ctx.ntimestep, plan.cycle, ctx.cfg.dt)
        n = window_steps(plan.iau_hours, ctx.cfg.dt)
        if k < n - n // 2:
            inc_dir = Path(sp.get("increment") or "")
            if not inc_dir.is_dir():
                raise FileNotFoundError(f"spinup: resumed inside the IAU window (step {k} from T0) but the "
                                        f"increment {inc_dir} is gone")
            hook = Forcing(plan, ctx.cfg.dt, ctx.grid.sg2, inc=_load_increment(inc_dir))
    ctx.step_hook = hook
    t = time.perf_counter()
    if ctx.start < plan.cycle:
        writer, ctx.restart_writer = getattr(ctx, "restart_writer", None), None
        ctx.history_off = True
        with ctx.stream:
            drv.run(float(plan.hours), log=log)
        rec["iau_to_t0_s"] = round(time.perf_counter() - t, 2)
        rec["rebase"] = rebase(ctx, plan, boco_dir, log)
        ctx.restart_writer = writer
    t3 = time.perf_counter()
    with ctx.stream:
        drv.run(hours, log=log)
    rec["forecast_s"] = round(time.perf_counter() - t3, 2)
    if hook is not None:
        rec["iau_steps_this_process"] = hook.applied
        rec["iau_last_k"] = hook.last_k
    ctx.step_hook = None
    return rec


def cleanup(out: Path) -> list:
    """The rewind set and the saved increment, once the run has finished (past the IAU window)."""
    import shutil
    gone = []
    for d in (out / "spinup" / "rewind", out / "spinup" / "increment"):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            gone.append(str(d))
    return gone
