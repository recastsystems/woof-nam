"""One plan for every door: which source run and hour a window starts from, how long the model runs, and the line
that says so (woof-nam fetch / nest / suite --at TIME or --cycle C --lead H, on every source row).

The user says which source, when (a time, or a source run and forecast hour), and how long. resolve() turns that into
a plan and never refuses over an hour the source does not post:

  - a source that posts forecast hours a run can start from (gfs, gefs): the wanted hour when it is posted, else the
    nearest posted hour BEFORE it. The model starts there and runs that much longer; the output times stay what was
    asked (namgpu/outputs.py counts them from the wanted time). GFS hour 200 is not posted: start at 198, 2 h longer.
  - the boundary spacing is the row's own, raised to the coarsest posted step inside the window (GEFS past 240 h:
    6 h), and the start moves back again when that is what puts every boundary hour on a posted file (GFS hour 100
    for 24 h: 101 to 124 crosses 120, where GFS goes 3-hourly, so the start is 99, 1 h earlier).
  - a window that runs past the source's last posted hour is cut there, and the line says by how much.
  - a source with analyses only (native, era5, ifs, ifs-0p25, rap, hrrr, rrfs, aifs, aigfs): its newest analysis at or
    before the wanted time (row analysis_every_h: hourly for era5, rap, hrrr, rrfs; 6-hourly for the rest), and the
    model runs that much longer. rap, hrrr and rrfs take GFS outside their grid and on the edges: at a time that is
    not a GFS cycle that half is the GFS forecast hour of the same valid time (gfs_lead_h), from the same rule.
  - it stops only when nothing can run: a date outside the source's archive, a time no run of it reaches, or a
    window longer than the source's own forecast reaches when its edges are that forecast (ifs and ifs-0p25 144 h,
    aifs 360 h, aigfs 384 h: sources.reach_h, the last lead the fetch reads). The message names the sources that do
    reach it.

The plan is written to <case>/raw/START.json (sources.START_FILE) by the fetch; the run, the nest door and a resume
read it, so they cannot disagree with the fetch.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta

from namgpu.init import sources as S

MAX_BACK_H = 6            # never start more than this far before the wanted hour to dodge an unposted hour
FORECAST_STARTS = ("gfs", "gefs")      # rows a start can be a forecast hour of (the start record set is in every lead)
GFS_FILL = ("rap", "hrrr", "rrfs")     # regional analyses whose fill and edges are GFS (row "fallback")
# how long after its valid time an analysis row's file is on its archive (h): a wanted time newer than this steps back
LAG_H = {"native": 3, "gfs": 4, "gefs": 6, "ifs": 10, "ifs-0p25": 8, "aifs": 8, "aigfs": 6, "rap": 1, "hrrr": 1,
         "rrfs": 2, "era5": 6 * 24}


def _hours(td: timedelta) -> float:
    return td.total_seconds() / 3600.0


def _fmt(t: datetime) -> str:
    return f"{t:%Y-%m-%d %H}Z" if t.minute == 0 else f"{t:%Y-%m-%d %H:%M}Z"


def _g(h: float) -> str:
    return f"{h:g}" if h == int(h) else f"{h:.2f}".rstrip("0")


def parse_time(text: str, what: str = "--at") -> datetime:
    """YYYYMMDDHH or YYYYMMDDHHMM (UTC; dashes, T, Z and colons allowed)."""
    s = "".join(c for c in str(text) if c.isdigit())
    fmt = {10: "%Y%m%d%H", 12: "%Y%m%d%H%M"}.get(len(s))
    if fmt:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"{what} {text!r}: want a time YYYYMMDDHH or YYYYMMDDHHMM (UTC)")


def leads_table(name: str, row: dict, cycle: datetime | None = None):
    """[[last hour, step], ...] of the forecast hours a run of `name` can start from (gfs, gefs), or None."""
    if name == "gfs":
        return S.lead_steps(row, cycle)
    if name == "gefs":
        return row["posted_leads"]
    return None


def edge_steps(name: str, row: dict) -> list:
    """Boundary spacings to try, finest first: the row's own, then the coarser posted steps (GEFS 3 then 6)."""
    k = int(row.get("boco_interval_h") or (row.get("boundary") or {}).get("interval_h") or 3)
    out = [k]
    for _, s in (leads_table(name, row) or []):
        if int(s) > out[-1]:
            out.append(int(s))
    return out


def _chain(posted: set, lead: int, k: int) -> int:
    """How many hours of boundaries, every k from `lead`, are posted in a row (0 when lead itself is not)."""
    if lead not in posted:
        return 0
    n = 0
    while lead + n + k in posted:
        n += k
    return n


def fit(posted: list, want_h: float, hours: float, ks: list) -> dict:
    """The start hour for a window that should begin `want_h` hours into a source run and last `hours`: the nearest
    posted hour at or before want_h (at most MAX_BACK_H back) whose boundaries, every k of ks (finest first), are all
    posted for the whole window, which is then that much longer. When none carries the whole window, the nearest
    start that carries the most of it, cut at the last posted hour. ValueError when no posted hour is near."""
    from namgpu.domains import bc_hours
    pset = set(int(p) for p in posted)
    cands = sorted((p for p in pset if p <= want_h + 1e-9 and want_h - p < MAX_BACK_H), reverse=True)
    if not cands:
        raise ValueError(f"no posted hour within {MAX_BACK_H} h before hour {_g(want_h)}")
    for p in cands:
        off = want_h - p
        for k in ks:
            if bc_hours(hours + off, k) <= _chain(pset, p, k):
                return {"lead": p, "offset_h": off, "model_hours": hours + off, "edges_h": k, "cut_h": 0.0}
    # nothing carries all of it: the start that carries the most, cut at its last posted boundary
    # only a start that carries a run at all (6 h of boundaries: hour 381 carries 3 h, so a window there starts at 378)
    fits = [(p, k, _chain(pset, p, k) - (want_h - p)) for p in cands for k in ks if _chain(pset, p, k) >= 6]
    best = max(fits, key=lambda x: (x[2], x[0])) if fits else None
    p, k, keep = best or (None, None, 0)
    if best is None or keep <= 0:
        raise ValueError(f"the source posts no boundaries past hour {max(pset)}, so no run starts near hour "
                         f"{_g(want_h)}")
    return {"lead": p, "offset_h": want_h - p, "model_hours": float(_chain(pset, p, k)), "edges_h": k,
            "cut_h": hours - keep}


def _reach(now: datetime | None) -> str:
    """The sources that reach a time ahead of now, for a refusal."""
    return "gfs (to 384 h ahead) and gefs (to 384 h)"


def reach_error(name: str, start: datetime, model_hours: float, edges_h: int | None = None, offset_h: float = 0.0,
                want: datetime | None = None) -> str | None:
    """Why a run of `model_hours` from `name`'s analysis at `start` cannot run, when its edges are that source's own
    forecast and it ends past the last lead the fetch reads (sources.reach_h: ifs and ifs-0p25 144 h, aifs 360 h,
    aigfs 384 h); else None. Refused here, by name, instead of planned and then stopped in the fetch."""
    reach = S.reach_h(name)
    if reach is None:
        return None
    from namgpu.domains import bc_hours
    row = S.SOURCES[name]
    k = int(edges_h or (row.get("boundary") or {}).get("interval_h") or 3)
    need = bc_hours(model_hours, k)
    if need <= reach:
        return None
    back = f" ({_g(offset_h)} h of it before {_fmt(want)})" if offset_h > 1e-9 and want is not None else ""
    return (f"{name}'s own forecast reaches {reach} h from its {_fmt(start)} analysis and this run needs {need} h of "
            f"edges{back}: at most {_g(reach - offset_h)} h from {name} here; "
            + (f"{_reach(None)} reach further" if reach < 378 else "no source woof-nam starts from reaches further"))


def _archive(name: str, row: dict, t: datetime):
    """ValueError when time t is outside the row's archive (valid_from, valid_before, a rolling window)."""
    s = t.strftime("%Y-%m-%dT%H")
    if row.get("valid_from") and s < row["valid_from"]:
        alt = f"; {row['before_use']} has it" if row.get("before_use") else ""
        raise ValueError(f"{name} begins {row['valid_from'].replace('T', ' ')}Z, after {_fmt(t)}{alt}")
    if row.get("valid_before") and s >= row["valid_before"]:
        raise ValueError(f"{name} ends {row['valid_before'].replace('T', ' ')}Z (the NAM was retired), before "
                         f"{_fmt(t)}; gfs, ifs, era5, rap and hrrr have later times")


def resolve(name: str, *, hours: float, at: datetime | None = None, cycle: datetime | None = None,
            lead: float | None = None, now: datetime | None = None, posted=None) -> dict:
    """The plan of one window (see the module text). Give `at` (the wanted time) or `cycle` with `lead` (the wanted
    hour of that source run). posted(cycle, lead) -> bool asks the archive for a GFS hour (None: the posted-hour
    table is trusted). now: UTC, default the clock. ValueError when nothing can run, naming why."""
    if name not in S.SOURCES or name == "parent":
        raise ValueError(f"source {name!r}: a window is planned for a SOURCES row other than parent "
                         f"({', '.join(n for n in S.SOURCES if n != 'parent')})")
    row = S.SOURCES[name]
    now = now or datetime.utcnow()
    if at is None and cycle is None:
        raise ValueError("give a time (--at) or a source run and hour (--cycle with --lead)")
    if at is not None and cycle is not None:
        raise ValueError("give --at TIME, or --cycle with --lead, not both")
    want = at if at is not None else cycle + timedelta(hours=float(lead or 0))
    hours = float(hours)
    if hours <= 0:
        raise ValueError(f"a {hours:g} h window: the length must be more than 0")
    plan = {"source": name, "wanted": want, "hours": hours, "how": "--at" if at is not None else "--lead",
            "notes": []}
    if name in FORECAST_STARTS:
        plan.update(_forecast_start(name, row, want, hours, cycle, now, posted))
    else:
        plan.update(_analysis_start(name, row, want, hours, now, posted, given_cycle=cycle, given_lead=lead))
    plan["line"] = line(plan)
    return plan


def _forecast_start(name, row, want, hours, cycle, now, posted) -> dict:
    """gfs, gefs: a run and a forecast hour of it."""
    step = int(row.get("cycle_step_h", 6))
    if cycle is not None:
        if cycle.hour % step or cycle.minute:
            raise ValueError(f"--cycle {cycle:%Y%m%d%H}: a {name} run is at 00, 06, 12 or 18Z")
        cands = [cycle]
    else:
        if want - now > timedelta(hours=384 + step):
            raise ValueError(f"{_fmt(want)} is more than 384 h ahead; no {name} run reaches it")
        c = want.replace(minute=0, hour=want.hour - want.hour % step)
        cands = []
        while _hours(want - c) <= 384:
            if c <= now:
                cands.append(c)
            c -= timedelta(hours=step)
    why = []
    for c in cands:
        try:
            table = leads_table(name, row, c)
            _archive(name, row, c)
        except ValueError as e:
            why.append(f"the {_fmt(c)} run: {e}")
            continue
        try:
            f = fit(S.posted_leads(table), _hours(want - c), hours, edge_steps(name, row))
        except ValueError as e:
            why.append(f"the {_fmt(c)} run: {e}")
            continue
        if posted is not None and cycle is None:
            from namgpu.domains import bc_hours
            last = f["lead"] + bc_hours(f["model_hours"], f["edges_h"])
            if not (posted(c, f["lead"]) and posted(c, last)):
                why.append(f"the {_fmt(c)} run has not posted hours {f['lead']} to {last} yet")
                continue
        if cycle is not None:
            _given_run_posted(name, c, f, now, posted)
        start = c + timedelta(hours=f["lead"])
        out = {"start": start, "cycle": c, "lead_h": int(f["lead"]), "offset_h": f["offset_h"],
               "model_hours": f["model_hours"], "edges_h": int(f["edges_h"]), "cut_h": f["cut_h"]}
        if name == "gfs":
            out["gfs_cycle"], out["gfs_lead_h"] = c, int(f["lead"])
        return out
    raise ValueError(f"no {name} run can start a {_g(hours)} h window at {_fmt(want)}: "
                     + ("; ".join(why[:2]) if why else "no run at or before it is posted"))


POSTING_H = 24     # a run younger than this may still be posting its hours: a given --cycle is asked about them


def _given_run_posted(name, c, f, now, posted) -> None:
    """A run given with --cycle (and --lead) that has not happened, or has not posted the hours the window reads yet:
    ValueError in one line, before any fetch. Breakage it prevents: `woof-nam fetch --cycle 2026111900 --lead 6` the
    afternoon before printed its plan, then three bare "HTTP Error 404" lines naming no file (rc 1), and the nest door
    held a card for 42 s before the same end. Only a run younger than POSTING_H is asked (two HEAD requests); an
    older run's missing file is an archive gap, which the fetch names."""
    if c > now:
        raise ValueError(f"the {_fmt(c)} {name} run has not happened yet (now {_fmt(now)}); --at TIME starts from the "
                         "newest posted run that reaches a time")
    if posted is None or _hours(now - c) >= POSTING_H:
        return
    from namgpu.domains import bc_hours
    last = f["lead"] + bc_hours(f["model_hours"], f["edges_h"])
    try:
        ok = posted(c, f["lead"]) and posted(c, last)
    except OSError:
        return              # the archive did not answer: the fetch says what it could not get
    if not ok:
        raise ValueError(f"the {_fmt(c)} {name} run has not posted hours {f['lead']} to {last} yet; --at TIME starts "
                         "from the newest posted run that reaches a time")


def _analysis_start(name, row, want, hours, now, posted, given_cycle=None, given_lead=None) -> dict:
    """A row with analyses only: the newest one at or before the wanted time, and the run that much longer."""
    every = int(row.get("analysis_every_h", 6))
    if want > now:
        raise ValueError(f"{name} has analyses only, and {_fmt(want)} is ahead of now; {_reach(now)} reach it")
    s = want.replace(minute=0, hour=want.hour - want.hour % every)
    lag = LAG_H.get(name, 6)
    while _hours(now - s) < lag:
        s -= timedelta(hours=every)
        if _hours(want - s) > max(every, MAX_BACK_H):
            raise ValueError(f"{name}'s newest analysis is about {lag} h behind the clock, and {_fmt(want)} is newer; "
                             f"{_reach(now)} reach it")
    _archive(name, row, s)
    out = {"start": s, "cycle": s, "lead_h": 0, "offset_h": _hours(want - s), "model_hours": hours + _hours(want - s),
           "edges_h": int((row.get("boundary") or {}).get("interval_h") or 3), "cut_h": 0.0}
    why = reach_error(name, s, out["model_hours"], out["edges_h"], out["offset_h"], want)
    if why:
        raise ValueError(why)
    notes = []
    if given_cycle is not None and given_lead:
        notes.append(f"{name} forecast hours are not a start in this release: the {name} analysis of "
                     f"{_fmt(s)} is used for {name} hour {_g(float(given_lead))} of the {_fmt(given_cycle)} run")
    if name in GFS_FILL and s.hour % 6:
        # the GFS half (outside the regional grid, above its top, the edges) at the same valid time
        g = S.SOURCES["gfs"]
        gp = _forecast_start("gfs", g, s, out["model_hours"], None, now, posted)
        if gp["offset_h"]:
            raise ValueError(f"no GFS forecast hour is valid at {_fmt(s)} to go around the {name} analysis")
        out.update(gfs_cycle=gp["cycle"], gfs_lead_h=gp["lead_h"], edges_h=gp["edges_h"])
    out["notes_extra"] = notes
    return out


def line(p: dict) -> str:
    """The one line every door prints for a plan."""
    from namgpu.domains import bc_hours
    name, want, start, off = p["source"], p["wanted"], p["start"], p["offset_h"]
    lead = int(p.get("lead_h") or 0)
    if lead:
        what = (f"{name.upper()} forecast hour {lead} of the {_fmt(p['cycle'])} run (a forecast, not an analysis)")
    else:
        what = f"the {name} analysis of {_fmt(start)}"
    txt = f"start {_fmt(start)} from {what}"
    if off > 1e-9:
        asked = (f"hour {_g(_hours(want - p['cycle']))}" if lead or p.get("how") == "--lead" else _fmt(want))
        why = "is not posted" if lead or name in FORECAST_STARTS else "has no analysis"
        txt += f"; {asked} {why}: {_g(off)} h earlier, the model runs {_g(off)} h longer, first output {_fmt(want)}"
    txt += f"; {_g(p['model_hours'])} h; edges every {p['edges_h']} h"
    if lead:
        txt += f" (hours {lead} to {lead + bc_hours(p['model_hours'], p['edges_h'])} of that run)"
    if p.get("gfs_lead_h") and name != "gfs":
        txt += f" from the GFS {_fmt(p['gfs_cycle'])} run hour {p['gfs_lead_h']}"
    if p.get("cut_h", 0) > 1e-9:
        txt += f"; cut {_g(p['cut_h'])} h short: the source posts nothing later"
    for n in p.get("notes_extra") or ():
        txt += f"; {n}"
    return txt


def record(case_dir, p: dict) -> dict:
    """Write the plan to <case>/raw/START.json (what the run, the nest door and a resume read)."""
    import json
    from pathlib import Path
    rec = {"start": f"{p['start']:%Y-%m-%dT%H}", "source": p["source"], "chosen_by": p.get("how", "--at"),
           "wanted": f"{p['wanted']:%Y-%m-%dT%H:%M}", "offset_h": round(float(p["offset_h"]), 6),
           "hours": p["hours"], "model_hours": round(float(p["model_hours"]), 6), "cut_h": p.get("cut_h", 0.0),
           "edges_h": int(p["edges_h"]), "line": p["line"],
           "cycle": f"{p['cycle']:%Y-%m-%dT%H}", "lead_h": int(p.get("lead_h") or 0)}
    if p.get("gfs_lead_h"):
        rec.update(gfs_cycle=f"{p['gfs_cycle']:%Y-%m-%dT%H}", gfs_lead_h=int(p["gfs_lead_h"]))
    if rec["lead_h"]:
        rec["what"] = (S.lead_credit(rec["lead_h"], p["start"]) if p["source"] == "gfs" else
                       f"{p['source'].upper()} forecast hour {rec['lead_h']} of the {p['cycle']:%Y-%m-%d %H}Z run")
    else:
        rec["what"] = f"the {p['source']} analysis of {p['start']:%Y-%m-%d %H}Z"
    path = Path(case_dir) / S.START_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def for_run(case_dir, hours: float) -> dict:
    """What a run of `hours` (counted from the wanted time) does with the case's plan: {"offset_h", "model_hours",
    "cut_h", "line"}; offset 0 and the hours as given when the case has no plan or a plan without an offset."""
    rec = S.read_start(case_dir) if case_dir is not None else None
    off = float((rec or {}).get("offset_h") or 0.0)
    model = float(hours) + off
    cut = 0.0
    cap = (rec or {}).get("model_hours")
    if cap is not None and (rec or {}).get("cut_h") and model > float(cap) + 1e-9:
        cut, model = model - float(cap), float(cap)
    return {"offset_h": off, "model_hours": model, "cut_h": cut, "plan": rec}


def env_for(rec: dict | None) -> dict:
    """The settings a door puts in the environment for a case's plan, so its card processes read the same files:
    the GFS hour (NAMGPU_START_LEAD_H) for gfs and the regional rows' GFS half, and the run age and boundary spacing
    (NAMGPU_BOCO_AGE_H, NAMGPU_BOCO_INTERVAL_H) for a gefs forecast-hour start."""
    if not rec:
        return {}
    out = {}
    if rec.get("gfs_lead_h"):
        out[S.LEAD_ENV] = str(int(rec["gfs_lead_h"]))
    if rec.get("source") == "gefs" and rec.get("lead_h") is not None:
        out["NAMGPU_BOCO_AGE_H"] = str(int(rec["lead_h"]))
        out[S.INTERVAL_ENV] = str(int(rec.get("edges_h") or 3))
    return out


def math_ceil(x: float) -> int:
    return int(math.ceil(float(x) - 1e-9))
