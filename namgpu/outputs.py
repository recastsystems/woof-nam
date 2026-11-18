"""Which times a run writes: `woof-nam run --write SPEC [--maps-at SPEC] [--grib-at SPEC]`.

A spec is one of, or a comma list of:
  an interval       1h, 30m, 15m, 3h          every that long, from the first output time to the run's end
  a time            0, 6, 12, 18.5            hours after the first output time (a bare number is a time, not an interval)
  a piece           0-24/1h, 24-186/6h        from hour A to hour B every STEP
so `1h` is every hour, `0,1.5,3,6` four times, `0-6/15m,6-24/1h` quarter-hourly for six hours and hourly after.

Times count from the first output time: the start the user asked for. A run whose source did not post that hour starts
earlier (sources.resolve_window: GFS hour 200 is not posted, the run starts at 198) and `offset_h` is that difference,
so the output times stay what was asked. Each time is the model step nearest it (25 s at 12 km: any multiple of
5 minutes is exact on every grid; another time gets the nearest step, and the file name carries that step's real time).

The schedule is a set of step counts, as restart sets are (restart.Writer: "every" plus "at"). driver._due("history")
asks it. The configfile key minutes_history stays 60 and stays what the convection driver reads, so the forecast is
the same whatever is written. With no spec the schedule is every 60 minutes from step 0: the files a run always wrote.

Carriers: history times nobody asked for that a product reads.
  rain   the rain bucket starts of each GRIB2 time (products._buckets: the 3 h and 12 h marks and the 3 h snow marks
         of the parent, a nest set's own buckets), and the frames 1 h and 6 h before each map time (the 1 h and 6 h
         rain maps difference against them). Written with RAINNC, RAINC and ACSNOW only (LEAN_FIELDS).
  state  the times a nest reads from this run for its edges (--carry SPEC, set by the suite door). Written with the
         run's own field selection.
A carrier is never drawn and never exported.

Light frames: an output time that is not a whole number of hours of lead from the run's start (offset_h after
the model start), and that no GRIB2 file and no nest reads, is only drawn, so with the default
--history-fields it carries MAP_FIELDS (what woof-nam's maps and the site's own maps read: the 2-D map fields and the
3-D T, QVAPOR, PRES, PINT, U and V the MSLP and 850 hPa wind maps read) instead of the whole products set. Breakage it
prevents: every 5-minute frame was a whole 3.3 GB products history (1.3 GB after zlib, 33 s on one core), the four
writers could not keep up, and the forecast stopped to wait for them (5-8 s a frame). Whole
lead hours, GRIB2 times and state carriers keep the run's own selection. A run that names --history-fields keeps it
at every time. Breakage the lead count prevents: a run started at 18:30 (model start 18:00, offset 0.5 h) wrote
every hourly frame at :30, off the model's hour, so all were light: no soundings, and a reader of a whole-state
field failed on them (2026-11-14).

Catalog frames (NAMGPU_CATALOG_EVERY=15m, or any interval: read from the environment, so a worker that sets it on
an older woof-nam changes nothing). A light time a whole number of that interval of lead from the run's start is a
catalog frame instead: it carries the run's own field selection (what a whole hour carries: everything the WOOF map
catalog's grid-218 frame and the column store read), written WITHOUT compression, and it is listed with
"catalog": true and "light": false, so readers treat it as a whole frame for maps and soundings, never for GRIB2.
Breakage it prevents: a whole products history at every 15-minute time costs 22 s of zlib on one core (1.33 GB from
3.6 GB), so at 15-minute output the writers took 88 s of every 26-27 s model hour and the map catalog could only be
hourly; the same fields written raw take a second or two, and the values (so every map and sounding drawn from them)
are byte for byte those of a compressed history at that time. Whole hours stay compressed: they are the run's kept
history.

<out>/OUTPUTS.json lists every history time with what it is for; the maps, the GRIB2 follower, the column store and a
resume read that one list.
"""
from __future__ import annotations

import json
import math
import re
from datetime import datetime, timedelta
from pathlib import Path

FILE = "OUTPUTS.json"
DEFAULT = "1h"
MAP_WINDOWS_H = (1, 6)            # io/maps.py WINDOW_H: the 1 h and 6 h rain maps
PARENT_BUCKETS_H = (12, 3)        # io/products.py BUCKETS_H (and its 3 h snow marks inside the 12 h window)
LEAN_FIELDS = ("RAINNC", "RAINC", "ACSNOW")
# Light frames (module docstring): namgpu/io/maps.py NEEDS for every map product, the MSLET inputs (T, QVAPOR, PRES,
# PINT, HGT: namgpu/products/memslp.py) and the snow fields the site's maps read (SNOWNC, SR).
MAP_FIELDS = ("PSFC", "HGT", "T2", "Q2", "U10", "V10", "MSLP", "RAINNC", "RAINC", "ACSNOW", "SNOWNC", "SR",
              "COMPOSITE", "T", "QVAPOR", "PRES", "PINT", "U", "V", "MSLET")
_NUM = r"\d+(?:\.\d+)?"
_STEP = re.compile(rf"({_NUM})\s*(h|m|min)")
_PIECE = re.compile(rf"({_NUM})\s*-\s*({_NUM})\s*/\s*({_NUM})\s*(h|m|min)")
EXAMPLES = "1h, 15m, 0,6,12,18.5 or 0-24/1h,24-186/6h"
CATALOG_ENV = "NAMGPU_CATALOG_EVERY"


def catalog_every_h(env=None):
    """Hours between catalog frames (module docstring, NAMGPU_CATALOG_EVERY: an interval such as 15m), or None."""
    import os
    text = str((os.environ if env is None else env).get(CATALOG_ENV) or "").strip().lower()
    if not text or text in ("0", "off", "none"):
        return None
    m = _STEP.fullmatch(text)
    if not m or float(m.group(1)) <= 0:
        raise ValueError(f"{CATALOG_ENV}={text!r}: want an interval such as 15m or 1h")
    return _hours(m.group(1), m.group(2))


def _hours(value: str, unit: str) -> float:
    v = float(value)
    return v / 60.0 if unit in ("m", "min") else v


def parse(spec) -> list:
    """A spec -> pieces: ("every", step_h), ("at", hour) or ("range", from_h, to_h, step_h). ValueError in plain
    words for anything else."""
    text = str(spec if spec is not None else DEFAULT).strip().lower()
    if not text:
        raise ValueError(f"an empty output spec (examples: {EXAMPLES})")
    out = []
    for tok in (t.strip() for t in text.split(",")):
        m = _PIECE.fullmatch(tok)
        if m:
            a, b, step = float(m.group(1)), float(m.group(2)), _hours(m.group(3), m.group(4))
            if b < a or step <= 0:
                raise ValueError(f"output piece {tok!r}: want FROM-TO/STEP with TO at or after FROM, e.g. 0-24/1h")
            out.append(("range", a, b, step))
            continue
        m = _STEP.fullmatch(tok)
        if m:
            step = _hours(m.group(1), m.group(2))
            if step <= 0:
                raise ValueError(f"output interval {tok!r}: must be longer than zero")
            out.append(("every", step))
            continue
        if re.fullmatch(_NUM, tok):
            out.append(("at", float(tok)))
            continue
        raise ValueError(f"output spec {tok!r}: want an interval (1h, 15m), hours after the start (0,6,12) or a "
                         f"piece (0-24/1h); examples: {EXAMPLES}")
    return out


def hours_of(spec, length_h: float) -> list:
    """Every output time of a spec inside a run of length_h hours: hours after the first output time, sorted."""
    eps = 1e-9
    got = set()
    for p in parse(spec):
        if p[0] == "at":
            ts = [p[1]]
        else:
            a, b, step = (0.0, float(length_h), p[1]) if p[0] == "every" else (p[1], min(p[2], float(length_h)), p[3])
            n = int(math.floor((b - a) / step + eps)) if b >= a else -1
            ts = [a + k * step for k in range(n + 1)]
        got.update(round(t, 6) for t in ts if -eps <= t <= float(length_h) + eps)
    return sorted(got)


def step_of(hours: float, dt: float) -> int:
    return int(round(float(hours) * 3600.0 / float(dt)))


def _floor_marks(lead_s: float, buckets_h) -> list:
    """The history times (seconds from the model start, above 0) the rain buckets ending at lead_s start from: every
    multiple of the smallest bucket from the largest bucket's start to the smallest bucket's start (products._buckets
    and _snow_bucket_12 for the parent's 12 h / 3 h set; one mark for a nest set's single bucket)."""
    if lead_s <= 0 or not buckets_h:
        return []
    small, large = min(buckets_h) * 3600, max(buckets_h) * 3600
    lo = int((lead_s - 1) // large) * large
    hi = int((lead_s - 1) // small) * small
    return [m for m in range(lo, hi + 1, small) if m > 0]


class Schedule:
    """The history steps of one run and what each is for.

    dt            the model step, s
    hours         the model's length, h (from the model start)
    offset_h      hours from the model start to the first output time (0 unless the run starts earlier than asked)
    write         the spec of the history times (None: every hour)
    maps, grib    specs of the map and GRIB2 times when they differ from `write` (None: the write times)
    carry         spec of state carriers (the times a nest reads this run at), counted from the model start
    buckets_h     the rain buckets of this run's GRIB2 set (parent 12 and 3 h; a nest set's own)
    products      False for a run that writes no GRIB2 (no --products): no GRIB2 times and none of their carriers
    grib_minutes  True when the GRIB2 writer can name a time off the hour; else GRIB2 keeps the whole model hours.
                  The pinned writer (rw_rrfs_upp 09c2dff1) cannot: given a 13:30 frame (XTIME 90 min) it wrote
                  hiresf01, the 13:00 name and hour (checked 2026-11-13). Breakage this prevents: a
                  half-hour field filed and labelled as the whole hour before it
    light_fields  the fields of a light frame (MAP_FIELDS), or None: every time keeps the run's own selection
    catalog_h     hours of lead between catalog frames (module docstring), or None: no catalog frames
    """

    def __init__(self, dt: float, hours: float, offset_h: float = 0.0, write=None, maps=None, grib=None, carry=None,
                 buckets_h=PARENT_BUCKETS_H, map_windows_h=MAP_WINDOWS_H, grib_minutes: bool = False,
                 products: bool = True, light_fields=None, catalog_h=None):
        self.dt, self.hours, self.offset_h = float(dt), float(hours), float(offset_h)
        self.light_fields = tuple(light_fields) if light_fields else None
        self.catalog_h = float(catalog_h) if catalog_h else None
        self.spec = {"write": write, "maps": maps, "grib": grib, "carry": carry}
        self.buckets_h, self.map_windows_h = tuple(buckets_h or ()), tuple(map_windows_h or ())
        self.notes = []
        if self.offset_h < 0:
            raise ValueError(f"output offset {offset_h} h: the first output cannot be before the model start")
        length = self.hours - self.offset_h
        if length < 0:
            raise ValueError(f"the run is {self.hours:g} h long and its first output is {self.offset_h:g} h in")
        end = step_of(self.hours, self.dt)

        def steps(spec, what, base_h):
            got = {}
            for h in hours_of(spec, self.hours - base_h):
                n = step_of(base_h + h, self.dt)
                if n <= end:
                    got.setdefault(n, h)
            given = [p[1] for p in parse(spec) if p[0] == "at" and p[1] > self.hours - base_h + 1e-9]
            if given:
                self.notes.append(f"{what} at hour {', '.join(f'{g:g}' for g in given)}: past the run's end "
                                  f"({self.hours - base_h:g} h), left out")
            return got

        w = steps(write, "output", self.offset_h)
        m = w if maps is None else steps(maps, "maps", self.offset_h)
        g = {} if not products else (w if grib is None else steps(grib, "GRIB2", self.offset_h))
        per_h = 3600.0 / self.dt
        if not grib_minutes:
            off = sorted(n for n in g if abs(n / per_h - round(n / per_h)) > 1e-9)
            if off and grib is not None:
                self.notes.append(f"GRIB2 is written on whole forecast hours: the "
                                  f"{len(off)} time{'s' if len(off) > 1 else ''} asked off the hour "
                                  f"({', '.join(self._tag(n) for n in off[:4])}{', ...' if len(off) > 4 else ''}) "
                                  "are history and maps only")
            g = {n: h for n, h in g.items() if n not in set(off)}
        c = steps(carry, "carried times", 0.0) if carry is not None else {}
        rain = set()
        for n in g:
            rain.update(step_of(s / 3600.0, self.dt) for s in _floor_marks(round(n * self.dt), self.buckets_h))
        for n in m:
            for back in self.map_windows_h:
                k = n - step_of(back, self.dt)
                if k >= 0:
                    rain.add(k)
        asked = set(w) | set(m) | set(g)
        whole_start = abs(self.offset_h - round(self.offset_h)) <= 1e-9

        def on_lead_hour(n):
            # a whole start: the model's whole hours are the lead's (the dev5 rule, unchanged); else the step a whole
            # lead hour from the run's start lands on
            if whole_start:
                return abs(n / per_h - round(n / per_h)) <= 1e-9
            k = round((n * self.dt / 3600.0) - self.offset_h)
            return k >= 0 and n == step_of(self.offset_h + k, self.dt)

        def on_catalog(n):
            # a whole number of catalog intervals of lead from the run's start (the first output time)
            if self.catalog_h is None:
                return False
            k = round((n * self.dt / 3600.0 - self.offset_h) / self.catalog_h)
            return k >= 0 and n == step_of(self.offset_h + k * self.catalog_h, self.dt)

        self.entries = {}
        for n in sorted(asked | set(c) | rain):
            what = [k for k, d in (("history", w), ("maps", m), ("grib", g)) if n in d]
            if n in asked:
                carrier = None
            elif n in c:
                carrier = "state"
            else:
                carrier = "rain"
            self.entries[n] = {"step": n, "model_h": round(n * self.dt / 3600.0, 6),
                               "h": round(n * self.dt / 3600.0 - self.offset_h, 6),
                               "history": n in w, "maps": n in m, "grib": n in g, "carrier": carrier,
                               "light": bool(self.light_fields) and carrier is None and n not in g and n not in c
                               and not on_lead_hour(n),
                               "for": what or ([f"a nest's edges"] if carrier == "state" else
                                               ["rain since this time (a GRIB2 bucket start or a rain map)"])}
            e = self.entries[n]
            e["catalog"] = bool(e["light"]) and on_catalog(n)
            if e["catalog"]:
                e["light"] = False
        self._steps = frozenset(self.entries)
        self._lean = frozenset(n for n, e in self.entries.items() if e["carrier"] == "rain")
        self._light = frozenset(n for n, e in self.entries.items() if e["light"])
        self._catalog = frozenset(n for n, e in self.entries.items() if e["catalog"])
        self._sorted = sorted(self._steps)

    def _tag(self, n: int) -> str:
        mins = int(round(n * self.dt / 60.0))
        return f"f{mins // 60:03d}" if mins % 60 == 0 else f"f{mins // 60:03d}m{mins % 60:02d}"

    # -- what the driver asks --------------------------------------------------------------------------------------
    def due(self, ntimestep: int) -> bool:
        return int(ntimestep) in self._steps

    def lean(self, ntimestep: int) -> bool:
        """This step's file is a rain carrier: only LEAN_FIELDS are written."""
        return int(ntimestep) in self._lean

    def light(self, ntimestep: int) -> bool:
        """This step's file is a light frame: only light_fields are written (module docstring)."""
        return int(ntimestep) in self._light

    def catalog(self, ntimestep: int) -> bool:
        """This step's file is a catalog frame: the run's own selection, written uncompressed (module docstring)."""
        return int(ntimestep) in self._catalog

    def frames_per_hour(self) -> float:
        """History files a model hour, on average over the run (the writers' pool sizes from it)."""
        return len(self.entries) / max(1.0, self.hours)

    def next_due(self, ntimestep: int):
        """The first scheduled step after ntimestep, or None."""
        import bisect
        k = bisect.bisect_right(self._sorted, int(ntimestep))
        return self._sorted[k] if k < len(self._sorted) else None

    # -- what the readers ask --------------------------------------------------------------------------------------
    def counts(self) -> dict:
        e = list(self.entries.values())
        return {"history": sum(1 for x in e if x["carrier"] is None), "maps": sum(1 for x in e if x["maps"]),
                "grib": sum(1 for x in e if x["grib"]), "carriers": sum(1 for x in e if x["carrier"]),
                "files": len(e), **({"catalog": k} if (k := sum(1 for x in e if x.get("catalog"))) else {})}

    def line(self) -> str:
        c = self.counts()
        first = min((x["model_h"] for x in self.entries.values() if x["carrier"] is None), default=None)
        said = {k: v for k, v in self.spec.items() if v is not None}
        txt = f"{c['history']} output times ({said.get('write', DEFAULT)}"
        if first:
            txt += f", the first {first:g} h after the model start"
        txt += f"), maps at {c['maps']}" + (f", GRIB2 at {c['grib']}" if c["grib"] else "")
        if c["carriers"]:
            txt += f", {c['carriers']} carrier frame{'s' if c['carriers'] > 1 else ''}"
        return txt

    def meta(self) -> dict:
        """What a restart set keeps so a resume builds the same schedule."""
        return {"offset_h": self.offset_h, **self.spec, "buckets_h": list(self.buckets_h), "catalog_h": self.catalog_h}

    def doc(self, start: datetime) -> dict:
        times = []
        for n in self._sorted:
            e = dict(self.entries[n])
            valid = start + timedelta(seconds=round(n * self.dt, 3))
            e["valid"] = f"{valid:%Y-%m-%dT%H:%M:%S}"
            e["file"] = f"wrfout_d01_{valid:%Y-%m-%d_%H:%M:%S}"
            e["tag"] = self._tag(n)
            times.append(e)
        return {"version": 1, "start": f"{start:%Y-%m-%dT%H:%M:%S}",
                "first_output": f"{start + timedelta(hours=self.offset_h):%Y-%m-%dT%H:%M:%S}",
                "offset_h": self.offset_h, "hours": self.hours, "dt": self.dt, **self.spec,
                "lean_fields": list(LEAN_FIELDS), "light_fields": list(self.light_fields or ()),
                "catalog_h": self.catalog_h, "catalog_uncompressed": True,
                "counts": self.counts(), "notes": list(self.notes), "times": times}

    def write(self, out_dir, start: datetime) -> Path:
        p = Path(out_dir) / FILE
        tmp = p.with_name(p.name + ".part")
        tmp.write_text(json.dumps(self.doc(start), indent=1) + "\n")
        tmp.replace(p)
        return p


def read(run_dir) -> dict | None:
    """<run>/OUTPUTS.json, or None for a run written before the schedule (every history file is an output)."""
    try:
        return json.loads((Path(run_dir) / FILE).read_text())
    except (OSError, ValueError):
        return None


def files(run_dir, what: str = "history") -> set | None:
    """The history file names of a run that are `what` ("history": not a carrier; "columns": not a carrier and not a
    light frame; "maps"; "grib"; "all": every scheduled file, carriers included), or None when the run has no list
    (every file counts)."""
    doc = read(run_dir)
    if doc is None:
        return None
    if what == "all":
        return {t["file"] for t in doc["times"]}
    if what == "history":
        return {t["file"] for t in doc["times"] if not t.get("carrier")}
    if what == "columns":
        # a sounding column needs the whole state: not a carrier and not a light frame (only what the maps read)
        return {t["file"] for t in doc["times"] if not t.get("carrier") and not t.get("light")}
    return {t["file"] for t in doc["times"] if t.get(what)}


def tag_of(valid: datetime, start: datetime) -> str:
    """fHHH for a whole forecast hour, fHHHmMM off the hour (the maps' tag; GRIB2 names take the same minutes)."""
    mins = int(round((valid - start).total_seconds() / 60.0))
    return f"f{mins // 60:03d}" if mins % 60 == 0 else f"f{mins // 60:03d}m{mins % 60:02d}"
