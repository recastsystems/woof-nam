"""woof-nam suite: one cycle of the WOOF-NAM suite, the NAM 12 km parent and the five nests NCEP ran beside it
(CONUS, Alaska, Hawaii and Puerto Rico at 3 km, fire weather at 1.5 km), on every card of one machine.

    woof-nam suite --cycle 2026110600 --out runs/suite-2026110600                # parent + nests, every card
    woof-nam suite --cycle 2026110600 --out runs/s --parent-run runs/p --parent-ended runs/p/.ended   # nests only
    woof-nam suite --cycle 2026110600 --out runs/s --dry-run                     # the plan, nothing run

What runs comes from the table namgpu/parm/nam_suite.json (lengths, grids, card layouts, file names) and the
fire-weather point from namgpu/parm/nam_firewx_loc (NCEP's file: one "HHz lat lon" line per cycle). The stages:
  parent   woof-nam fetch (GFS) + woof-nam run --domain conus12-suite: the NAM 12 km grid with the hand-off set in
           its hourly history (one card), 84 h. Skipped with --parent-run (the WOOF worker runs its own).
  nests    each nest, once the parent's history covers its hours: woof-nam nest --source parent (one-way, on the
           cards its layout row gives it; nests sharing a card start, in plan order, only when the suite can reserve
           their memory estimate on it, and each door sizes itself to that reservation), 60 h (fire weather 36 h).
  products per domain, while it runs: `woof-nam products --follow` into <out>/publish/<domain>/ (GRIB2 first, its
           .idx renamed in after it). The 12 km parent writes NOAA's awphys files; a nest writes NOAA's hiresf
           files when the installed woof-nam has that nest's writer (`woof-nam products --set NEST`), else the suite says
           so and keeps that nest's history.
  finish   <out>/publish/<domain>/cycle.json (state, grid, packing, every file with bytes and SHA-256),
           <out>/publish/firewxnest/<prefix>.tHHz.firewxnest_center_latlon, <out>/suite-receipt.json (per stage
           wall, ms/step, peak VRAM, cores), and each nest's history deleted once its GRIB2 hours are all there.
A domain that fails never stops the others; the receipt and that domain's cycle.json say "failed" and why.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from namgpu.suite import table as T


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def woof_nam() -> list[str]:
    return [sys.executable, "-m", "namgpu.cli"]


def visible_cards() -> list[str]:
    v = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if v:
        return [x.strip() for x in v.split(",") if x.strip()]
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout
        return [x.strip() for x in out.splitlines() if x.strip()]
    except (OSError, subprocess.SubprocessError):
        return []


def nest_door_options() -> str:
    """The installed `woof-nam nest --help` (the suite passes --ncep-nest when the door has it)."""
    p = subprocess.run([*woof_nam(), "nest", "--help"], capture_output=True, text=True, timeout=120)
    return p.stdout if p.returncode == 0 else ""


def products_nest_sets() -> set[str]:
    """The nests the installed `woof-nam products` writes NOAA's hiresf files for (its --set choices; master bce24d9
    retired the 0.1.3 nest refusal). Empty: no nest writer (the proof cycle of 2026-11-06 ran a build without one and
    published no nest file)."""
    p = subprocess.run([*woof_nam(), "products", "--help"], capture_output=True, text=True, timeout=120)
    if p.returncode != 0 or "--set" not in p.stdout:
        return set()
    return {n for n in T.NEST_NAMES if n in p.stdout}


def _tree_pids(root: int) -> set[int]:
    """root and every process below it (from /proc; just root where /proc cannot be read)."""
    kids: dict[int, list[int]] = {}
    try:
        for p in os.listdir("/proc"):
            if not p.isdigit():
                continue
            try:
                with open(f"/proc/{p}/stat") as f:
                    ppid = int(f.read().rsplit(")", 1)[1].split()[1])
            except (OSError, ValueError, IndexError):
                continue
            kids.setdefault(ppid, []).append(int(p))
    except OSError:
        return {root}
    out, todo = set(), [root]
    while todo:
        q = todo.pop()
        if q not in out:
            out.add(q)
            todo.extend(kids.get(q, []))
    return out


def _used_gib_by_pid(card: str) -> dict[int, float]:
    """Card memory in use per process (nvidia-smi compute apps); empty when it cannot be read. Inside a container
    nvidia-smi may name host pids, which match nothing here: a run's use then reads 0 and its whole reservation
    stays held (slower, never over the card)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
                              "-i", card], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    used: dict[int, float] = {}
    for ln in out.splitlines():
        try:
            pid, mib = (x.strip() for x in ln.split(","))
            used[int(pid)] = used.get(int(pid), 0.0) + float(mib) / 1024.0
        except ValueError:
            continue
    return used


class Proc:
    """A child with its own log, started with a card set, timed."""

    def __init__(self, name: str, argv: list[str], log: Path, env: dict, cwd: Path | None = None):
        self.name, self.argv, self.log = name, argv, log
        log.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(log, "a")
        self.fh.write(f"# {_now()} {' '.join(argv)}\n")
        self.fh.flush()
        self.t0 = time.monotonic()
        self.started_utc = _now()
        self.p = subprocess.Popen(argv, stdout=self.fh, stderr=subprocess.STDOUT, env=env, cwd=cwd)

    def wait(self) -> int:
        rc = self.p.wait()
        self.fh.close()
        self.wall_s = round(time.monotonic() - self.t0, 1)
        return rc


class Suite:
    def __init__(self, a, t: dict, plan: dict, cycle: datetime):
        self.a, self.t, self.plan, self.cycle = a, t, plan, cycle
        self.out = a.out.resolve()
        self.pub = self.out / "publish"
        self.logs = self.out / "logs"
        self.rec: dict = {"suite": "woof-nam suite", "cycle": plan["cycle"], "started_utc": _now(),
                          "argv": sys.argv[1:], "domains": {}, "declared": t.get("declared", [])}
        self.lock = threading.Lock()
        self.parent_run: Path | None = a.parent_run.resolve() if a.parent_run else None
        self.parent_ended: Path | None = a.parent_ended.resolve() if a.parent_ended else None
        self.parent_failed = False
        self.nest_products: set[str] = set()
        self.door_options = ""
        self.ncep_rows: set[str] = set()
        # card memory reserved for the nest doors this suite has started (card -> nest -> GiB), the nests waiting for
        # a reservation (plan order) and the reservations granted but not yet taken up (nest -> GiB, None = no share)
        self.cv = threading.Condition()
        self.claims: dict[str, dict[str, float]] = {}
        self.claim_procs: dict[str, "Proc"] = {}
        self.waiting: list[str] = []
        self.granted: dict[str, float | None] = {}
        self.cands: dict[str, list] = {}

    def offset_h(self) -> float:
        """Hours the parent starts before the time asked (woof-nam suite --lead H on an hour GFS does not post: the
        parent starts at the nearest posted hour before it, runs that much longer and counts its outputs from H)."""
        return float(self.plan.get("offset_h") or 0.0)

    def nest_start(self) -> datetime:
        """When the nests start: the time asked (the parent's start plus its offset)."""
        return self.cycle + timedelta(hours=self.offset_h())

    # ---- card memory for nests that share a card
    #
    # Five nests started together on one 95 GiB card (release check 2026-11-10, suite-1h-d): each door read 94.4 GiB
    # free at the same moment and sized itself to the whole card (226.6 GiB of estimates), and CONUS, Hawaii and fire
    # weather ran out of memory. Now the suite reserves each door's estimate before starting it, in plan order, and
    # passes the reservation as NAMGPU_CARD_SHARE_GIB (domain.free_gib reads no more than that), so the doors that run
    # together fit the card together. A nest that does not fit waits for a running one to end.

    def _candidates(self, d: str) -> list[dict]:
        """The door's own estimates in its own order (domain.preflight_cards: memory path, then RRTM rows), each as
        {card: estimate}; computed once (they do not depend on free memory)."""
        if d not in self.cands:
            from namgpu.nest import domain
            row = self.plan["domains"][d]
            cards, g = row["cards"], row["grid"]
            self.cands[d] = [{c: domain.vram_estimate(g, r, len(cards), c, path=path) for c in cards}
                             for path in domain._paths_to_try(cards[0]) for r in domain.RRTM_ROW_CHOICES]
        return self.cands[d]

    def _avail(self, card: str) -> float:
        """Free memory on the card now, less what the running doors reserved and have not taken yet."""
        from namgpu.nest import domain
        free = domain.free_gib(card)
        mine = self.claims.get(card) or {}
        used = _used_gib_by_pid(card) if mine else {}
        for m, gib in mine.items():
            p = self.claim_procs.get(m)
            took = sum(used.get(q, 0.0) for q in _tree_pids(p.p.pid)) if p is not None else 0.0
            free -= max(0.0, gib - took)
        return free

    def _grant(self) -> None:
        """Called with self.cv held: grant every waiting nest that fits, in plan order."""
        from namgpu.nest import domain
        if not self.waiting:
            return
        margin = domain.VRAM["margin_gib"]
        cards = {c for w in self.waiting for c in self.plan["domains"][w]["cards"]}
        avail = {c: self._avail(c) for c in cards}
        for w in [d for d in self.plan["domains"] if d in self.waiting]:
            wc = self.plan["domains"][w]["cards"]
            pick = next((es for es in self._candidates(w) if all(es[c]["total_gib"] + margin <= avail[c] for c in wc)),
                        None)
            if pick is None:
                smallest = min(max(es[c]["total_gib"] for c in wc) for es in self._candidates(w)) + margin
                if smallest > min(domain.card_total_gib(c) or 1e9 for c in wc) and not any(self.claims.get(c) for c in wc):
                    # larger than the card even when empty: the door refuses it by name, as before
                    self.granted[w] = None
                    self.waiting.remove(w)
                continue
            share = round(max(es["total_gib"] for es in pick.values()) + margin + 0.01, 2)
            for c in wc:
                self.claims.setdefault(c, {})[w] = share
                avail[c] -= share
            e = next(iter(pick.values()))
            self.granted[w] = share
            self.waiting.remove(w)
            self.say(f"{w}: starts with {share} GiB of card {','.join(wc)} reserved ({e['memory_path']} memory path, "
                     f"NAMGPU_RRTM_ROWS {e['rrtm_rows']}); then {round(min(avail[c] for c in wc), 2)} GiB left for others")

    def _admit(self, d: str) -> float | None:
        """Wait until d's door can have its estimate of the card; the reserved GiB (None: no reservation)."""
        rec = self.dom(d)
        with self.cv:
            self.waiting.append(d)
            said = False
            while d not in self.granted:
                self._grant()
                if d in self.granted:
                    break
                if not said:
                    said = True
                    rec["state"] = "waiting-for-card-memory"
                    t0 = time.monotonic()
                    self.say(f"{d}: waits for room on card {','.join(self.plan['domains'][d]['cards'])} "
                             f"(the running nests' reservations leave too little)")
                self.cv.wait(timeout=15)
            if said:
                rec["waited_for_card_memory_s"] = round(time.monotonic() - t0, 1)
            return self.granted.pop(d)

    def _release(self, d: str) -> None:
        with self.cv:
            for c in self.plan["domains"][d]["cards"]:
                (self.claims.get(c) or {}).pop(d, None)
            self.claim_procs.pop(d, None)
            self.cv.notify_all()

    # ---- helpers

    def env(self, cards: list[str] | None, extra: dict | None = None, nest: bool = False) -> dict:
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        if nest:
            # a nest's processes read their own grid: the door writes and passes its configfile; a caller's
            # NAMGPU_CONFIGFILE (the WOOF worker sets the parent's) would put the 12 km grid under them
            env.pop("NAMGPU_CONFIGFILE", None)
        if cards is not None and len(cards) == 1:
            env["CUDA_VISIBLE_DEVICES"] = cards[0]
        env.update(extra or {})
        return env

    def say(self, msg: str):
        print(f"woof-nam suite {_now()}: {msg}", flush=True)

    def dom(self, d: str) -> dict:
        with self.lock:
            return self.rec["domains"].setdefault(d, {"state": "waiting"})

    def parent_hours_written(self) -> tuple[int, bool]:
        """(the last parent hour whose history file is complete, whether the parent has ended)."""
        ended = (self.parent_ended is not None and self.parent_ended.exists()) or self.parent_failed or \
            (self.parent_run is not None and (self.parent_run / "forecast-receipt.json").is_file())
        if self.parent_run is None or not self.parent_run.is_dir():
            return -1, ended
        hours = set()
        for f in self.parent_run.glob("wrfout_d01_*"):
            try:
                v = datetime.strptime(f.name[len("wrfout_d01_"):], "%Y-%m-%d_%H:%M:%S")
            except ValueError:
                continue
            hours.add(int(round((v - self.cycle.replace(tzinfo=None)).total_seconds() / 3600)))
        # a parent that started offset_h before the time asked writes from the time asked (its hours before are not
        # outputs), so the hours are counted whole from there
        h = int(round(self.offset_h())) - 1
        while h + 1 in hours:
            h += 1
        # one card writes each hour in place, the next only after it: hour h is whole once h + 1 exists or the run ended
        return (h if ended else h - 1), ended

    # ---- parent

    def run_parent(self):
        d = self.t["parent"]["domain"]
        row = self.plan["domains"][d]
        rec = self.dom(d)
        case = (self.a.cases_root / f"{self.cycle:%Y-%m-%d_%H}z").resolve()
        run = self.out / d / "run"
        self.parent_run = run
        self.parent_ended = self.out / d / ".ended"
        try:
            rec.update(state="fetch", started_utc=_now())
            t0 = time.monotonic()
            lead = int(self.plan.get("lead_h") or 0)
            # the fetch plans the hour asked (wanted_lead_h): it starts at `lead` and records the offset in the case
            # (raw/START.json), which the run reads to run that much longer and count its outputs from the hour asked
            want = int(self.plan.get("wanted_lead_h") or lead)
            f = Proc("fetch", [*woof_nam(), "fetch", "--cycle", f"{self.cycle - timedelta(hours=lead):%Y%m%d%H}",
                               *(["--lead", str(want)] if want else []), "--hours", f"{row['hours']:g}",
                               "--case-dir", str(case), "--gfs-only",
                               "--workers", str(max(8, min(32, (os.cpu_count() or 8) * 2)))],
                     self.logs / "12km-fetch.log", self.env(None))
            if f.wait():
                raise RuntimeError(f"fetch exit {f.p.returncode} (log {f.log})")
            rec["fetch_s"] = round(time.monotonic() - t0, 1)
            rec["state"] = "forecast"
            argv = [*woof_nam(), "run", "--case", str(case), "--source", row["source"], "--hours", f"{row['hours']:g}",
                    "--out", str(run), "--no-maps", "--no-columns", "--domain", row["row"],
                    *(["--lead", str(lead)] if lead else []), *self.output_words("parent")]
            p = Proc("12km", argv, self.logs / "12km-run.log", self.env(row["cards"]))
            prod = None
            if not self.a.no_products:
                prod = self.start_products(d, run, row["hours"], nest=None, cards=row["cards"])
            rc = p.wait()
            self.parent_ended.touch()
            rec.update(forecast_rc=rc, forecast_wall_s=p.wall_s)
            self._engine_receipt(rec, run)
            if prod is not None:
                rec["products_rc"] = prod.wait()
                rec["products_tail_s"] = round(time.monotonic() - t0 - rec["fetch_s"] - p.wall_s, 1)
            rec["state"] = "complete" if rc == 0 else "failed"
            if rc:
                rec["error"] = f"woof-nam run exit {rc} (log {p.log})"
        except Exception as e:  # noqa: BLE001 - every nest then fails by name, never hangs
            rec.update(state="failed", error=f"{type(e).__name__}: {e}")
        finally:
            if rec.get("state") != "complete":
                self.parent_failed = True
            if self.parent_ended is not None:
                self.parent_ended.touch()
            rec["finished_utc"] = _now()

    def output_words(self, who: str) -> list[str]:
        """The output schedule words (namgpu/outputs.py) for the parent or a nest. The parent of a suite with nests
        also writes every hour its nests read for their edges (--carry 1h: state carriers, never exported)."""
        a = self.a
        if who == "parent":
            out = [("--write", a.write), ("--grib-at", a.grib_at)]
            nests = any(r["kind"] == "nest" for r in self.plan["domains"].values())
            if a.write and nests:
                out.append(("--carry", "1h"))
        else:
            out = [("--write", a.nest_write), ("--grib-at", a.nest_grib_at)]
        return [x for k, v in out if v for x in (k, str(v))]

    def grib_hours(self, d: str, run: Path, hours: float) -> set[int]:
        """The GRIB2 hours a domain owes: its run's output list (OUTPUTS.json, whole hours since the start), else every
        hour 0..hours as before."""
        from namgpu import outputs
        doc = outputs.read(run)
        if doc is None:
            return set(range(0, int(hours) + 1))
        listed = {int(round(t["model_h"])) for t in doc["times"] if t.get("grib")}
        # a list that names no GRIB2 time owes every hour, so a run that wrote none is never called complete
        return listed or set(range(0, int(hours) + 1))

    # ---- products

    def start_products(self, d: str, run: Path, hours: float, nest: str | None, cards: list[str] | None = None):
        out = self.pub / d
        out.mkdir(parents=True, exist_ok=True)
        until = run.parent / ".ended"
        # the parent of a suite from an unposted hour started `offset` hours before the time asked: its GRIB2 hours are
        # model hours offset .. offset + hours (the follower counts from the run's start); a nest starts at the time asked
        off = int(round(self.offset_h())) if nest is None else 0
        argv = [*woof_nam(), "products", "--run", str(run), "--out", str(out), "--hours", f"{off}-{off + int(hours)}",
                "--prefix", self.t["prefix"], "--follow", "--until", str(until)]
        if nest is not None:
            # the nest's own NOAA set, and enough writers in flight to keep up with the forecast (a writer on the
            # whole conusnest grid takes minutes an hour; the table row gives the count, each needs its host RAM)
            argv += ["--set", nest, "--workers", str(int(self.t["nests"][nest].get("writers", 2)))]
        # the writer's GPU post on the domain's own first card (with every card visible it took any card with room,
        # on a shared box another lane's; suite proof 2026-11-06), its CPU work at low priority
        return Proc(f"{d}-products", ["nice", "-n", "10", *argv], self.logs / f"{d}-products.log",
                    self.env(cards[:1] if cards else None, nest=nest is not None))

    # ---- nests

    def run_nest(self, d: str):
        row = self.plan["domains"][d]
        rec = self.dom(d)
        # the door's parent fetch (nest/cli.py fetch_parent) checks the parent's history every 3 h up to
        # max(6, hours): a nest waits for every hour that check reads
        # (counted from the parent's start: a nest starts at the time asked, offset_h after it)
        need = int(round(self.offset_h())) + T.parent_hours_for_nest(row["hours"])
        rec.update(state="waiting-for-parent", hours=row["hours"], cards=len(row["cards"]))
        t_wait = time.monotonic()
        while True:
            h, ended = self.parent_hours_written()
            if h >= need:
                break
            if ended:
                rec.update(state="failed", error=f"the parent ended at hour {h}, before this nest's {need} h")
                return
            time.sleep(10)
        rec["waited_for_parent_s"] = round(time.monotonic() - t_wait, 1)
        out = self.out / d
        out.mkdir(parents=True, exist_ok=True)
        until = out / ".ended"
        until.unlink(missing_ok=True)
        rec.update(state="waiting-for-card-memory", started_utc=_now())
        argv = [*woof_nam(), *T.door_argv(d, row, self.nest_start(), self.parent_run, out, self.door_options,
                                          self.ncep_rows, self.output_words("nest"))]
        rec["door"] = "ncep-nest" if "--ncep-nest" in argv else "centre-size (no NCEP row for this nest in the door)"
        prod = None
        if not self.a.no_products and d in self.nest_products:
            prod = self.start_products(d, out / "run", row["hours"], nest=d, cards=row["cards"])
        for attempt in (1, 2):
            share = self._admit(d)
            rec["state"] = "forecast"
            rec.setdefault("card_share_gib", []).append(share)
            extra = {"NAMGPU_CARD_SHARE_GIB": repr(share)} if share is not None else None
            with self.cv:
                p = Proc(d, argv, self.logs / f"{d}-nest.log", self.env(row["cards"], extra, nest=True))
                self.claim_procs[d] = p
            try:
                rc = p.wait()
            finally:
                self._release(d)
            # A nest that shares the parent's card can run out of memory before its forecast (its start and boundary
            # build is not in the door's estimate): the Alaska nest on the 2-card layout stopped at init-gfs-bocos
            # with the 12 km parent still on its card (suite proof 2026-11-06 18:14Z). It runs once more when the
            # parent's card is free.
            # The same holds for nests that share a card: on the one-card layout the CONUS nest saw 67.9 GiB free once
            # the parent had ended, started beside the Alaska nest at NAMGPU_RRTM_ROWS 8 and ran out of memory at its
            # 62.6 GiB pool cap before its first hour (release check 2026-11-09). So the second attempt waits until
            # every other domain on its cards has ended, and has the card to itself.
            mates = self._card_mates(d)
            if rc == 0 or attempt == 2 or not self._init_out_of_memory(p.log) or not mates:
                break
            rec["retried_alone"] = {"first_rc": rc, "first_wall_s": p.wall_s, "card_mates": mates}
            # said in suite.log too: a quiet retry hid three of five nests out of memory (suite-1h-d, 2026-11-10)
            self.say(f"{d}: out of memory before its first hour (rc {rc}, log {p.log}); it runs once more when "
                     f"{', '.join(mates)} have left its card")
            rec["state"] = "waiting-for-card"
            while self._busy_mates(d):
                time.sleep(10)
            # the writer follows until the run's forecast-receipt.json: a failed attempt's receipt would stop it at
            # the second attempt's first hour (the Hawaii rerun's writer quit on the first run's receipt, 2026-11-06)
            (out / "run" / "forecast-receipt.json").unlink(missing_ok=True)
            if prod is not None and prod.p.poll() is not None:
                prod.wait()
                prod = self.start_products(d, out / "run", row["hours"], nest=d, cards=row["cards"])
        until.touch()
        rec.update(forecast_rc=rc, door_wall_s=p.wall_s)
        rp = out / "nest-receipt.json"
        if rp.is_file():
            try:
                nr = json.loads(rp.read_text())
                fc = nr.get("forecast") or {}
                rec["door_receipt"] = {"grid": nr.get("grid"), "keys": {k: (nr.get("keys") or {}).get(k) for k in
                                                                 ("dt", "steps_per_hour", "nrads", "lnsh")},
                               "stages": nr.get("stages"), "vram_peak_gib": nr.get("vram_peak_gib"),
                               "ms_per_step": fc.get("ms_per_step"), "ms_per_step_by_rank": fc.get("ms_per_step_by_rank"),
                               "steps_done": fc.get("steps_done"), "run_s": fc.get("run_s"),
                               "nonfinite_end": fc.get("nonfinite_end"), "preflight_pick": (nr.get("preflight") or {}).get("pick"),
                               "single_core_windows_over_30s": nr.get("single_core_windows_over_30s"),
                               "refused": nr.get("refused"), "failed_stage": nr.get("failed_stage")}
            except (OSError, ValueError) as e:
                rec["door_receipt"] = f"receipt unreadable: {e}"
        if prod is not None:
            rec["products_rc"] = prod.wait()
        elif not self.a.no_products:
            rec["products"] = (f"not written: this woof-nam has no {d} product writer (woof-nam products --set {d}); "
                               "the nest's history is kept")
        rec["state"] = "complete" if rc == 0 else "failed"
        if rc:
            rec["error"] = f"woof-nam nest exit {rc} (log {p.log})"
        rec["finished_utc"] = _now()

    def _init_out_of_memory(self, log: Path) -> bool:
        """Whether the door's last attempt (the tail of its log) stopped out of memory before its first forecast hour
        (the door's receipt names no error for it)."""
        try:
            tail = log.read_text(errors="replace").splitlines()[-400:]
        except OSError:
            return False
        last = max((i for i, ln in enumerate(tail) if ln.startswith("# ")), default=0)   # Proc's header line
        tail = tail[last:]
        return any("OutOfMemoryError" in ln for ln in tail) and not any(ln.startswith("fh ") for ln in tail)

    def _card_mates(self, d: str) -> list[str]:
        """The other domains of this cycle planned on any of d's cards (the parent only when this suite runs it)."""
        pd = self.t["parent"]["domain"]
        mine = set(self.plan["domains"][d]["cards"])
        return [m for m, row in self.plan["domains"].items()
                if m != d and mine & set(row.get("cards") or []) and (m != pd or self.a.parent_run is None)]

    def _busy_mates(self, d: str) -> list[str]:
        """The card mates of d still on the card: the parent until it ends, a nest while its forecast runs."""
        pd = self.t["parent"]["domain"]
        busy = []
        for m in self._card_mates(d):
            if m == pd:
                if not self.parent_hours_written()[1]:
                    busy.append(m)
            elif self.dom(m).get("state") == "forecast":
                busy.append(m)
        return busy

    def _engine_receipt(self, rec: dict, run: Path):
        rp = run / "forecast-receipt.json"
        if rp.is_file():
            try:
                fr = json.loads(rp.read_text())
                rec["engine"] = {k: fr.get(k) for k in ("memory_path", "init_s", "tdfi_s", "run_s", "ms_per_step",
                                                         "steps_done", "nonfinite_end", "card_used_gib", "pool_end_gib",
                                                         "error")}
            except (OSError, ValueError):
                pass

    # ---- finish

    def finish_domain(self, d: str):
        row = self.plan["domains"][d]
        rec = self.dom(d)
        if self.a.no_products:
            # forecasts only: no GRIB2 is due, so the domain's verdict is its forecast's, and its history stays (it is
            # the run's only output). The suite counted the GRIB2 hours it had been told not to write and returned
            # rc 1 after a clean 12 h forecast ("13 GRIB2 hours missing", t3k-conus4-12h-noprod, 2026-11-10).
            rec["publish_state"] = "not written (--no-products)"
            return
        folder = self.pub / d
        folder.mkdir(parents=True, exist_ok=True)
        cyc = self.nest_start() if row["kind"] == "nest" else self.cycle     # a nest's files are named for its start
        if d == "firewxnest" and row.get("firewx"):
            side = self.t["nests"][d]["side_file"].format(prefix=self.t["prefix"], HH=f"{cyc:%H}")
            (folder / side).write_text(T.write_center_latlon(row["firewx"]["lat"], row["firewx"]["lon"]))
        files, hours = [], set()
        for f in sorted(folder.iterdir()):
            if f.name.startswith(".") or f.name in ("cycle.json", "index.json", "DONE") or not f.is_file():
                continue
            fh = T.hour_of(self.t, d, f.name)
            if f.name.endswith(".grib2") and fh is None:
                continue
            if f.name.endswith(".grib2") and not (folder / (f.name + ".idx")).is_file():
                continue                     # a file whose .idx never landed is not finished
            if fh is not None and not f.name.endswith(".idx"):
                hours.add(fh)
            files.append({"name": f.name, "bytes": f.stat().st_size, "sha256": _sha256(f)})
        want = self.grib_hours(d, self.out / d / "run" if row["kind"] == "nest" else (getattr(self, "parent_run", None) or Path("-")),
                               row["hours"])
        missing = sorted(want - hours)
        rec["grib2_hours"] = len(hours)
        rec["grib2_missing"] = missing
        rec["grib2_bytes"] = sum(x["bytes"] for x in files if x["name"].endswith(".grib2"))
        state = "complete" if rec.get("state") == "complete" and not missing else "failed"
        if rec.get("state") == "complete" and missing:
            rec.setdefault("error", f"{len(missing)} GRIB2 hours missing: {missing[:6]}")
        doc = {"model": "WOOF-NAM", "domain": d, "cycle": cyc.strftime("%Y-%m-%dT%H:%M:%SZ"), "state": state,
               "hours": row["hours"], "files": files, "hours_available": len(hours),
               "first_file_utc": rec.get("started_utc"), "finished_utc": _now(),
               "grid": row.get("product_grid") or self.t["parent"].get("product_grid"),
               "packing": row.get("packing") or self.t["parent"].get("packing"),
               **({"firewx_center": {"lat": row["firewx"]["lat"], "lon": row["firewx"]["lon"],
                                     "parent_nest": row["firewx"]["region"]}} if row.get("firewx") else {})}
        (folder / "cycle.json").write_text(json.dumps(doc, indent=1) + "\n")
        rec["publish_state"] = state
        # raw nest history: deleted once every GRIB2 hour of the nest is kept (the recipe reruns it)
        if row["kind"] == "nest" and state == "complete" and not self.a.keep_history:
            run = self.out / d / "run"
            if run.is_dir():
                freed = sum(p.stat().st_size for p in run.glob("wrfout_d01_*"))
                for p in run.glob("wrfout_d01_*"):
                    p.unlink()
                rec["history_deleted_gb"] = round(freed / 1e9, 1)

    def run(self) -> int:
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "suite-plan.json").write_text(json.dumps(self.plan, indent=1) + "\n")
        nests = [d for d, r in self.plan["domains"].items() if r["kind"] == "nest"]
        self.door_options = nest_door_options() if nests else ""
        self.ncep_rows = T.ncep_rows()
        self.rec["door_ncep_rows"] = sorted(self.ncep_rows)
        if nests and not self.a.no_products:
            self.nest_products = products_nest_sets()
            self.rec["nest_products"] = sorted(self.nest_products)
        if nests and not self.a.no_warm:
            t0 = time.monotonic()
            w = Proc("warm", [sys.executable, "-c", "from namgpu import kernels; r = kernels.warm(); "
                              "print({k: r[k] for k in ('workers', 'modules', 'failed')})"],
                     self.logs / "warm.log", self.env(self.plan["cards"][:1]))
            self.rec["warm"] = {"rc": w.wait(), "wall_s": round(time.monotonic() - t0, 1)}
        threads = []
        pd = self.t["parent"]["domain"]
        if pd in self.plan["domains"] and self.a.parent_run is None:
            threads.append(threading.Thread(target=self.run_parent, name="parent"))
        elif self.a.parent_run is None and nests:
            raise SystemExit("woof-nam suite: nests need the parent: run it too, or give --parent-run")
        for d in nests:
            threads.append(threading.Thread(target=self.run_nest, args=(d,), name=d))
        for th in threads:
            th.start()
        last = 0.0
        while any(th.is_alive() for th in threads):
            time.sleep(5)
            if time.monotonic() - last > 60:
                last = time.monotonic()
                with self.lock:
                    states = {d: r.get("state") for d, r in self.rec["domains"].items()}
                h, _ = self.parent_hours_written()
                ph = f"parent hour {h} complete" if h >= 0 else "parent: no hour complete yet"
                self.say(f"{ph}; " + ", ".join(f"{d} {s}" for d, s in states.items()))
                self._write()
        for d in self.plan["domains"]:
            if d == pd and self.a.parent_run is not None:
                continue
            try:
                self.finish_domain(d)
            except Exception as e:  # noqa: BLE001
                self.dom(d).update(publish_state="failed", error=f"finish: {type(e).__name__}: {e}")
        self.rec["finished_utc"] = _now()
        self._write()
        bad = self.not_complete()
        self.say("done: " + ("every domain complete" if not bad else f"not complete: {bad}"))
        return 0 if not bad else 1

    def not_complete(self) -> list[str]:
        """The domains that did not finish: with --no-products, a forecast that did not complete; otherwise a domain
        whose GRIB2 set is not complete."""
        if self.a.no_products:
            return [d for d, r in self.rec["domains"].items() if r.get("state") != "complete"]
        return [d for d, r in self.rec["domains"].items() if r.get("publish_state") != "complete"]

    def _write(self):
        with self.lock:
            (self.out / "suite-receipt.json").write_text(json.dumps(self.rec, indent=1, default=str) + "\n")


def build_parser() -> argparse.ArgumentParser:
    from namgpu.cli import HelpFormatter
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=HelpFormatter,
                                 epilog="\n\n".join(__doc__.split("\n\n")[1:]))
    ap.add_argument("--cycle", required=True, help="YYYYMMDDHH, a 00/06/12/18Z cycle (with --lead: the GFS run the "
                    "start is read from)")
    ap.add_argument("--lead", type=int, default=0, metavar="H",
                    help="start the parent from forecast hour H of the --cycle GFS run instead of its analysis: the "
                         "suite starts at cycle + H and every file is named for that start time; the nests follow "
                         "the parent. An hour GFS does not post (it goes 3-hourly after 120) starts at the nearest "
                         "posted hour before it, runs that much longer with its outputs counted from H and its nests starting at H, and the suite says so; a parent longer than the posted hours is cut "
                         "there. With --parent-run: the hour that parent was started from")
    ap.add_argument("--write", default=None, metavar="SPEC", help="the parent's output times (woof-nam run --write: "
                    "1h, 3h, 0,6,12 or 0-24/1h,24-84/3h); default every hour. Its nests' edge hours are written too")
    ap.add_argument("--grib-at", default=None, metavar="SPEC", help="the parent's GRIB2 times, when not every "
                    "--write time (whole hours)")
    ap.add_argument("--nest-write", default=None, metavar="SPEC", help="every nest's output times (15m, 1h, ...)")
    ap.add_argument("--nest-grib-at", default=None, metavar="SPEC", help="every nest's GRIB2 times (whole hours)")
    ap.add_argument("--out", type=Path, required=True, help="the cycle's folder (domains, logs, publish/)")
    ap.add_argument("--card-list", default="", help="comma list of the cards to use (default: every visible card)")
    ap.add_argument("--domains", default="", help="comma list of domains (default all: " + ", ".join(T.SUITE_ORDER) + ")")
    ap.add_argument("--parent-run", type=Path, default=None,
                    help="the 12 km parent's run folder, written by someone else with the hand-off set (woof-nam run "
                         "--domain conus12-suite); the suite then runs the nests only")
    ap.add_argument("--parent-ended", type=Path, default=None,
                    help="with --parent-run: a file that appears when the parent's forecast has ended")
    ap.add_argument("--hours-cap", type=float, default=None, help="cap every domain's length (a short check); the parent still runs "
                    "to the hours its nests read, at least 6")
    ap.add_argument("--firewx-loc", type=Path, default=None, help="the fire-weather point file (default "
                    "namgpu/parm/nam_firewx_loc, NCEP's last production points)")
    ap.add_argument("--cases-root", type=Path, default=Path(os.environ.get("NAMGPU_CASES", "cases")))
    ap.add_argument("--table", type=Path, default=None, help="the suite table (default namgpu/parm/nam_suite.json)")
    ap.add_argument("--no-products", action="store_true", help="forecasts only, no GRIB2: every history is kept, "
                    "and the run passes when every forecast completes")
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--keep-history", action="store_true", help="keep each nest's history after its GRIB2")
    ap.add_argument("--dry-run", action="store_true", help="write and print the plan only")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    cycle = datetime.strptime(a.cycle, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    t = T.load(a.table)
    want_lead, offset = a.lead, 0.0
    if a.lead:
        # --cycle is the GFS run; the suite's own cycle (case folder, file names, valid times) is its start time
        from namgpu.init import sources
        if cycle.hour not in t["cycles"]:
            raise SystemExit(f"woof-nam suite: refused: --cycle {a.cycle}: with --lead, --cycle is the GFS run "
                             f"({t['cycles']} UTC) and the suite starts {a.lead} h after it")
        # with --parent-run the parent has already run from that hour (the nests only follow its history), so only
        # the hour itself is checked; else the parent's own length must fit in what GFS posts after it
        hours = None if a.parent_run is not None else \
            (t["parent"]["hours"] if a.hours_cap is None else min(t["parent"]["hours"], a.hours_cap))
        if a.parent_run is None:
            # the nearest posted hour at or before the one asked (namgpu/init/window.py): never a refusal over that
            from namgpu.init import window
            try:
                from namgpu.init import fetch_case
                p = window.resolve("gfs", hours=float(hours), cycle=cycle.replace(tzinfo=None), lead=float(a.lead),
                                   posted=fetch_case.posted_for("gfs"))
            except ValueError as e:
                raise SystemExit(f"woof-nam suite: refused: {e}") from None
            if p["lead_h"] != a.lead or p["cut_h"]:
                print(f"woof-nam suite: {p['line']}; the suite starts there (its files are named for that start)",
                      flush=True)
            a.lead = int(p["lead_h"])
            offset = float(p["offset_h"])
            if p["cut_h"] and a.hours_cap is None:
                a.hours_cap = p["model_hours"] - p["offset_h"]
        try:
            sources.check_lead(sources.SOURCES["gfs"], a.lead, None if a.parent_run is not None else
                               (t["parent"]["hours"] if a.hours_cap is None else min(t["parent"]["hours"], a.hours_cap)),
                               cycle.replace(tzinfo=None))
        except ValueError as e:
            raise SystemExit(f"woof-nam suite: refused: {e}") from None
        cycle = cycle + timedelta(hours=a.lead)
        print(f"woof-nam suite: the parent starts {cycle:%Y-%m-%d %H}Z from "
              f"{sources.lead_credit(a.lead, cycle.replace(tzinfo=None))} (a GFS forecast, not an analysis); "
              + (f"its outputs count from {cycle + timedelta(hours=offset):%Y-%m-%d %H}Z, the hour asked, and the "
                 "nests start there" if offset > 1e-9 else "the nests follow it"), flush=True)
    cards = [x.strip() for x in a.card_list.split(",") if x.strip()] or visible_cards()
    if not cards:
        raise SystemExit("woof-nam suite: no cards (nvidia-smi lists none; give --card-list)")
    only = [x.strip() for x in a.domains.split(",") if x.strip()] or None
    if a.parent_run is not None and only is None:
        only = [d for d in T.domains(t) if d != t["parent"]["domain"]]
    plan = T.plan(t, cycle, cards, only, a.hours_cap, a.firewx_loc, lead=a.lead)
    if offset > 1e-9:
        # the parent starts at the posted hour, runs `offset` longer and counts its outputs from the hour asked (its
        # fetch plans that hour, so the case carries the offset the run reads); the nests start at the hour asked
        plan.update(wanted_lead_h=int(want_lead), offset_h=offset,
                    nest_start=(cycle + timedelta(hours=offset)).strftime("%Y-%m-%dT%HZ"))
    if a.dry_run:
        a.out.mkdir(parents=True, exist_ok=True)
        (a.out / "suite-plan.json").write_text(json.dumps(plan, indent=1) + "\n")
        print(json.dumps(plan, indent=1))
        return 0
    return Suite(a, t, plan, cycle).run()


if __name__ == "__main__":
    sys.exit(main())
