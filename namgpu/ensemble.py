"""woof-nam ensemble: run a named member roster (one WOOF-NAM forecast per member) from one command.

A roster is data (JSON, schema woof-nam-roster.v1):

    {"schema": "woof-nam-roster.v1", "name": "sref_nmb", "domain": "sref16", "source": "gefs",
     "about": "...",
     "members": [{"name": "ctl", "source_member": "c00",
                  "configfile": {"convection": "bmj", "newswap": ".false.", "cleffamp": 1.0, ...},
                  "recipe": {...free text the roster carries for the receipt...}}, ...]}

Each member is the roster domain's configfile with that member's keys set (configfile keys, as NAM's own job
scripts set them with sed: physics selectors, the BMJ shallow switches, cleffamp, plus the port key
soil_moisture_scale) and the roster source's member (gefs c00, p01-p30). The door writes
<out>/<member>/configfile, fetches the member's inputs into the shared case folder (members of one source member
share them) and runs `woof-nam run` for it, one member per card at a time over --card-list. The ensemble's
receipt <out>/ensemble-receipt.json lists every member: its configfile keys, its source member, the forecast
receipt's wall time, ms per step and peak memory.

A member whose keys select a scheme this port does not run (WSM6, SAS, GFS PBL, GFDL radiation, ...) is refused by
name before anything is fetched or started: the forecast would otherwise stop at that scheme's first call after the
init (phys-bmj-driver and its siblings refuse unknown selectors). --runnable-only runs the rest of the roster and
lists the refused members in the receipt.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCHEMA = "woof-nam-roster.v1"
# The selectors the WOOF-NAM physics driver runs (namgpu/physics/driver); every other value of these keys stops
# the forecast at the scheme's first call.
PORTED = {"convection": ("bmj", "none"), "microphysics": ("fer_hires",), "turbulence": ("myj",),
          "sfc_layer": ("myj",), "land_surface": ("noah",), "shortwave": ("rrtm",), "longwave": ("rrtm",)}
_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def refusals(member: dict) -> list[str]:
    """The member's selectors this port does not run (empty when it runs)."""
    out = []
    for k, v in member.get("configfile", {}).items():
        if k in PORTED and str(v).strip().lower() not in PORTED[k]:
            out.append(f"{k}={v}")
    return out


def member_configfile(base: Path, keys: dict, extra: dict) -> str:
    """The base configfile with `keys` set: an existing `key:` line takes the new value (its comment kept), a key
    the file does not have is appended. `extra` (port keys the door itself sets) is appended the same way."""
    lines = base.read_text().splitlines()
    want = {**keys, **extra}
    for k in want:
        if not _KEY.match(k):
            raise SystemExit(f"woof-nam ensemble: configfile key {k!r} is not a configfile key")
    done = set()
    for i, line in enumerate(lines):
        m = re.match(r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:\s*)(\S+)(.*)$", line)
        if m and m.group(2) in want and m.group(2) not in done:
            lines[i] = f"{m.group(1)}{m.group(2)}{m.group(3)}{want[m.group(2)]}{m.group(5)}"
            done.add(m.group(2))
    add = [f"{k}: {v}" for k, v in want.items() if k not in done]
    if add:
        lines += ["", "# woof-nam ensemble: member keys"] + add
    return "\n".join(lines) + "\n"


def read_roster(path: Path) -> dict:
    r = json.loads(Path(path).read_text())
    if r.get("schema") != SCHEMA:
        raise SystemExit(f"woof-nam ensemble: {path} is not a {SCHEMA} roster")
    names = [m["name"] for m in r["members"]]
    if len(set(names)) != len(names):
        raise SystemExit(f"woof-nam ensemble: {path}: member names repeat")
    return r


def _peak_gib(rec: dict):
    for k in ("peak_vram_gib", "peak_gib", "vram_peak_gib"):
        if k in rec:
            return rec[k]
    mem = rec.get("memory") or {}
    return mem.get("peak_gib") or mem.get("peak_vram_gib")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", type=Path, required=True, help="the member roster (JSON, woof-nam-roster.v1)")
    ap.add_argument("--case", required=True, help="case folder name YYYY-MM-DD_HHz (start time) under cases/")
    ap.add_argument("--hours", type=int, required=True, help="forecast length, hours")
    ap.add_argument("--out", type=Path, required=True, help="ensemble folder: one subfolder per member")
    ap.add_argument("--members", default="", help="comma list of member names to run (default: every member)")
    ap.add_argument("--card-list", default="0", help="cards to run members on, one member per card at a time")
    ap.add_argument("--runnable-only", action="store_true",
                    help="run the members whose schemes this port runs; list the others as refused")
    ap.add_argument("--no-fetch", action="store_true", help="inputs are already in the case folder")
    ap.add_argument("--dry-run", action="store_true", help="write the member configfiles and the plan, run nothing")
    ap.add_argument("--run-args", default="--no-columns",
                    help="extra arguments for every member's woof-nam run (default: --no-columns)")
    a = ap.parse_args(argv)

    from namgpu import domains, paths
    roster = read_roster(a.roster)
    row = domains.get(roster.get("domain"))
    base = Path(row["configfile_path"])
    source = roster["source"]
    if source not in row["sources"]:
        raise SystemExit(f"woof-nam ensemble: domain {row['name']} starts from {row['sources']}, not {source!r}")
    pick = [m for m in roster["members"] if not a.members or m["name"] in a.members.split(",")]
    refused = {m["name"]: refusals(m) for m in pick if refusals(m)}
    if refused and not a.runnable_only:
        for n, why in refused.items():
            print(f"woof-nam ensemble: member {n}: {', '.join(why)} is not ported to WOOF-NAM", file=sys.stderr)
        raise SystemExit("woof-nam ensemble: refused (members above select schemes this port does not run); "
                         "--runnable-only runs the rest and lists these in the receipt")
    run = [m for m in pick if m["name"] not in refused]
    out = a.out if a.out.is_absolute() else paths.work_root() / a.out
    out.mkdir(parents=True, exist_ok=True)
    case_dir = paths.work_root() / "cases" / a.case
    start = domains.start_of(case_dir)
    plan = []
    for m in run:
        mdir = out / m["name"]
        mdir.mkdir(parents=True, exist_ok=True)
        cf = mdir / "configfile"
        # domain_name keeps the static file shared and gives the member its own boundary folder
        extra = {"source_member": m.get("source_member", "")} if m.get("source_member") else {}
        cf.write_text(member_configfile(base, m.get("configfile", {}), extra))
        plan.append({"member": m["name"], "configfile": str(cf), "source": source,
                     "source_member": m.get("source_member"), "keys": m.get("configfile", {}),
                     "recipe": m.get("recipe")})
    receipt = {"schema": "woof-nam-ensemble-receipt.v1", "roster": str(a.roster), "name": roster.get("name"),
               "domain": row["name"], "source": source, "case": a.case, "start": f"{start:%Y-%m-%dT%H}",
               "hours": a.hours, "cards": a.card_list.split(","), "members": plan,
               "refused": [{"member": n, "not_ported": w} for n, w in refused.items()]}
    (out / "ensemble-plan.json").write_text(json.dumps(receipt, indent=1))
    print(f"woof-nam ensemble: {len(plan)} members to run, {len(refused)} refused; plan {out / 'ensemble-plan.json'}",
          flush=True)
    if a.dry_run:
        return 0

    exe = [sys.executable, "-m", "namgpu.cli"]
    t0 = time.perf_counter()
    # inputs: one fetch per source member (members that share it share the case folder files)
    if not a.no_fetch:
        for mem in sorted({p["source_member"] or "" for p in plan}):
            env = dict(os.environ, NAMGPU_CONFIGFILE=str(base))
            if mem:
                env["NAMGPU_SOURCE_MEMBER"] = mem
            cmd = exe + ["fetch", "--cycle", f"{start:%Y%m%d%H}", "--hours", str(a.hours), "--case-dir",
                         str(case_dir), "--source", source, "--inputs-only"]
            with open(out / f"fetch-{mem or source}.log", "w") as log:
                rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            if rc != 0:
                raise SystemExit(f"woof-nam ensemble: fetch for member {mem or source} failed (exit {rc}); "
                                 f"{out / f'fetch-{mem or source}.log'}")
    t_fetch = time.perf_counter() - t0
    cards = [c.strip() for c in a.card_list.split(",") if c.strip()]
    queue = list(plan)
    running: dict[str, tuple] = {}
    results = {}
    while queue or running:
        for card in cards:
            if card in running or not queue:
                continue
            p = queue.pop(0)
            env = dict(os.environ, NAMGPU_CONFIGFILE=p["configfile"], CUDA_VISIBLE_DEVICES=card)
            if p["source_member"]:
                env["NAMGPU_SOURCE_MEMBER"] = p["source_member"]
            mdir = Path(p["configfile"]).parent
            cmd = exe + ["run", "--case", a.case, "--source", source, "--hours", str(a.hours),
                         "--out", str(mdir / "run")] + a.run_args.split()
            log = open(mdir / "run.log", "w")
            print(f"woof-nam ensemble: member {p['member']} on card {card}", flush=True)
            running[card] = (p, subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT), log,
                             time.perf_counter())
        time.sleep(5)
        for card, (p, proc, log, t1) in list(running.items()):
            if proc.poll() is None:
                continue
            log.close()
            del running[card]
            rpath = Path(p["configfile"]).parent / "run" / "forecast-receipt.json"
            rec = json.loads(rpath.read_text()) if rpath.is_file() else {}
            results[p["member"]] = {"exit": proc.returncode, "card": card, "wall_s": round(time.perf_counter() - t1, 1),
                                    "run_s": rec.get("run_s"), "ms_per_step": rec.get("ms_per_step"),
                                    "peak_gib": _peak_gib(rec), "receipt": str(rpath) if rpath.is_file() else None}
            print(f"woof-nam ensemble: member {p['member']} exit {proc.returncode} "
                  f"({results[p['member']]['wall_s']} s)", flush=True)
    receipt.update(fetch_s=round(t_fetch, 1), wall_s=round(time.perf_counter() - t0, 1), results=results,
                   finished=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"))
    (out / "ensemble-receipt.json").write_text(json.dumps(receipt, indent=1))
    bad = [n for n, r in results.items() if r["exit"] != 0]
    print(f"woof-nam ensemble: done, {len(results) - len(bad)} of {len(results)} members ok; "
          f"{out / 'ensemble-receipt.json'}", flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
