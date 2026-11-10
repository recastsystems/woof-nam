"""Winter and CAD verification for ``woof-nam score --winter``.

Reuses the contingency/amount methods in the recovered namwinter draft. Acquisition
belongs to rw_asos; raw ASOS CSV and NetCDF sampling run in winter_bridge.rs.
Python orchestrates the bridge and scores the small station receipts. No forecast
setting is changed. See docs/winter-score.md for units, proxies and missing data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from namgpu.io import namscore as ns

CLASSES = ("none", "rain", "snow", "freezing_rain", "sleet")
# Reused from the retained CAD diagnostics: Piedmont, Blue Ridge front to Fall Line.
# This is a geographic verification subset, not an automatic synoptic CAD detector.
PIEDMONT = ((-85.0, 32.5), (-85.6, 33.3), (-84.6, 34.4), (-83.4, 34.8),
            (-82.6, 35.1), (-81.7, 35.7), (-80.8, 36.3), (-79.9, 37.2),
            (-78.9, 38.0), (-78.3, 38.6), (-77.4, 38.6), (-77.4, 37.5),
            (-77.9, 36.6), (-78.6, 35.8), (-79.8, 34.9), (-81.0, 34.0),
            (-82.0, 33.5), (-83.6, 32.8))


def in_wedge(lat, lon):
    inside = False
    for (x, y), (xx, yy) in zip(PIEDMONT, PIEDMONT[1:] + PIEDMONT[:1]):
        if (y > lat) != (yy > lat) and lon < x + (lat - y) * (xx - x) / (yy - y):
            inside = not inside
    return inside


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else math.nan
    except (ValueError, TypeError):
        return math.nan


def category(value):
    x = number(value)
    return int(x) if x in range(5) else None


def contingency(hits, false, miss, correct):
    """Binary scores. Unknown reports must be excluded before calling."""
    def ratio(n, d):
        return None if not d else round(n / d, 6)
    n = hits + false + miss + correct
    chance = (hits + false) * (hits + miss) / n if n else 0
    return dict(n=n, hits=hits, false=false, miss=miss, correct=correct,
                hit_rate=ratio(hits, hits + miss), far=ratio(false, hits + false),
                csi=ratio(hits, hits + false + miss), frequency_bias=ratio(hits + false, hits + miss),
                ets=ratio(hits - chance, hits + false + miss - chance))


def ptype_table(matrix):
    """Forecast rows, observation columns, including dry. Also expose conditional wet-only HSS."""
    m = np.asarray(matrix)
    if m.shape != (5, 5) or not np.isfinite(m).all() or (m < 0).any() or (m != np.floor(m)).any():
        raise ValueError("ptype matrix must contain 5x5 nonnegative integer counts")
    m = m.astype(np.int64)
    def skill(a):
        n = int(a.sum())
        correct = int(np.trace(a))
        chance = float(a.sum(0) @ a.sum(1)) / n if n else 0
        return dict(n=n, accuracy=correct / n if n else None,
                    heidke=(correct - chance) / (n - chance) if n > chance else None)
    out = {**skill(m), "classes": CLASSES, "forecast_rows_observation_columns": m.tolist(),
           "both_wet": skill(m[1:, 1:]), "per_type": {}}
    for k, name in enumerate(CLASSES[1:], 1):
        a = int(m[k, k]); b = int(m[k].sum()) - a; c = int(m[:, k].sum()) - a
        out["per_type"][name] = contingency(a, b, c, int(m.sum()) - a - b - c)
    return out


def ptype_scores(pairs):
    m = np.zeros((5, 5), dtype=np.int64)
    missing = 0
    for fc, ob in pairs:
        f, o = category(fc), category(ob)
        if f is None or o is None:
            missing += 1
        else:
            m[f, o] += 1
    return {**ptype_table(m), "missing_pairs": missing}


def amount_scores(pairs, thresholds=(0.254, 2.54, 6.35)):
    """Paired window amounts in mm. Missing and negative reports are excluded, never made zero."""
    valid = [(f, o) for f, o in pairs if math.isfinite(f) and math.isfinite(o) and min(f, o) >= 0]
    f, o = np.asarray(valid, float).reshape(-1, 2).T
    out = ns.point_stats(f, o)
    out["units"] = "mm"
    out["thresholds"] = {}
    for t in thresholds:
        fy, oy = f >= t, o >= t
        out["thresholds"][str(t)] = contingency(int((fy & oy).sum()), int((fy & ~oy).sum()),
                                               int((~fy & oy).sum()), int((~fy & ~oy).sum()))
    return out


def erosion(series, fraction=0.5, sustain=2):
    """First sustained loss after an established cold pool, no interpolation across gaps.

    series is hourly (lead, fraction colder than the temperature threshold).
    Missing data before a resolved event makes the event indeterminate.
    """
    established = False
    warm = []
    for i, (h, value) in enumerate(series):
        if (i and h != series[i - 1][0] + 1) or value is None:
            return {"status": "missing_hours", "lead": None}
        if value >= fraction:
            established = True
            warm.clear()
        elif established:
            warm.append(h)
            if len(warm) >= sustain:
                return {"status": "resolved", "lead": warm[0]}
    return {"status": "right_censored" if established else "not_established", "lead": None}


def read_pairs(path, cycle, hours):
    rows = {}
    with Path(path).open(newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"cycle", "station", "lead", "latitude", "longitude", "obs_t2_f", "model_t2_f"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError(f"{path}: missing pair columns {sorted(required - set(reader.fieldnames or []))}")
        for raw in reader:
            if raw["cycle"] != cycle:
                raise ValueError(f"{path}: pair cycle {raw['cycle']} differs from {cycle}")
            lead = int(raw["lead"] or "")
            if not 0 <= lead <= hours:
                continue
            station = (raw["station"] or "").strip()
            if not station:
                raise ValueError(f"{path}: empty station")
            key = (lead, station)
            if key in rows:
                raise ValueError(f"{path}: duplicate station/hour {key}")
            r = {k: number(v) for k, v in raw.items() if k not in ("cycle", "station")}
            if not (-90 <= r["latitude"] <= 90 and -180 <= r["longitude"] <= 180):
                raise ValueError(f"{path}: invalid station location {key}")
            rows[key] = r
    if not rows:
        raise ValueError(f"{path}: no station pairs in requested window")
    return rows


def score_pairs(runs, hours, stations=None, cold_f=40., fraction=0.5, sustain=2):
    """Compare arms on common station/hour support per variable; retain coverage counts."""
    common = set.intersection(*(set(r) for r in runs.values()))
    if not common:
        raise ValueError("runs have no common station-hours")
    ref = next(iter(runs.values()))
    for key in common:
        for r in runs.values():
            for field in ("latitude", "longitude", "observation_epoch", "obs_t2_f", "obs_td2_f", "obs_type", "obs_precip_1h_mm"):
                x, y = ref[key].get(field, math.nan), r[key].get(field, math.nan)
                if (math.isfinite(x) != math.isfinite(y)) or (math.isfinite(x) and abs(x - y) > 1e-6):
                    raise ValueError(f"arms disagree on observation {key} {field}")
    wedge = {key for key in common if (key[1] in stations if stations is not None
             else in_wedge(ref[key]["latitude"], ref[key]["longitude"]))}
    by_hour = {h: [] for h in range(hours + 1)}
    wedge_by_hour = {h: [] for h in range(hours + 1)}
    for key in sorted(common):
        by_hour[key[0]].append(key)
        if key in wedge:
            wedge_by_hour[key[0]].append(key)
    # A fixed station panel over the entire window prevents outages changing the cold fraction.
    complete = {s for s in {station for _, station in wedge} if all((h, s) in wedge and all(
        math.isfinite(r[(h, s)].get(k, math.nan)) for r in runs.values()
        for k in ("model_t2_f", "obs_t2_f")) for h in range(hours + 1))}
    out = {}
    for name, rows in runs.items():
        hourly = {}
        for h in range(hours + 1):
            keys = wedge_by_hour[h]
            rec = {}
            for var in ("t2_f", "td2_f"):
                good = [k for k in keys if all(math.isfinite(r[k].get(col, math.nan)) for r in runs.values()
                        for col in ("model_" + var, "obs_" + var))]
                model = np.array([rows[k]["model_" + var] for k in good])
                obs = np.array([rows[k]["obs_" + var] for k in good])
                rec[var] = {**ns.point_stats(model, obs), "model_mean": float(model.mean()) if len(good) else None,
                            "obs_mean": float(obs.mean()) if len(good) else None}
            for prefix in ("model", "obs"):
                rec[prefix + "_cold_fraction"] = (sum(rows[(h, s)][prefix + "_t2_f"] < cold_f for s in complete)
                                                   / len(complete) if complete else None)
            pkeys = by_hour[h] if h > 0 else []
            # Same available category support in all arms.
            pkeys = [k for k in pkeys if all(category(r[k].get("model_type")) is not None for r in runs.values())]
            rec["ptype"] = ptype_scores([(rows[k].get("model_type"), rows[k].get("obs_type")) for k in pkeys])
            hourly[str(h)] = rec
        matrices = [hourly[str(h)]["ptype"]["forecast_rows_observation_columns"] for h in range(1, hours + 1)]
        pe = ptype_table(np.sum(matrices, axis=0))
        pe["unscored_common_station_hours"] = sum(h > 0 for h, _ in common) - pe["n"]
        events = {p: erosion([(h, hourly[str(h)][p + "_cold_fraction"]) for h in range(hours + 1)], fraction, sustain)
                  for p in ("model", "obs")}
        events["error_hours"] = (events["model"]["lead"] - events["obs"]["lead"]
                                 if all(events[p]["lead"] is not None for p in ("model", "obs")) else None)
        out[name] = {"hourly": hourly, "ptype": pe, "erosion": events,
                     "coverage": {"input_station_hours": len(rows), "common_station_hours": len(common),
                                  "wedge_station_hours": len(wedge), "erosion_stations": sorted(complete)},
                     "gaps": []}
        if pe["n"] == 0:
            out[name]["gaps"].append("No paired known precipitation types. Ptype scores are unavailable.")
        if not complete:
            out[name]["gaps"].append("No complete hourly station panel. Erosion timing is unavailable.")
        # FZRA p01i is liquid-equivalent in a reporting hour, not ice thickness.
        fp = []
        for key in sorted(common):
            h, _ = key
            r = rows[key]
            if h == 0 or category(r.get("obs_type")) != 3 or not all(
                    math.isfinite(arm[key].get("model_fzra_proxy_1h_mm", math.nan))
                    for arm in runs.values()):
                continue
            fp.append((r.get("model_fzra_proxy_1h_mm", math.nan), r.get("obs_precip_1h_mm", math.nan)))
        out[name]["fzra_reporting_hours"] = amount_scores(fp)
        out[name]["fzra_reporting_hours"]["method"] = "hourly model p1 where endpoint type is FZRA vs p01i in FZRA reports; conditional proxy"
    return out


def score_reports(path, rows, cycle, hours):
    """Exact hourly windows from optional reports/analysis station samples, no invented zeros."""
    pairs = {"snowfall": [], "freezing_rain_liquid": []}
    sources, seen = set(), set()
    with Path(path).open(newline="") as fh:
        for r in csv.DictReader(fh):
            if r["cycle"] != cycle:
                raise ValueError("amount report cycle differs from forecast")
            kind = r["kind"]
            if kind not in pairs or r["units"] != "mm" or not (r["source"] or "").strip():
                raise ValueError("reports require snowfall/freezing_rain_liquid, units mm and a source")
            start, end = int(r["start_lead"] or ""), int(r["end_lead"] or "")
            if not 0 <= start < end <= hours:
                raise ValueError("report window lies outside forecast")
            key = (kind, r["station"], start, end)
            if key in seen:
                raise ValueError(f"duplicate amount report {key}")
            seen.add(key)
            var = "model_snow_proxy_1h_mm" if kind == "snowfall" else "model_fzra_proxy_1h_mm"
            values = [rows.get((h, r["station"]), {}).get(var, math.nan) for h in range(start + 1, end + 1)]
            pairs[kind].append((sum(values), number(r["amount"])))
            sources.add(r["source"])
    return {"sources": sorted(sources), "method": "hourly endpoint-type/SR proxies; snowfall uses explicit 10:1 ratio",
            **{k: amount_scores(v, (2.54, 25.4, 50.8) if k == "snowfall" else (0.254, 2.54, 6.35)) for k, v in pairs.items()}}


def bridge_binary():
    p = os.environ.get("NAMGPU_WINTER_BRIDGE")
    if p:
        if not Path(p).is_file():
            raise ValueError(f"NAMGPU_WINTER_BRIDGE does not exist: {p}")
        return Path(p).resolve()
    p = shutil.which("rw_namwinter")
    if p:
        return Path(p)
    source = Path(__file__).with_name("winter_bridge.rs")
    raise ValueError(f"winter data path requires rw_namwinter: rustc --edition=2021 -O {source} -o rw_namwinter "
                     "(libnetcdf required); set NAMGPU_WINTER_BRIDGE to the resulting binary")


def sample_run(run, asos, cycle, hours, method, km, workers, dest):
    bridge = bridge_binary()
    base = datetime.strptime(cycle, "%Y%m%d%H")
    def one(h):
        v = base + timedelta(hours=h)
        frame = Path(run) / f"wrfout_d01_{v:%Y-%m-%d_%H:%M:%S}"
        prev = Path(run) / f"wrfout_d01_{v - timedelta(hours=1):%Y-%m-%d_%H:%M:%S}" if h else "-"
        command = [str(bridge), "sample", str(frame), str(prev), str(asos), f"{v:%Y-%m-%dT%H:%M:%SZ}", str(h), method, str(km)]
        t0 = time.monotonic()
        r = subprocess.run(command, text=True, capture_output=True)
        if r.returncode:
            raise ValueError(r.stderr.strip())
        return list(csv.DictReader(io.StringIO(r.stdout))), {"lead": h, "command": command, "seconds": time.monotonic() - t0}
    timing = []
    with dest.open("w", newline="") as fh, ThreadPoolExecutor(workers) as ex:
        writer = None
        for records, elapsed in ex.map(one, range(hours + 1)):
            timing.append(elapsed)
            for r in records:
                r["cycle"] = cycle
                if writer is None:
                    writer = csv.DictWriter(fh, fieldnames=list(r)); writer.writeheader()
                writer.writerow(r)
    if writer is None:
        raise ValueError("no ASOS reports within 10 minutes and the station radius")
    return timing


def score_md(record):
    lines = [f"# Winter/CAD scores, {record['cycle']}", "",
             "CAD subset: explicit station list or Southeast Piedmont polygon (see JSON criteria). Temperatures in F.",
             "Erosion is a thermal cold-pool proxy on a fixed complete station panel, not a synoptic CAD diagnosis.", "",
             "| source | paired ptype n | HSS incl. dry | rain hit rate | snow | freezing rain | sleet | model/obs erosion hour |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, r in record["runs"].items():
        p, e = r["ptype"], r["erosion"]
        rates = " | ".join(str(p["per_type"][k]["hit_rate"]) for k in CLASSES[1:])
        events = " / ".join(str(e[k]["lead"]) if e[k]["lead"] is not None else e[k]["status"] for k in ("model", "obs"))
        lines.append(f"| {name} | {p['n']} | {p['heidke']} | {rates} | {events} |")
    lines += ["", "| lead | source | T2 bias/MAE/RMSE (n) | Td2 bias/MAE/RMSE (n) | cold fraction model/obs |",
              "|---|---|---|---|---|"]
    for h in range(record["hours"] + 1):
        for name, r in record["runs"].items():
            x = r["hourly"][str(h)]
            stats = [ns.fmt({"x": x[k]}, "x") for k in ("t2_f", "td2_f")]
            lines.append(f"| {h} | {name} | {' | '.join(stats)} | {x['model_cold_fraction']} / {x['obs_cold_fraction']} |")
    for name, r in record["runs"].items():
        lines += ["", f"{name}: " + "; ".join(r["gaps"] or ["No station coverage gaps flagged."]),
                  f"FZRA reporting-hour liquid proxy: {json.dumps(r['fzra_reporting_hours'])}",
                  f"Report/analysis amounts: {json.dumps(r['amounts'])}"]
    return "\n".join(lines) + "\n"


def draw_sheet(record, path):
    """Statistical time-series sheet. No geographic/weather-field maps are drawn here."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1200, 880), "white"); d = ImageDraw.Draw(im)
    d.text((25, 15), f"Winter/CAD station time series | {record['cycle']} | matched observations and forecast arms", fill="black")
    colors = ["#1967d2", "#ce401f", "#008577", "#9546a6"]
    runs = list(record["runs"].items())
    for j, (name, _) in enumerate(runs):
        d.text((30 + j * 240, 40), name, fill=colors[j % len(colors)])
    d.text((1000, 40), "observations (black)", fill="black")
    for panel, (key, title) in enumerate((("t2_f", "Wedge mean T2 (F)"), ("td2_f", "Wedge mean Td2 (F)"),
                                         ("cold_fraction", "Fixed-panel fraction colder than threshold"))):
        top = 90 + panel * 255
        curves = []
        for j, (name, r) in enumerate(runs):
            values = [r["hourly"][str(h)]["model_cold_fraction"] if key == "cold_fraction"
                      else r["hourly"][str(h)][key]["model_mean"] for h in range(record["hours"] + 1)]
            curves.append((colors[j % len(colors)], values))
        r = runs[0][1]
        obs = [r["hourly"][str(h)]["obs_cold_fraction"] if key == "cold_fraction"
               else r["hourly"][str(h)][key]["obs_mean"] for h in range(record["hours"] + 1)]
        curves.append(("black", obs))
        vals = [v for _, vv in curves for v in vv if v is not None]
        lo, hi = (0, 1) if key == "cold_fraction" else ((math.floor(min(vals) - 1), math.ceil(max(vals) + 1)) if vals else (0, 1))
        d.text((25, top - 20), title, fill="black")
        for k in range(5):
            y = top + 180 - k * 45
            d.line((70, y, 1160, y), fill="#dddddd")
            d.text((25, y - 5), f"{lo + (hi-lo)*k/4:.1f}", fill="black")
        for col, values in curves:
            last = None
            for h, v in enumerate(values):
                if v is None:
                    last = None; continue
                pt = (70 + h * 1090 / max(1, record["hours"]), top + 180 - 180 * (v - lo) / (hi - lo))
                if last is not None:
                    d.line((last, pt), fill=col, width=3)
                d.ellipse((pt[0]-2, pt[1]-2, pt[0]+2, pt[1]+2), fill=col)
                last = pt
        for h in range(0, record["hours"] + 1, max(1, record["hours"] // 12)):
            d.text((66 + h * 1090 / max(1, record["hours"]), top + 190), str(h), fill="black")
        d.text((540, top + 210), "forecast lead (hours)", fill="black")
    im.save(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cycle", required=True)
    ap.add_argument("--hours", type=int, default=12)
    class Source(argparse.Action):
        def __call__(self, parser, namespace, value, option_string=None):
            namespace.sources.append((self.dest, value))
    ap.set_defaults(sources=[])
    ap.add_argument("--run", action=Source, help="NAME=hourly history directory; list control first")
    ap.add_argument("--pairs", action=Source, help="NAME=retained station CSV; see docs/winter-score.md")
    ap.add_argument("--asos", type=Path, help="rw_asos/IEM CSV with wxcodes; required for history runs")
    ap.add_argument("--reports", type=Path, help="optional matched snowfall/freezing-rain amount report CSV")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ptype-method", choices=("explicit-sr", "bourgouin", "stored"), default="explicit-sr")
    ap.add_argument("--stations", help="explicit comma-separated CAD station subset, otherwise the Piedmont polygon")
    ap.add_argument("--cold-f", type=float, default=40.)
    ap.add_argument("--cold-fraction", type=float, default=0.5)
    ap.add_argument("--sustain-hours", type=int, default=2)
    ap.add_argument("--max-km", type=float, default=15.)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-sheets", action="store_true")
    a = ap.parse_args(argv)
    try:
        datetime.strptime(a.cycle, "%Y%m%d%H")
        if not 1 <= a.hours <= 384 or not 1 <= a.workers <= 4 or not 1 <= a.sustain_hours <= a.hours:
            raise ValueError("hours must be 1..384, workers 1..4, sustain-hours 1..hours")
        if not 0 < a.cold_fraction <= 1 or not math.isfinite(a.cold_f) or not 0 < a.max_km <= 100:
            raise ValueError("invalid temperature/fraction/radius threshold")
        if not a.sources:
            raise ValueError("at least one --run or --pairs is required")
        if any(kind == "run" for kind, _ in a.sources) and not a.asos:
            raise ValueError("--run requires --asos (cached rw_asos/IEM CSV)")
        a.out.mkdir(parents=True, exist_ok=True)
        t0 = time.monotonic()
        runs, provenance, stages = {}, {}, []
        for kind, spec in a.sources:
            name, source = spec.split("=", 1)
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) or name in runs:
                raise ValueError("source names must be unique simple labels")
            path = Path(source)
            if kind == "run":
                path = a.out / (name + "-pairs.csv")
                if path.exists():
                    raise ValueError(f"refusing to overwrite station receipt {path}")
                stages += sample_run(source, a.asos, a.cycle, a.hours, a.ptype_method, a.max_km, a.workers, path)
            runs[name] = read_pairs(path, a.cycle, a.hours)
            provenance[name] = {"source": source, "kind": kind, "pairs": str(path.resolve()),
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            if kind == "run":
                provenance[name]["asos_sha256"] = hashlib.sha256(a.asos.read_bytes()).hexdigest()
                provenance[name]["bridge_sha256"] = hashlib.sha256(bridge_binary().read_bytes()).hexdigest()
        stations = set(a.stations.split(",")) if a.stations else None
        scored = score_pairs(runs, a.hours, stations, a.cold_f, a.cold_fraction, a.sustain_hours)
        for name, r in scored.items():
            r["amounts"] = score_reports(a.reports, runs[name], a.cycle, a.hours) if a.reports else {"status": "unavailable", "reason": "no matched snowfall/freezing-rain reports or analyses supplied"}
        record = {"schema": "woof-nam.winter-score.v1", "cycle": a.cycle, "hours": a.hours,
                  "criteria": {"cold_f": a.cold_f, "cold_fraction": a.cold_fraction, "sustain_hours": a.sustain_hours,
                               "stations": sorted(stations) if stations else None, "piedmont_polygon": PIEDMONT,
                               "ptype_method": a.ptype_method, "max_km": a.max_km},
                  "provenance": provenance, "runs": scored}
        if a.reports:
            record["report_provenance"] = {"path": str(a.reports.resolve()),
                                           "sha256": hashlib.sha256(a.reports.read_bytes()).hexdigest()}
        (a.out / "winter_scores.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
        (a.out / "WINTER.md").write_text(score_md(record))
        if not a.no_sheets:
            draw_sheet(record, a.out / "cad-timeseries.png")
        parallelism = (f"Rust data readers, up to {a.workers} processes; small receipt statistics on host" if stages
                       else "retained small receipts and statistics on one host thread; no raw data reader or forecast")
        stages.append({"stage": "total", "seconds": time.monotonic() - t0, "parallelism": parallelism})
        (a.out / "timing.json").write_text(json.dumps(stages, indent=2) + "\n")
        print(f"winter scores: {a.out / 'WINTER.md'}")
        return 0
    except (ValueError, OSError, KeyError) as e:
        print(f"DEFECT winter scorer: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
