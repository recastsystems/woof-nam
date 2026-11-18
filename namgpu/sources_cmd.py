"""woof-nam sources: every source a run can start from, as data (the site's table is checked against this list).

    woof-nam sources            one line per source
    woof-nam sources --json     the same as JSON (the WOOF site keeps a copy in its tests, so the two cannot drift)

Each row: what it starts from (forecast-hour: a start can be any posted hour of a run; analysis: its analyses only;
parent: our own run), its archive (valid_from, valid_before), the cycle and analysis spacing, the forecast hours it
posts ([[last hour, step], ...]), the boundary spacing, reach_h (a source whose edges are its own forecast: the
longest run from its analysis, the last lead the fetch reads), and its credit. Read from namgpu/init/sources.py SOURCES.
"""
from __future__ import annotations

import argparse
import json
import sys


def table() -> list[dict]:
    from namgpu.init import sources as S
    from namgpu.init import window
    out = []
    for name, row in S.SOURCES.items():
        if name == "parent":
            starts = "parent"
        elif name in window.FORECAST_STARTS:
            starts = "forecast-hour"
        else:
            starts = "analysis"
        posted = (S.lead_steps(row) if name == "gfs" else row.get("posted_leads")) if starts == "forecast-hour" else None
        edges = None if name == "parent" else int((row.get("boundary") or {}).get("interval_h")
                                                  or row.get("boco_interval_h") or 3)
        out.append({"name": name, "starts": starts, "valid_from": row.get("valid_from"),
                    "valid_before": row.get("valid_before"), "retention_h": row.get("retention_h"),
                    "analysis_every_h": row.get("analysis_every_h"),
                    "cycle_step_h": int(row.get("cycle_step_h", 6)) if starts == "forecast-hour" else None,
                    "posted": posted, "posted_by_cycle": row.get("posted_by_cycle"),
                    "edges_every_h": edges, "edges_from": ("gfs" if ("boundary" not in row and name != "parent")
                                                           else name),
                    # the longest run from a source whose edges are its own forecast: the last lead the fetch reads
                    # (posted_by_cycle is what the source posts; a run from it reads only to reach_h)
                    "reach_h": S.reach_h(name),
                    "members": row.get("members"), "attribution": row.get("attribution"),
                    "licence": row.get("licence")})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="woof-nam sources", description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="print the table as JSON")
    a = ap.parse_args(argv)
    rows = table()
    if a.json:
        json.dump(rows, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0
    for r in rows:
        span = f"{(r['valid_from'] or 'any')[:13]} to {(r['valid_before'] or 'now')[:13]}"
        what = (f"forecast hours {r['posted']}" if r["posted"] else
                f"analyses every {r['analysis_every_h']} h" if r["analysis_every_h"] else r["starts"])
        reach = f", to {r['reach_h']} h" if r.get("reach_h") else ""
        print(f"{r['name']:<9} {span:<32} {what}; edges every {r['edges_every_h']} h from {r['edges_from']}{reach}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
