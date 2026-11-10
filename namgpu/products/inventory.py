"""Account for every archived GRIB2 record, including records with no mapped source.

Run ``woof-nam inventory ARCHIVE OURS --set awphys --json receipt.json``.
Exit 1 for an unexplained omission, an extra record, or a changed baseline message.
An omission is explained only by an unmapped row with a nonempty release note.
Mapped-but-unwritten records always fail. The archive, never the table, defines
the records to check. Dates and packing may differ between NOAA and our forecast;
parameters, both surfaces, statistics, time windows and multiplicity must agree.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import struct

from namgpu.products import spec


def _signed(value, bits):
    sign = 1 << (bits - 1)
    return -(value & (sign - 1)) if value & sign else value


def _surface(section, at):
    typ, scale = section[at:at + 2]
    value = int.from_bytes(section[at + 2:at + 6], "big")
    if typ == 255:
        return None, None
    # Valueless surfaces are conventionally encoded with either zeros or missing.
    value = 0.0 if scale == 255 or value == 0xffffffff else value * 10.0 ** -_signed(scale, 8)
    return typ, value


def _seconds(unit, value):
    factors = {0: 60, 1: 3600, 2: 86400, 10: 10800, 11: 21600, 12: 43200, 13: 1}
    if unit not in factors:
        raise ValueError(f"unsupported GRIB time unit {unit}")
    return factors[unit] * value


def scan(path, hashes=False):
    """Read only section headers, seeking over packed values. No decoder or GPU.

    Optional SHA-256 covers the complete original message, including both fields
    of a U/V pair. Thus a baseline check requires unchanged encoded bytes, not
    merely equal decoded values. Memory is bounded by one message.
    """
    path = Path(path)
    size = path.stat().st_size
    out = []
    with path.open("rb") as f:
        msg = 0
        while f.tell() < size:
            offset = f.tell()
            header = f.read(16)
            if len(header) != 16 or header[:4] != b"GRIB" or header[7] != 2:
                raise ValueError(f"{path}:{offset}: expected GRIB edition 2 header")
            total = int.from_bytes(header[8:16], "big")
            end = offset + total
            if total < 20 or end > size:
                raise ValueError(f"{path}:{offset}: invalid/truncated message length {total}")
            msg += 1
            fields = []
            current = None
            while f.tell() < end - 4:
                pos = f.tell()
                h = f.read(5)
                if len(h) != 5:
                    raise ValueError(f"{path}:{pos}: truncated section header")
                length, number = struct.unpack(">IB", h)
                if length < 5 or pos + length > end - 4:
                    raise ValueError(f"{path}:{pos}: invalid section length {length}")
                if number == 4:
                    if current is not None:
                        raise ValueError(f"{path}:{pos}: product section before previous field's data")
                    s = h + f.read(length - 5)
                    if len(s) < 34:
                        raise ValueError(f"{path}:{pos}: truncated product section")
                    pdt = int.from_bytes(s[7:9], "big")
                    if pdt not in (0, 8):
                        raise ValueError(f"{path}:{pos}: unsupported product template 4.{pdt}")
                    t1, v1 = _surface(s, 22)
                    t2, v2 = _surface(s, 28)
                    start = _seconds(s[17], int.from_bytes(s[18:22], "big"))
                    stat, duration = None, 0
                    if pdt == 8:
                        if len(s) < 58 or s[41] != 1:
                            raise ValueError(f"{path}:{pos}: expected one complete statistical time range")
                        stat = s[46]
                        duration = _seconds(s[48], int.from_bytes(s[49:53], "big"))
                    current = {"dcp": f"{header[6]}.{s[9]}.{s[10]}", "t1": t1, "v1": v1,
                               "t2": t2, "v2": v2, "pdt": pdt, "stat": stat,
                               "start_seconds": start, "duration_seconds": duration}
                else:
                    f.seek(length - 5, 1)
                if number == 7:
                    if current is None:
                        raise ValueError(f"{path}:{pos}: data section without product section")
                    fields.append(dict(current, offset=offset, bytes=total))
                    current = None
            if f.read(4) != b"7777" or not fields or current is not None:
                raise ValueError(f"{path}:{offset}: incomplete GRIB message")
            digest = None
            if hashes:
                f.seek(offset)
                digest = hashlib.sha256(f.read(total)).hexdigest()
                f.seek(end)
            for sub, record in enumerate(fields, 1):
                record["rec"] = f"{msg}.{sub}" if len(fields) > 1 else str(msg)
                if hashes:
                    record["message_sha256"] = digest
                out.append(record)
    if not out:
        raise ValueError(f"{path}: empty inventory")
    return out


def table_key(r):
    return (r["dcp"], r["t1"], r["v1"], r["t2"], r["v2"], r["stat"])


def identity(r):
    return table_key(r) + (r["pdt"], r["start_seconds"], r["duration_seconds"])


def compare(reference, ours, rows):
    notes = defaultdict(list)
    for r in rows:
        key = spec.inventory_key(r)
        notes[key].append(r)
    available = defaultdict(deque)
    for r in ours:
        available[identity(r)].append(r)
    records = []
    for r in reference:
        candidates = notes.get(table_key(r), [])
        # Duplicate parameter/level rows may represent different accumulation
        # windows. Never excuse a missing mapped row using its unmapped sibling.
        omitted = candidates and all(x["source"] == "-" and x["note"].strip() for x in candidates)
        matched = available[identity(r)].popleft() if available[identity(r)] else None
        status = "written" if matched else "left_out" if omitted else "unexplained"
        reason = "" if matched else "; ".join(dict.fromkeys(x["note"] for x in candidates))
        if not matched and not omitted:
            reason = ("mapped record absent from output" if candidates else "record absent from package table")
        records.append({**r, "status": status, "ours_rec": matched["rec"] if matched else None,
                        "name": candidates[0].get("noaa_name", candidates[0]["name"]) if candidates else r["dcp"],
                        "level": candidates[0].get("noaa_level", candidates[0]["level"]) if candidates else
                                 f"{r['t1']}:{r['v1']} / {r['t2']}:{r['v2']}",
                        "table_lines": [x["line"] for x in candidates], "reason": reason})
    extra = [r for group in available.values() for r in group]
    counts = Counter(r["status"] for r in records)
    return {"reference_records": len(reference), "ours_records": len(ours),
            "written": counts["written"], "left_out": counts["left_out"],
            "unexplained": counts["unexplained"], "extra": extra, "records": records,
            "ok": not counts["unexplained"] and not extra}


def preserved(baseline, ours):
    """Each old record must remain in order with exactly the same message bytes."""
    remaining = iter(ours)
    changed = []
    for old in baseline:
        match = next((r for r in remaining if identity(r) == identity(old)), None)
        if match is None or match["message_sha256"] != old["message_sha256"]:
            changed.append(old["rec"])
    return {"baseline_records": len(baseline), "changed_or_missing": changed, "ok": not changed}


def markdown(report):
    lines = [f"{report['reference_records']} archived records; {report['written']} written; "
             f"{report['left_out']} named omissions; {report['unexplained']} unexplained; "
             f"{len(report['extra'])} extra records.", "",
             "| Archive record | Name | Level | Start/duration (s) | Status | Reason |",
             "|---|---|---|---|---|---|"]
    for r in report["records"]:
        cells = (r["rec"], r["name"], r["level"], f"{r['start_seconds']}/{r['duration_seconds']}",
                 r["status"], r["reason"])
        lines.append("| " + " | ".join(str(c).replace("|", "/") for c in cells) + " |")
    if report["extra"]:
        lines += ["", "Extra output records:", ""]
        lines += [f"- {r['rec']}: {identity(r)}" for r in report["extra"]]
    if "preservation" in report:
        p = report["preservation"]
        lines += ["", f"Baseline: {p['baseline_records']} records; "
                  f"{len(p['changed_or_missing'])} changed or missing."]
        if p["changed_or_missing"]:
            lines += ["Changed or missing baseline records: " + ", ".join(p["changed_or_missing"])]
    return "\n".join(lines) + "\n"


def main(argv=None):
    from namgpu.products import sets
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("archive", type=Path)
    ap.add_argument("ours", type=Path)
    ap.add_argument("--set", required=True, choices=tuple(sets.SETS))
    ap.add_argument("--baseline", type=Path, help="require every existing GRIB message to remain byte-identical")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--md", type=Path)
    a = ap.parse_args(argv)
    try:
        ps = sets.get(a.set)
        rows = spec.attach_noaa(spec.read(sets.table_path(ps)), noaa_path=sets.noaa_path(ps))
        ours = scan(a.ours, hashes=a.baseline is not None)
        report = compare(scan(a.archive), ours, rows)
        report.update(archive=str(a.archive), ours=str(a.ours), table=str(sets.table_path(ps)), product_set=a.set)
        if a.baseline:
            report["preservation"] = preserved(scan(a.baseline, hashes=True), ours)
            report["ok"] &= report["preservation"]["ok"]
        if a.json:
            a.json.write_text(json.dumps(report, indent=2) + "\n")
        if a.md:
            a.md.write_text(markdown(report))
        print(markdown(report).splitlines()[0])
        if a.baseline:
            p = report["preservation"]
            print(f"Baseline: {p['baseline_records']} records; {len(p['changed_or_missing'])} changed or missing.")
        return 0 if report["ok"] else 1
    except (ValueError, OSError) as exc:
        print(f"DEFECT inventory: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
