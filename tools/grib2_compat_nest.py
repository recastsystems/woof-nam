"""Record-by-record compatibility of a WOOF-NAM nest file against NOAA's NAM nest file of the same nest and hour.

NOAA's NAM nests wrote `nam.tHHz.<nest>.hiresfFF.tm00.grib2` (conusnest, alaskanest, hawaiinest, priconest,
firewxnest); WOOF-NAM writes `woof-nam.tHHz.<nest>.hiresfFF.tm00.grib2`. For each of NOAA's records, in NOAA's
order, this says whether ours carries it, under the same record number (wgrib2's `N`, or `N.M` for the two fields
of one U/V message), with the same inventory text, and with the same GRIB2 keys a decoder selects on (name, level,
time range, units, codes, bitmap), and with the same packing template (its own column: one setting per file
family, not a property of the record). The grid section is compared once. Precision is reported
apart: NOAA's own nests pick the decimal and binary scale (D, E) per file from the data (the same record has different
D and E in NOAA's CONUS and Alaska files of one hour), so "precision" says whether our packing step 2^E / 10^D is the
same as, finer than or coarser than NOAA's in that file. Values are not compared.

    python -I tools/grib2_compat_nest.py NOAA.grib2 [OURS.grib2] [--md OUT.md] [--tsv OUT.tsv] [--plain]

With NOAA's file alone it lists NOAA's records (the target). The .idx beside each file (`<file>.idx`, what
`wgrib2 -s` prints) is read when present: record numbers and inventory text are compared from it, which is what a
converter keyed on NOAA's index sees. Needs eccodes (pip install eccodes).
"""
from __future__ import annotations

import argparse
import collections
import functools
import sys
from pathlib import Path

import eccodes as ec

KEYS = ("shortName", "typeOfLevel", "level", "stepType", "stepRange", "units", "discipline", "parameterCategory",
        "parameterNumber", "typeOfFirstFixedSurface", "scaleFactorOfFirstFixedSurface",
        "scaledValueOfFirstFixedSurface", "typeOfSecondFixedSurface", "scaleFactorOfSecondFixedSurface",
        "scaledValueOfSecondFixedSurface", "productDefinitionTemplateNumber", "typeOfStatisticalProcessing",
        "lengthOfTimeRange",
        "bitmapPresent", "centre", "subCentre", "tablesVersion", "localTablesVersion", "typeOfGeneratingProcess")
GRID = ("gridDefinitionTemplateNumber", "gridType", "shapeOfTheEarth", "Nx", "Ny", "Ni", "Nj",
        "latitudeOfFirstGridPointInDegrees", "longitudeOfFirstGridPointInDegrees", "DxInMetres", "DyInMetres",
        "LoVInDegrees", "LaDInDegrees", "Latin1InDegrees", "Latin2InDegrees", "projectionCentreFlag",
        "resolutionAndComponentFlags", "scanningMode", "latitudeOfLastGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees")
EXTRA = ("dataRepresentationTemplateNumber", "generatingProcessIdentifier", "offset", "numberOfValues", "numberOfMissing", "decimalScaleFactor",
         "binaryScaleFactor", "bitsPerValue")
LABEL = {"decimalScaleFactor": "D", "binaryScaleFactor": "E", "dataRepresentationTemplateNumber": "packing template",
         "bitmapPresent": "bitmap"}
STAT = {0: "ave", 1: "acc", 2: "max", 3: "min"}


@functools.lru_cache(maxsize=4)
def fields(path: Path) -> list[dict]:
    """Every field of the file in order, with wgrib2's record number (`N`, or `N.M` inside a multi-field message)."""
    ec.codes_grib_multi_support_on()
    out = []
    with open(path, "rb") as f:
        while True:
            h = ec.codes_grib_new_from_file(f)
            if h is None:
                break
            d = {}
            for k in KEYS + GRID + EXTRA:
                try:
                    d[k] = ec.codes_get(h, k)
                except Exception:  # noqa: BLE001
                    d[k] = None
            ec.codes_release(h)
            out.append(d)
    ec.codes_grib_multi_support_off()
    per_msg = collections.Counter(d["offset"] for d in out)
    msg, last, sub = 0, None, 0
    for d in out:
        if d["offset"] != last:
            msg, last, sub = msg + 1, d["offset"], 0
        sub += 1
        d["rec"] = f"{msg}.{sub}" if per_msg[d["offset"]] > 1 else str(msg)
        d["messages"] = msg
    return out


def idx(path: Path) -> dict[str, str]:
    """rec -> the inventory text after the date (`VAR:level:time:`), from `<file>.idx` if it exists."""
    p = Path(str(path) + ".idx")
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        parts = line.split(":")
        if len(parts) >= 4:
            out[parts[0]] = ":".join(parts[3:])
    return out


def _real(d: dict, which: str):
    sf, sv = d.get(f"scaleFactorOf{which}FixedSurface"), d.get(f"scaledValueOf{which}FixedSurface")
    if sv is None or sf is None or sv in (2147483647, 4294967295) or sf in (127, 255, -127):
        return 0.0
    return round(sv / 10.0 ** sf, 6)


def ident(d: dict) -> tuple:
    """The record's identity as wgrib2 prints it: parameter, both surfaces, template, statistic, time range."""
    t2 = d["typeOfSecondFixedSurface"]
    t2 = None if t2 in (255, "255", None) else t2
    return (d["discipline"], d["parameterCategory"], d["parameterNumber"], d["typeOfFirstFixedSurface"],
            _real(d, "First"), t2, None if t2 is None else _real(d, "Second"),
            d["productDefinitionTemplateNumber"], d["typeOfStatisticalProcessing"], d["stepRange"])


def name(d: dict, inv: dict) -> str:
    """NOAA's inventory text for the record (`VAR level, time`), else a decoded label."""
    text = inv.get(d["rec"])
    if text:
        var, _, rest = text.rstrip(":").partition(":")
        level, _, when = rest.partition(":")
        return f"{var} {level}, {when}" if when else f"{var} {level}"
    s = f"{d['shortName']} {d['typeOfLevel']} {d['level']}"
    if d["productDefinitionTemplateNumber"] == 8:
        s += f" ({d['stepRange']} h {STAT.get(d['typeOfStatisticalProcessing'], 'stat')})"
    return s


def step(d: dict):
    """The packing quantum 2^E / 10^D (the value step a decoder sees), or None."""
    if d["decimalScaleFactor"] is None or d["binaryScaleFactor"] is None:
        return None
    return 2.0 ** d["binaryScaleFactor"] / 10.0 ** d["decimalScaleFactor"]


def precision(d: dict, b: dict) -> str:
    a, c = step(d), step(b)
    if a is None or c is None:
        return "-"
    if b.get("bitsPerValue") == 0:
        return "constant"                              # a constant field: no packing step to compare
    if (d["decimalScaleFactor"], d["binaryScaleFactor"]) == (b["decimalScaleFactor"], b["binaryScaleFactor"]):
        return "same"
    return "finer" if c < a else ("same" if c == a else "coarser")


def grid_text(d: dict) -> str:
    g = {k: d[k] for k in GRID if d[k] not in (None, "MISSING")}
    return ", ".join(f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}" for k, v in g.items())


def compare(noaa: Path, ours: Path | None, plain: bool = False):
    """(header, rows, summary lines) for NOAA's file against ours (or NOAA's records alone when ours is None)."""
    ref = fields(noaa)
    ref_inv = idx(noaa)
    mine = fields(ours) if ours else []
    mine_inv = idx(ours) if ours else {}
    mine_at = collections.defaultdict(list)
    for i, d in enumerate(mine):
        d["_i"] = i
        mine_at[ident(d)].append(d)
    order = []
    seen = collections.Counter()
    counts = collections.Counter()
    rows = []
    for d in ref:
        k = ident(d)
        n = seen[k]
        seen[k] += 1
        label = name(d, ref_inv)
        if ours is None:
            rows.append((d["rec"], label, str(d["dataRepresentationTemplateNumber"])))
            continue
        b = mine_at[k][n] if n < len(mine_at[k]) else None
        if b is None:
            rows.append((d["rec"], label, "no", "-", "-", "-", "-",
                         "not written yet" if plain else "no record with this identity"))
            counts["missing"] += 1
            continue
        diff = [f"{LABEL.get(x, x)} NOAA {d[x]} / ours {b[x]}" for x in KEYS if d[x] != b[x]]
        if ref_inv and mine_inv and ref_inv.get(d["rec"]) != mine_inv.get(b["rec"]):
            diff.append(f"inventory text NOAA '{ref_inv.get(d['rec'])}' / ours '{mine_inv.get(b['rec'])}'")
        same_num = b["rec"] == d["rec"]
        counts["same keys" if not diff else "differ"] += 1
        counts["same number" if same_num else "renumbered"] += 1
        if not diff and same_num:
            counts["identical"] += 1
        order.append(b["_i"])
        pr = precision(d, b)
        counts["precision " + pr] += 1
        pk_a, pk_b = d["dataRepresentationTemplateNumber"], b["dataRepresentationTemplateNumber"]
        pack = "same" if pk_a == pk_b else f"NOAA 5.{pk_a} / ours 5.{pk_b}"
        counts["packing same" if pk_a == pk_b else "packing differs"] += 1
        notes = list(diff)
        if pr == "coarser":
            notes.append(f"step NOAA 2^{d['binaryScaleFactor']}/10^{d['decimalScaleFactor']} / ours "
                         f"2^{b['binaryScaleFactor']}/10^{b['decimalScaleFactor']}")
        rows.append((d["rec"], label, "yes", "yes" if not diff else "no",
                     "yes" if same_num else f"no: ours {b['rec']}", pack, pr, "; ".join(notes)))
    summary = [f"NOAA: {len(ref)} fields in {ref[-1]['messages'] if ref else 0} messages; "
               f"grid: {grid_text(ref[0]) if ref else '-'}"]
    if ours is None:
        return ("rec", "NOAA record", "packing template"), rows, summary
    ref_ids = collections.Counter(ident(d) for d in ref)
    got = collections.Counter()
    extra = []
    for b in mine:
        got[ident(b)] += 1
        if got[ident(b)] > ref_ids[ident(b)]:
            extra.append((b["rec"], name(b, mine_inv)))
    grid_diff = [f"{k}: NOAA {ref[0][k]} / ours {mine[0][k]}" for k in GRID if mine and ref[0][k] != mine[0][k]]
    summary += [
        f"ours: {len(mine)} fields in {mine[-1]['messages'] if mine else 0} messages",
        f"{len(rows)} NOAA records: {counts['same keys']} present with NOAA's inventory text and keys, "
        f"{counts['differ']} present with a difference, {counts['missing']} not in ours; "
        f"{len(extra)} of ours not in NOAA's",
        f"record numbers: {counts['same number']} of the present records have NOAA's number, {counts['renumbered']} "
        f"another; ours are in NOAA's order: {'yes' if order == sorted(order) else 'no'}",
        f"identical (number, text and keys): {counts['identical']}",
        f"packing template: {counts['packing same']} as NOAA's, {counts['packing differs']} another",
        f"precision against NOAA's file: {counts['precision same']} same, {counts['precision finer']} finer, "
        f"{counts['precision coarser']} coarser, {counts['precision constant']} constant fields",
        "grid section: identical" if not grid_diff else "grid section differs: " + "; ".join(grid_diff),
        "generating process: NOAA "
        + ", ".join(f"{k} x{v}" for k, v in collections.Counter(d["generatingProcessIdentifier"] for d in ref).items())
        + " / ours "
        + ", ".join(f"{k} x{v}" for k, v in collections.Counter(b["generatingProcessIdentifier"] for b in mine).items()),
    ] + [f"extra in ours: {r} {t}" for r, t in extra]
    return (("rec", "NOAA record", "present", "NOAA's text and keys", "NOAA's record number", "packing", "precision",
             "notes"), rows, summary)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("noaa", type=Path)
    ap.add_argument("ours", type=Path, nargs="?")
    ap.add_argument("--md", type=Path)
    ap.add_argument("--tsv", type=Path)
    ap.add_argument("--plain", action="store_true", help="notes in plain words for the package docs")
    a = ap.parse_args(argv)
    head, rows, summary = compare(a.noaa, a.ours, a.plain)
    if a.tsv:
        a.tsv.write_text("\n".join("\t".join(map(str, x)) for x in [head] + rows) + "\n")
    if a.md:
        md = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        md += ["| " + " | ".join(str(c).replace("|", "/") for c in x) + " |" for x in rows]
        a.md.write_text("\n".join(md) + "\n")
    print("\n".join(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
