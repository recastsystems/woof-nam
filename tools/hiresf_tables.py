"""Build the NAM nest (hiresf) record tables from NOAA's own files, and NOAA's facts per record for each nest.

    python tools/hiresf_tables.py <folder with nam.tHHz.<nest>.hiresf01.tm00.grib2 (+ .idx) for the five nests>

Writes namgpu/products/hiresf-conusnest.tsv (the CONUS nest's record list: 876 records at f01-f60) and hiresf.tsv
(Alaska, Hawaii, Puerto Rico and fire weather: one record list, 871 records, checked identical across the four),
hiresf-firewxnest.tsv (that list with the fire-weather nest's 1 h bucket, below), and hiresf-<nest>-noaa.tsv for all
five (NOAA's .idx text and per-record precision, spec.noaa_table). f00 is the same
list without the hour's maxima and minima, which the writer leaves out at the initial time as NOAA's f00 does.

A row's source (spec.py documents the columns) comes from, in order: the rules below (NCEP's nest windows and the
nest-only records), then the awphys table's row of the same parameter, surfaces and statistic. The nest windows are
NCEP's: accumulations and averages over the 3 h bucket (since:3h: 0-1, 0-2, 0-3, 3-4, ...), maxima and minima over
the last hour (the writer's rule). A row with no source is left out of every file, named in the run index.
The fire-weather nest empties its accumulators every hour (nam.v4.2.11 parm/nam_configfile_firewxnest_{conus,alaska}
nhrs_prec/heat/clod/rdlw/rdsw/srfc 1, where the other nests have 3), so NOAA's firewx files carry 1-2, 2-3, ..., 11-12
windows (2026-11-03 06Z f02-f36 .idx): its table is the shared list with since:1h and the 1 h carriers.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from namgpu.io import products as P  # noqa: E402
from namgpu.products import spec  # noqa: E402

HERE = Path(__file__).resolve().parents[1] / "namgpu" / "products"
NESTS = ("conusnest", "alaskanest", "hawaiinest", "priconest", "firewxnest")


def lt(r):
    return int(r["level"].split(":")[0])


# (NOAA name, first level type or None, statistic or None) -> (source, window, note); "-" = left out, with why
RULES = {
    # ---- 3 h bucket accumulations (NMMB's 3 h accumulators, or the history's run totals differenced)
    ("APCP", 1, "acc"): ("carrier:APCP_S3", "since:3h", "RAINNC + RAINC over NCEP's 3 h bucket, budget interpolation"),
    ("NCPCP", 1, "acc"): ("carrier:NCPCP_S3", "since:3h", "grid-scale precipitation over the 3 h bucket"),
    ("WEASD", 1, "acc"): ("carrier:NAMH_WEASD_ACC", "since:3h", "ACSNOW * 1000 (NMMB 3 h bucket), budget"),
    ("SNOM", 1, "acc"): ("carrier:NAMH_SNOM_ACC", "since:3h", "ACSNOM * 1000 (snow melt, 3 h bucket)"),
    ("SSRUN", 1, "acc"): ("carrier:NAMH_SSRUN_ACC", "since:3h", "SSROFF (surface runoff, 3 h bucket)"),
    ("BGRUN", 1, "acc"): ("carrier:NAMH_BGRUN_ACC", "since:3h", "BGROFF (subsurface runoff, 3 h bucket)"),
    ("EVP", 1, "acc"): ("carrier:NAMH_EVP_ACC", "since:3h", "SFCEVP * 1000 (3 h bucket)"),
    ("PEVAP", 1, "acc"): ("carrier:NAMH_PEVAP_ACC", "since:3h", "POTEVP * 1000 (3 h bucket)"),
    # ---- 3 h bucket averages (accumulator / counter, UPP SURFCE.f / CLDRAD.f)
    ("LHTFL", 1, "ave"): ("carrier:NAMH_LHTFL_AVE", "since:3h", "-SFCLHX / ASRFC"),
    ("SHTFL", 1, "ave"): ("carrier:NAMH_SHTFL_AVE", "since:3h", "-SFCSHX / ASRFC"),
    ("GFLUX", 1, "ave"): ("carrier:NAMH_GFLUX_AVE", "since:3h", "SUBSHX / ASRFC"),
    ("SNOHF", 1, "ave"): ("carrier:NAMH_SNOHF_AVE", "since:3h", "SNOPCX / ASRFC"),
    ("DSWRF", 1, "ave"): ("carrier:NAMH_DSWRF_AVE", "since:3h", "ASWIN / ARDSW"),
    ("DLWRF", 1, "ave"): ("carrier:NAMH_DLWRF_AVE", "since:3h", "ALWIN / ARDLW"),
    ("USWRF", 1, "ave"): ("carrier:NAMH_USWRF_AVE", "since:3h", "-ASWOUT / ARDSW"),
    ("ULWRF", 1, "ave"): ("carrier:NAMH_ULWRF_AVE", "since:3h", "-ALWOUT / ARDLW"),
    ("USWRF", 8, "ave"): ("carrier:NAMH_USWRF_TOA_AVE", "since:3h", "ASWTOA / ARDSW"),
    ("ULWRF", 8, "ave"): ("carrier:NAMH_ULWRF_TOA_AVE", "since:3h", "ALWTOA / ARDLW"),
    ("LRGHR", 200, "ave"): ("-", "-", "large-scale latent heating rate: needs TRAIN/AVRAIN in the history"),
    # ---- hourly extremes (io-max-fields, the last hour)
    ("MAXUVV", 100, "max"): ("carrier:UPVVELMAX", "-", "io-max-fields hourly maximum updraft, 100-1000 hPa"),
    ("MAXDVV", 100, "max"): ("carrier:DNVVELMAX", "-", "io-max-fields hourly maximum downdraft, 100-1000 hPa"),
    ("MAXREF", 103, "max"): ("carrier:REFDMAX", "-", "io-max-fields hourly maximum 1 km reflectivity"),
    ("REFD", 20, "max"): ("carrier:NAMH_REFD263K_MAX", "-", "io-max-fields hourly maximum -10 C reflectivity"),
    ("PRATE", 1, "max"): ("carrier:NAMH_PRATE_MAX", "-", "io-max-fields PRATEMAX (mm/h) / 3600"),
    ("SRWEQ", 1, "max"): ("carrier:NAMH_SRWEQ_MAX", "-", "io-max-fields FPRATEMAX (mm/h) / 3600"),
    # ---- native levels (1 and 2 hybrid level: the lowest model levels)
    ("PRES", 105, None): ("PRES_ON_HYBRID_LVL", "-", ""),
    ("CLMR", 105, None): ("CLMR_ON_HYBRID_LVL", "-", "QCLOUD"),
    ("CICE", 105, None): ("-", "-", "cloud ice: the history keeps Ferrier-Aligo's ice whole in QSNOW (SNMR), no split"),
    ("RWMR", 105, None): ("RWMR_ON_HYBRID_LVL", "-", "QRAIN"),
    ("SNMR", 105, None): ("SNMR_ON_HYBRID_LVL", "-", "QSNOW"),
    ("TCOND", 105, None): ("NAM_TCOND_HYBRID", "-", "QCLOUD + QRAIN + QICE + QSNOW (CWM)"),
    ("RIME", 105, None): ("carrier:F_RIMEF[0]", "-", "Ferrier-Aligo rime factor, lowest level"),
    ("HGT", 105, None): ("HGT_ON_HYBRID_LVL", "-", ""),
    ("TMP", 105, None): ("TMP_ON_HYBRID_LVL", "-", ""),
    ("RH", 105, None): ("NAM_RH_HYBRID", "-", ""),
    ("SPFH", 105, None): ("SPFH_ON_HYBRID_LVL", "-", ""),
    ("UGRD", 105, None): ("UGRD_ON_HYBRID_LVL", "-", ""),
    ("VGRD", 105, None): ("VGRD_ON_HYBRID_LVL", "-", ""),
    ("VVEL", 105, None): ("VVEL_ON_HYBRID_LVL", "-", ""),
    ("BMIXL", 105, None): ("-", "-", "Blackadar mixing length: not in the history (as awphys)"),
    # ---- isobaric
    ("SPFH", 100, None): ("SPFH_ON_ISOBARIC_SFC", "-", "NCEP's NMM isobaric rule"),
    ("CLMR", 100, None): ("CLMR_ON_ISOBARIC_SFC", "-", ""),
    ("CICE", 100, None): ("-", "-", "cloud ice: the history keeps Ferrier-Aligo's ice whole in QSNOW (SNMR), no split"),
    ("RWMR", 100, None): ("RWMR_ON_ISOBARIC_SFC", "-", ""),
    ("SNMR", 100, None): ("SNMR_ON_ISOBARIC_SFC", "-", ""),
    ("RIME", 100, None): ("iso:F_RIMEF", "-", "rime factor, linear in ln p"),
    ("TKE", 100, None): ("Q2_ON_ISOBARIC_SFC", "-", "as awphys"),
    ("DPT", 100, None): ("DPT_ON_ISOBARIC_SFC", "-", ""),
    ("MDIV", 100, None): ("-", "-", "moisture divergence: not ported"),
    ("ABSV", 100, None): ("ABSV_ON_ISOBARIC_SFC", "-", "conformal absolute vorticity of the grid-relative winds"),
    ("STRM", 100, None): ("-", "-", "stream function: not ported"),
    ("HGT", 100, None): ("HGT_ON_ISOBARIC_SFC", "-", "NCEP's NMM isobaric rule"),
    ("TMP", 100, None): ("TMP_ON_ISOBARIC_SFC", "-", "NCEP's NMM isobaric rule"),
    ("RH", 100, None): ("RH_ON_ISOBARIC_SFC", "-", "NCEP's NMM isobaric rule"),
    ("VVEL", 100, None): ("VVEL_ON_ISOBARIC_SFC", "-", "NCEP's NMM isobaric rule"),
    ("DZDT", 100, None): ("DZDT_ON_ISOBARIC_SFC", "-", "NCEP's NMM isobaric rule"),
    ("UGRD", 100, None): ("UGRD_ON_ISOBARIC_SFC", "-", "NCEP's NMM isobaric rule"),
    ("VGRD", 100, None): ("VGRD_ON_ISOBARIC_SFC", "-", "NCEP's NMM isobaric rule"),
    # ---- flight levels (m above mean sea level)
    ("TMP", 102, None): ("NAM_TMP_MSL", "-", "UPP FDLVL: linear in height, lowest level under the ground"),
    ("UGRD", 102, None): ("NAM_UGRD_MSL", "-", ""),
    ("VGRD", 102, None): ("NAM_VGRD_MSL", "-", ""),
    # ---- 30 hPa boundary layers
    ("PRES", 108, None): ("NAM_PRES_PDIFF", "-", "UPP BNDLYR layer mean"),
    ("POT", 108, None): ("NAM_POT_PDIFF", "-", ""),
    ("DPT", 108, None): ("NAM_DPT_PDIFF", "-", ""),
    ("SPFH", 108, None): ("NAM_SPFH_PDIFF", "-", ""),
    ("VVEL", 108, None): ("NAM_VVEL_PDIFF", "-", ""),
    ("PWAT", 108, None): ("NAM_PWAT_PDIFF", "-", ""),
    ("MDIV", 108, None): ("-", "-", "moisture divergence: not ported"),
    ("PLPL", 108, None): ("PLPL_ON_SPEC_PRES_ABOVE_GRND", "-", "lifted parcel level, 255-0 hPa"),
    # ---- columns
    ("TCOLW", 200, None): ("NAM_TCOL:QCLOUD", "-", ""),
    ("TCOLI", 200, None): ("-", "-", "cloud ice column: the history keeps the ice whole in QSNOW (TCOLS)"),
    ("TCOLR", 200, None): ("NAM_TCOL:QRAIN", "-", ""),
    ("TCOLS", 200, None): ("NAM_TCOL:QSNOW", "-", ""),
    ("TCOLC", 200, None): ("NAM_TCOL:QCLOUD+QRAIN+QICE+QSNOW", "-", "total condensate"),
    ("TCLSW", 200, None): ("NAM_TCOL<273.15:QCLOUD", "-", "supercooled liquid"),
    ("TCOLM", 200, None): ("-", "-", "melting ice column: not ported"),
    ("SWHR", 200, None): ("-", "-", "column shortwave heating: RSWTT not in the history"),
    ("LWHR", 200, None): ("-", "-", "column longwave heating: RLWTT not in the history"),
    ("MDIV", 200, None): ("-", "-", "moisture divergence: not ported"),
    ("GFLUX", 1, None): ("-", "-", "instantaneous ground heat flux: not in the history"),
    ("VIL", 200, None): ("VIL_ON_ENTIRE_ATMOS", "-", ""),
    # ---- surface and near-surface
    ("SPFH", 1, None): ("carrier:NAMH_SPFH_SFC", "-", "QSH"),
    ("POT", 103, None): ("carrier:NAMH_POT_10M", "-", "TH10"),
    ("CPOFP", 1, None): ("carrier:NAMH_CPOFP", "-", "SR * 100 where it precipitates, -50 elsewhere"),
    ("PRATE", 1, None): ("carrier:NAMH_PRATE", "-", "PREC * 1000 / (DT NPHS)"),
    ("CSNOW", 1, None): ("-", "-", "precipitation type: NAM's four-algorithm type not ported (as awphys)"),
    ("CICEP", 1, None): ("-", "-", "precipitation type: not ported (as awphys)"),
    ("CFRZR", 1, None): ("-", "-", "precipitation type: not ported (as awphys)"),
    ("CRAIN", 1, None): ("-", "-", "precipitation type: not ported (as awphys)"),
    ("UFLX", 1, None): ("-", "-", "momentum flux: not ported"),
    ("VFLX", 1, None): ("-", "-", "momentum flux: not ported"),
    ("SFEXC", 1, None): ("carrier:NAMH_SFEXC", "-", "SFCEXC"),
    ("CCOND", 1, None): ("-", "-", "Noah canopy conductance: not in the history"),
    ("RSMIN", 1, None): ("-", "-", "Noah minimum stomatal resistance: not in the history"),
    ("RCS", 1, None): ("-", "-", "Noah resistance factor: not in the history"),
    ("RCT", 1, None): ("-", "-", "Noah resistance factor: not in the history"),
    ("RCQ", 1, None): ("-", "-", "Noah resistance factor: not in the history"),
    ("RCSOL", 1, None): ("-", "-", "Noah resistance factor: not in the history"),
    ("CSDSF", 1, None): ("-", "-", "clear-sky downward shortwave: not in the history"),
    ("LTNG", 1, None): ("-", "-", "lightning: not ported (as awphys)"),
    ("SNFALB", 1, None): ("carrier:NAMH_SNFALB", "-", "ALBASE * 100"),
    ("MXSALB", 1, None): ("carrier:NAMH_MXSALB", "-", "MXSNAL * 100"),
    ("WTMP", 1, None): ("carrier:NAMH_WTMP", "-", "SST"),
    ("ULWRF", 8, None): ("carrier:NAMH_ULWRF_TOA", "-", "RLWTOA"),
    ("BRTMP", 8, None): ("-", "-", "brightness temperature: NAM's is not the OLR emission temperature (-25 K)"),
    # ---- clouds, levels
    ("PRES", 2, None): ("PRES_ON_CLOUD_BASE|nan=-50000", "-", "no cloud: -50000, as NCEP"),
    ("HGT", 215, None): ("GSD_HGT_ON_CLOUD_CEILING|nan=20000", "-", "no ceiling: 20000 m, as NCEP"),
    ("PRES", 206, None): ("-", "-", "grid-scale cloud bottom pressure: not ported"),
    ("PRES", 207, None): ("-", "-", "grid-scale cloud top pressure: not ported"),
    ("PRES", 3, None): ("PRES_ON_CLOUD_TOP|nan=-50000", "-", "no cloud: -50000, as NCEP"),
    ("HGT", 3, None): ("HGT_ON_CLOUD_TOP|nan=-5000", "-", "no cloud: -5000, as NCEP"),
    ("TMP", 3, None): ("-", "-", "cloud-top temperature: not ported"),
    ("VIS", 3, None): ("-", "-", "cloud-top visibility: not ported (as awphys)"),
    ("HGT", 253, None): ("-", "-", "supercooled liquid layer base: not ported"),
    ("HGT", 254, None): ("-", "-", "supercooled liquid layer top: not ported"),
    ("LCDC", 214, None): ("carrier:NAMH_LCDC", "-", "CFRACL * 100"),
    ("MCDC", 224, None): ("carrier:NAMH_MCDC", "-", "CFRACM * 100"),
    ("HCDC", 234, None): ("carrier:NAMH_HCDC", "-", "CFRACH * 100"),
    ("HLCY", 103, None): ("NAM_HLCY_ON_SPEC_HGT_LVL_ABOVE_GRND", "-", "CALHEL, the record's depth"),
    ("USTM", 103, None): ("NAM_USTM_ON_SPEC_HGT_LVL_ABOVE_GRND", "-", "CALHEL storm motion"),
    ("VSTM", 103, None): ("NAM_VSTM_ON_SPEC_HGT_LVL_ABOVE_GRND", "-", "CALHEL storm motion"),
    ("UPHL", 103, None): ("-", "-", "instantaneous 2-5 km updraft helicity: not ported"),
    ("VWSH", 7, None): ("-", "-", "tropopause wind shear: not ported"),
    ("HGT", 6, None): ("NAM_HGT_MAXWIND", "-", ""),
    ("HGT", 204, None): ("HGT_ON_HGHST_TROP_FRZ_LVL", "-", ""),
    ("HGT", 5, None): ("HGT_ON_LVL_OF_ADIAB_COND_FROM_SFC", "-", ""),
    ("PRES", 5, None): ("-", "-", "LCL pressure: not ported"),
    ("REFD", 20, None): ("REFD_ON_ISOTHERMAL", "-", "-10 C reflectivity"),
}
# depth-specific rows (NOAA name, level, level2) -> source
LAYER_RULES = {
    ("MXUPHL", "103:5000", "103:2000"): ("carrier:UPHLMAX25", "-", "io-max-fields hourly 2-5 km UH maximum"),
    ("MNUPHL", "103:5000", "103:2000"): ("carrier:UPHLMIN25", "-", "io-max-fields hourly 2-5 km UH minimum"),
    ("MXUPHL", "103:3000", "103:0"): ("carrier:UPHLMAX03", "-", "io-max-fields hourly 0-3 km UH maximum"),
    ("MNUPHL", "103:3000", "103:0"): ("carrier:UPHLMIN03", "-", "io-max-fields hourly 0-3 km UH minimum"),
    ("CAPE", "108:9000", "108:0"): ("MIXED_LAYER_CAPE_ON_SPEC_PRES_ABOVE_GRND", "-", "mixed-layer parcel, 90 hPa"),
    ("CIN", "108:9000", "108:0"): ("MIXED_LAYER_CIN_ON_SPEC_PRES_ABOVE_GRND", "-", ""),
    ("CAPE", "108:25500", "108:0"): ("UNSTABLE_CAPE_ON_SPEC_PRES_ABOVE_GRND", "-", "most unstable parcel, 255 hPa"),
    ("CIN", "108:25500", "108:0"): ("UNSTABLE_CIN_ON_SPEC_PRES_ABOVE_GRND", "-", ""),
    ("SPFH", "103:10", "-"): ("carrier:NAMH_SPFH_10M", "-", "Q10"),
}


def tke_hybrid(level):
    return f"carrier:TKE_PBL[{int(float(level.split(':')[1])) - 1}]"


def build(ref: Path, idx: Path, awphys: list) -> list[str]:
    skel = spec.skeleton(ref).splitlines()
    names = [ln.split(":")[3] for ln in idx.read_text().splitlines() if ln.strip()]
    body = [ln for ln in skel if ln and not ln.startswith("#") and not ln.startswith("rec\t")]
    if len(body) != len(names):
        raise SystemExit(f"{ref}: {len(body)} records, {idx}: {len(names)} lines")
    amap = {}
    for a in awphys:
        amap.setdefault((a["dcp"], a["level"], a["level2"], a["stat"] or "-"), []).append(a)
    out = [f"# WOOF-NAM NAM nest hiresf record table (namgpu/products/spec.py documents the columns). Order = NCEP's; "
           f"generated by tools/hiresf_tables.py from {ref.name}.", "\t".join(spec.COLUMNS)]
    stats = {"rules": 0, "awphys": 0, "left": 0}
    for ln, nm in zip(body, names):
        c = ln.split("\t")
        c += [""] * (len(spec.COLUMNS) - len(c))
        r = dict(zip(spec.COLUMNS, c))
        r["name"] = nm
        t1 = lt(r)
        stat = None if r["stat"] in ("-", "") else r["stat"]
        got = LAYER_RULES.get((nm, r["level"], r["level2"])) or RULES.get((nm, t1, stat))
        if got is None and nm == "TKE" and t1 == 105:
            got = (tke_hybrid(r["level"]), "-", "MYJ TKE (Q2 / 2)")
        if got is None:
            cand = [a for a in amap.get((r["dcp"], r["level"], r["level2"], r["stat"] or "-"), [])
                    if a["source"] not in ("", "-")]
            if cand:
                a = cand[0]
                window = "-" if stat in ("max", "min", None) else "since:3h"
                got = (a["source"], window, f"as awphys rec {a['rec']}")
                stats["awphys"] += 1
        else:
            stats["rules"] += 1
        if got is None:
            got = ("-", "-", "no source yet")
        if got[0] == "-":
            stats["left"] += 1
        r["source"], r["window"], r["note"] = got[0], got[1], got[2] or r["note"]
        r["bits"] = "-"
        out.append("\t".join(r[k] for k in spec.COLUMNS))
    print(f"{ref.name}: {len(body)} records; {stats['rules']} by rule, {stats['awphys']} as awphys, "
          f"{stats['left']} left out")
    return out


FIREWX_1H = (("since:3h", "since:1h"), ("carrier:APCP_S3", "carrier:APCP_S1"), ("carrier:NCPCP_S3", "carrier:NCPCP_S1"),
             ("NCEP's 3 h bucket", "NCEP's 1 h fire-weather bucket"), ("3 h bucket", "1 h bucket"))


def firewx_table(shared: list[str]) -> list[str]:
    """hiresf.tsv's lines for the fire-weather nest: the same records, its accumulations and averages over the last
    hour (NCEP's nhrs_* = 1 for firewx; the model empties its accumulators every hour, ncep_nests firewxnest keys)."""
    out = [shared[0].split(". Order", 1)[0].replace("record table", "record table, fire-weather nest (1 h bucket)")
           + ". Order = NCEP's; generated by tools/hiresf_tables.py: hiresf.tsv with NCEP's 1 h firewx bucket "
           "(nam_configfile_firewxnest nhrs_* 1)."]
    for ln in shared[1:]:
        for a, b in FIREWX_1H:
            ln = ln.replace(a, b)
        out.append(ln)
    return out


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    src = Path(argv[0])
    awphys = spec.read(spec.TABLE)
    files = {n: (next(src.glob(f"nam.t*z.{n}.hiresf01.tm00.grib2")), ) for n in NESTS}
    tables = {}
    for n in NESTS:
        ref = files[n][0]
        tables[n] = build(ref, Path(str(ref) + ".idx"), awphys)
    rest = [tables[n][2:] for n in NESTS[1:]]
    for n, t in zip(NESTS[1:], rest):
        if [ln.split("\t")[:6] for ln in t] != [ln.split("\t")[:6] for ln in rest[0]]:
            raise SystemExit(f"{n}: its record list differs from alaskanest's")
    (HERE / "hiresf-conusnest.tsv").write_text("\n".join(tables["conusnest"]) + "\n")
    shared = tables["alaskanest"][:]
    shared[0] = shared[0].replace(files["alaskanest"][0].name, "the alaskanest/hawaiinest/priconest/firewxnest "
                                  "hiresf01 files (one record list)")
    (HERE / "hiresf.tsv").write_text("\n".join(shared) + "\n")
    (HERE / "hiresf-firewxnest.tsv").write_text("\n".join(firewx_table(shared)) + "\n")
    texts = {}
    for n in NESTS:
        ref = files[n][0]
        table = HERE / {"conusnest": "hiresf-conusnest.tsv", "firewxnest": "hiresf-firewxnest.tsv"}.get(n, "hiresf.tsv")
        texts[n] = spec.noaa_table(ref, Path(str(ref) + ".idx"), table).splitlines()
    # NOAA's decimal/binary scale per record differs between the nests' files where wgrib2 repacked a field that was
    # constant there (Hawaii's TCOLS: D = 20, which no real snow column fits in 32 bits). Each record takes the scale
    # most of the five nests' files carry (ties: the coarser), so one quiet field in one file sets no precision.
    votes = {}
    for n, lines in texts.items():
        seen = {}
        for ln in lines[2:]:
            c = ln.split("\t")
            k = tuple(c[2:6])
            occ = seen.get(k, 0)
            seen[k] = occ + 1
            votes.setdefault(k + (occ,), []).append((int(c[8]), int(c[9])))
    pick = {}
    for k, v in votes.items():
        counts = {}
        for de in v:
            counts[de] = counts.get(de, 0) + 1
        pick[k] = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0][0]))[0][0]
    for n, lines in texts.items():
        out, seen, changed = lines[:2], {}, 0
        out[0] += " Scales: the five nests' majority per record (tools/hiresf_tables.py)."
        for ln in lines[2:]:
            c = ln.split("\t")
            k = tuple(c[2:6])
            occ = seen.get(k, 0)
            seen[k] = occ + 1
            d, e = pick[k + (occ,)]
            changed += (str(d), str(e)) != (c[8], c[9])
            c[8], c[9] = str(d), str(e)
            out.append("\t".join(c))
        (HERE / f"hiresf-{n}-noaa.tsv").write_text("\n".join(out) + "\n")
        print(f"hiresf-{n}-noaa.tsv written ({changed} records take the majority scale)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
