"""Build fix/noah/{VEGPARM,SOILPARM,GENPARM}.TBL in the layout NMMB's LSM_PARM_INIT reads
(phys/module_LS_NOAHLSM.F90:1298-1547): VEGPARM first section 'IGBP', 11 items per row
(IINDEX, ALBEDO, Z0, SHDFAC, NROOT, RS, RGL, HS, SNUP, LAI, MAXALB).

VEGPARM.TBL and GENPARM.TBL are NCEP's operational NMMB Noah tables, byte for byte: the IGBP-first
IGBP_VEGPARM.TBL and the GENPARM.TBL of the HMON package's NMMB fix set (hmon.v3.2.13/fix/nmmb on the
NCO code site, NOAA US government work). HMON runs the same NMMB module_LS_NOAHLSM as the NAM
(land_surface: noah). The NAM's own fix copies (nam_VEGPARM.TBL, nam_GENPARM.TBL, copied to VEGPARM.TBL
and GENPARM.TBL by scripts/exnam_fcst.sh:630-633) were taken off the NCO site when the NAM retired and
could not be fetched. The tables carry SALP 4.0 and the IGBP SNUP list 0.080 x5, 0.020, 0.020, 0.060,
0.040, 0.020, 0.010, 0.020, 0.020, 0.020, 0.013, 0.013, 0.010, 0.020, 0.020, 0.020 m. With SFLX's SNUP = 4*SNUP (module_LS_NOAHLSM.F90:1989) they
reproduce the archived NAM's surface albedo over snow. The WRF V3.0 MODIFIED_IGBP_MODIS_NOAH stand-in that came before them had
SNUP 0.04 and SALP 2.6, which halved the snow cover fraction on shallow and moderate snow.

SOILPARM.TBL is NCEP's operational NMMB soil table, byte for byte, from the same HMON fix set (its STAS
section only, 19 classes). It replaced WRF V3.0's STAS table, which differed in several classes (sand
MAXSMC 0.339 against NCEP's 0.395, BB 2.79 against 4.05; clay MAXSMC 0.468 against 0.457; organic,
bedrock, playa and lava too). The awphys soil-table records (namgpu/products/land_cloud.py: POROS, SMREF, WILT, SMDRY)
carry these values (sand 0.395, 0.236, 0.023, 0.023). The GFS start does
not rescale soil moisture by the table's porosity (namgpu/init/soil_start.py rows; init-native-sfc uses
the init tables in namgpu/data/noah_tables for the urban remap and the frozen-soil split only).

The NAM state carries its own ALBASE / Z0BASE / VEGFRC, so the table ALBEDO/Z0/SHDFAC are fallbacks in
NOAH_LSM_INIT.

    python tools/build_noah_fix.py [dest=fix/noah]
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

NCEP = "https://www.nco.ncep.noaa.gov/pmb/codes/nwprod/hmon.v3.2.13/fix/nmmb/"
PINS = {  # published name -> (name written here, sha256 of the NCEP bytes)
    "IGBP_VEGPARM.TBL": ("VEGPARM.TBL", "4dc8d3ea7a27551bc3ddfbfdc995913bfc2f4aaebf704d420a274a38d1a29cdd"),
    "GENPARM.TBL": ("GENPARM.TBL", "7f82cb68458f9a3ba7dcac08a0455edafee4705dcea9e06ad780b3d2ee2e7a31"),
    "SOILPARM.TBL": ("SOILPARM.TBL", "350923cd6b21f580d9ceb1868e97fb7ac24bf914314355a43e328846bd85e448"),
}

dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "namgpu" / "data" / "fix" / "noah"
dest.mkdir(parents=True, exist_ok=True)
for name, (out, sha) in PINS.items():
    req = urllib.request.Request(NCEP + name, headers={"User-Agent": "build_noah_fix"})
    data = urllib.request.urlopen(req, timeout=60).read()
    got = hashlib.sha256(data).hexdigest()
    if got != sha:
        raise SystemExit(f"build_noah_fix: {name} sha256 {got} is not the pinned {sha}")
    (dest / out).write_bytes(data)
(dest / "PROVENANCE.txt").write_text(__doc__)
print(f"build_noah_fix: NCEP NMMB VEGPARM/GENPARM/SOILPARM (sha256 pinned) -> {dest}")
