"""The awphys land, soil, snow, surface-flux, radiation and cloud records, computed from a woof-nam history file.

Each carrier is a field on the model grid that namgpu.io.products regrids to the output grid and the GRIB2 writer
encodes (awphys rows `carrier:NAM_*`, and the nest rows built from them). compute(history, name) returns the
native-grid array, or None when the history lacks an input, and the regrid mode: "nearest" for classes, masks,
table values and the soil state, None (the set's continuous rule) for the rest.

Written from the published sources below; the values are not bitwise to NCEP's own files.

- Soil state and soil depths: the Noah land-surface model as the NAM runs it (four layers 0.1, 0.3, 0.6 and 1.0 m
  thick, 2 m in all), Ek et al. (2003), section 2; Noah LSM User's Guide 2.7.1 (Mitchell 2005), section 5
  (SLDPTH) and section 7 (SMC, SH2O, STC, CMC, SNOWH). Total soil moisture (SOILM, 0-2 m) is the column water
  rho_w sum(SMC_k dz_k); moisture availability (MSTAV, 0-1 m) is the model's root-zone availability (Chen et al.
  1996, section 2; Ek et al. 2003, section 2) as a percentage.
- The five soil-table records and the root layers: the Noah parameters as the User's Guide, section 8, defines
  them (SMCMAX porosity, SMCREF onset of transpiration stress, SMCWLT wilting point, SMCDRY end of direct soil
  evaporation, NROOT soil layers reached by roots), read for each point's soil and vegetation class from the
  shipped NCEP tables namgpu/data/fix/noah/SOILPARM.TBL (STAS) and VEGPARM.TBL (IGBP).
- Canopy water: CMC (m of water) as kg m-2 (User's Guide, section 8, CMCMAX 0.0005 m).
- Surface radiation at the output time: the model's own fluxes from its last radiation call, carried to the
  output step as the model carries them to the surface between calls: shortwave by cos(zenith now) / mean
  cos(zenith) of the call, downward longwave by sigma T_LM^4 / SIGT4 (T_LM the lowest-layer temperature at the
  output step, SIGT4 the same at the call); the emitted longwave is the surface's own (RADOT).
- Drag coefficient: the bulk relation u*^2 = C_D M^2 (Stull 1988, section 7.4), M the lowest-layer wind speed.
- Field definitions, units and sign conventions: the NCEP GRIB2 tables (discipline 0 and 2 parameter tables,
  level types 1, 2, 106 and 200) and the DTC's published product list for NCEP's GRIB2 output.
- Snow cover: the GRIB2 definition at grid-point scale, 100 % where the point holds any snow, as NCEP's archived
  awphys records show.
- Total cloud (TCDC, entire atmosphere as one layer): the model radiation's layer cloud fraction (CLDFRA), each
  layer's fraction taken as its mean over the 3 x 3 block of grid points centred on the point (edge rows and
  columns repeated at the domain boundary), and the layers combined by maximum overlap (Geleyn and Hollingsworth
  1979, Contrib. Atmos. Phys. 52, 1-16, section 3: the cover of maximally overlapped layers is the largest layer
  cover). The block size and the overlap are the choices NCEP's own native-grid files fix: in them the total equals
  the largest of the low, middle and high cloud records at every point, and each of those equals the largest
  3 x 3 block mean of the layer cloud records in its pressure band (low above 642 hPa, high below 350 hPa) to
  0.15, 0.11 and 0.07 % mean difference.
- Cloud base (CLDBASE, GRIB2 level type 2): the height of the middle of the lowest cloudy layer, where a layer is
  cloudy when it lies at or above the bottom of the model's convective cloud (CNVBOT, the lowest convective cloud
  bottom since the last output) or holds more than CLOUD_Q of cloud water, cloud ice and snow together; -5000 gpm
  where the column has neither. NCEP's native-grid files fix the form: their cloud-base pressure is the higher of
  their grid-scale cloud bottom and convective cloud bottom pressures at every point, and the grid-scale bottom
  is the lowest layer whose condensate passes a threshold between 1e-5 and 2e-5 kg kg-1 (the files hold mixing
  ratios to 1e-5). CLOUD_Q = 1e-5 is the value in that range that best fits NCEP's awphys cloud-base records.
- The precipitation-type flags: namgpu.products.ptype.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

# model constants (NMMB module_CONSTANTS, as namgpu/kernels/nmmb_common.cuh)
G = 9.8060226                       # m s-2
CAPPA = 287.04 / 1004.6             # R_d / c_p
STBOL = 5.67051e-8                  # Stefan-Boltzmann constant, W m-2 K-4
RHOWATER = 1000.0                   # kg m-3
NSOIL_DZ = np.array([0.1, 0.3, 0.6, 1.0], np.float32)   # Noah soil layer thicknesses, m (Ek et al. 2003)
WATER_SOILM = 1000.0                # SOILM over open water, kg m-2 (NCEP's records)
NO_CLOUD = -5000.0                  # CLDBASE where the column has no cloud, gpm (NCEP's records)
CLOUD_Q = 1.0e-5                    # kg kg-1 of cloud water + cloud ice + snow that makes a layer cloudy for CLDBASE
MIN_WIND = 1.0e-3                   # m s-1, the floor of M in C_D = u*^2 / M^2 (a calm point stays finite)

SOIL_DIMS = ("soil_layers_stag", "south_north", "west_east")
FIX = Path(__file__).resolve().parents[1] / "data" / "fix" / "noah"


def _get(s, name):
    if name not in s.variables:
        return None
    return np.asarray(s[name][0], dtype=np.float32)


def _need(s, *names):
    out = [_get(s, n) for n in names]
    return None if any(a is None for a in out) else out


def _masks(s):
    """(open water, sea ice, land) [j, i] booleans: SM is the model's water mask, SICE its sea-ice flag."""
    sm = _get(s, "SM")
    if sm is None:
        return None
    sice = _get(s, "SICE")
    sice = np.zeros_like(sm) if sice is None else sice
    water = sm > 0.5
    ice = (~water) & (sice > 0.5)
    return water, ice, ~(water | ice)


# --- the shipped Noah tables --------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _tables():
    """The shipped tables, read the way the forecast's LSM reads them (phys_noah_parm_tables, IGBP classes)."""
    from namgpu.physics.noah import phys_noah_parm_tables as parm
    return parm.parse_tables(FIX, ivegsrc=1)


def soil_table() -> dict:
    """STAS soil parameters by class (index 0 is class 1): DRYSMC, MAXSMC, REFSMC, WLTSMC."""
    t = _tables()
    return {k: np.asarray(t.soil[k][:t.slcats], np.float32) for k in ("DRYSMC", "MAXSMC", "REFSMC", "WLTSMC")}


def veg_table() -> dict:
    """IGBP vegetation parameters by class (index 0 is class 1): NROOT."""
    t = _tables()
    return {"NROOT": np.asarray(t.veg["NROTBL"][:t.lucats], np.float32)}


def _by_class(cls, table, land):
    k = np.rint(cls).astype(np.int64) - 1
    ok = land & (k >= 0) & (k < table.size)
    out = np.zeros(cls.shape, np.float32)
    out[ok] = table[k[ok]]
    return out


def _soil_param(key):
    def f(s):
        got = _need(s, "ISLTYP")
        m = _masks(s)
        if got is None or m is None:
            return None
        return _by_class(got[0], soil_table()[key], m[2])
    return f


def rlyrs(s):
    got = _need(s, "IVGTYP")
    m = _masks(s)
    if got is None or m is None:
        return None
    return _by_class(got[0], veg_table()["NROOT"], m[2])


# --- soil, snow and land state ------------------------------------------------------------------------------------

def _layers(name):
    def f(s):
        return _get(s, name)
    return f


def soilm(s):
    """Total soil moisture 0-2 m, kg m-2: the model's SMSTOT over land, rebuilt as rho_w sum(SMC dz) when the
    history's SMSTOT is all zero over land; WATER_SOILM over open water, 0 on sea ice."""
    m = _masks(s)
    if m is None:
        return None
    water, ice, land = m
    tot = _get(s, "SMSTOT")
    if tot is None or not np.any(tot[land]):
        smc = _get(s, "SMOIS")
        if smc is None:
            return None
        nz = min(smc.shape[0], NSOIL_DZ.size)
        tot = RHOWATER * np.tensordot(NSOIL_DZ[:nz], smc[:nz], axes=1)
    out = np.where(land, tot, 0.0)
    return np.where(water, WATER_SOILM, out).astype(np.float32)


def mstav(s):
    """Moisture availability 0-1 m, %: 100 x the model's SMSTAV over land, 0 elsewhere."""
    a, m = _get(s, "SMSTAV"), _masks(s)
    if a is None or m is None:
        return None
    return np.where(m[2], 100.0 * np.clip(a, 0.0, 1.0), 0.0).astype(np.float32)


def _scaled(name, k=1.0):
    def f(s):
        a = _get(s, name)
        return None if a is None else (a * np.float32(k)).astype(np.float32)
    return f


def snowc(s):
    """Snow cover, %: 100 where the point holds any snow water, else 0."""
    a = _get(s, "SNO")
    return None if a is None else np.where(a > 0.0, 100.0, 0.0).astype(np.float32)


def land(s):
    m = _masks(s)
    return None if m is None else m[2].astype(np.float32)


def icec(s):
    """Sea-ice cover, fraction: the model's SICE."""
    a = _get(s, "SICE")
    return None if a is None else np.clip(a, 0.0, 1.0).astype(np.float32)


def _class(name):
    def f(s):
        a, m = _get(s, name), _masks(s)
        if a is None or m is None:
            return None
        return np.where(m[2], np.rint(a), 0.0).astype(np.float32)
    return f


# --- surface fluxes, drag and radiation ---------------------------------------------------------------------------

def _neg(name):
    def f(s):
        a = _get(s, name)
        return None if a is None else (-a).astype(np.float32)
    return f


def cd(s):
    """Drag coefficient u*^2 / M^2, M the lowest-layer wind speed (Stull 1988, 7.4)."""
    got = _need(s, "USTAR", "U", "V")
    if got is None:
        return None
    us, u, v = got
    m2 = np.maximum(u[0] * u[0] + v[0] * v[0], MIN_WIND * MIN_WIND)
    return (us * us / m2).astype(np.float32)


def _sw_factor(s):
    got = _need(s, "CZEN", "CZMEAN")
    if got is None:
        return None
    cz, czm = got
    return np.where(czm > 0.0, cz / np.where(czm > 0.0, czm, 1.0), 0.0).astype(np.float32)


def dswrf(s):
    a, f = _get(s, "RSWIN"), _sw_factor(s)
    return None if a is None or f is None else (a * f).astype(np.float32)


def uswrf(s):
    a, f = _get(s, "RSWOUT"), _sw_factor(s)
    return None if a is None or f is None else (a * f).astype(np.float32)


def dlwrf(s):
    """RLWIN x sigma T_LM^4 / SIGT4; RLWIN as it stands where SIGT4 is not set."""
    got = _need(s, "RLWIN", "SIGT4", "T", "PRES")
    if got is None:
        return None
    rlw, sig, th, p = got
    t = (th[0] + 300.0) * (p[0] / 1.0e5) ** CAPPA
    fac = np.where(sig > 0.0, STBOL * t ** 4 / np.where(sig > 0.0, sig, 1.0), 1.0)
    return (rlw * fac).astype(np.float32)


def ulwrf(s):
    return _get(s, "RADOT")


# --- cloud --------------------------------------------------------------------------------------------------------

def _block3(a):
    """Mean of a [j, i] field over the 3 x 3 block of points centred on each point, the edge rows and columns
    repeated at the domain boundary."""
    p = np.pad(a, 1, mode="edge")
    jm, im = a.shape
    out = np.zeros(a.shape, np.float64)
    for dj in range(3):
        for di in range(3):
            out += p[dj:dj + jm, di:di + im]
    return out / 9.0


def tcdc(s):
    """Total cloud cover, %: the largest over the layers of the layer cloud fraction's 3 x 3 block mean (maximum
    overlap of the block-scale layer covers)."""
    cf = _get(s, "CLDFRA")
    if cf is None:
        return None
    top = np.zeros(cf.shape[1:], np.float64)
    for layer in cf:
        np.maximum(top, _block3(np.clip(layer, 0.0, 1.0)), out=top)
    return (100.0 * np.clip(top, 0.0, 1.0)).astype(np.float32)


def cldbase(s):
    """Cloud base, gpm: the middle of the lowest cloudy layer, NO_CLOUD where the column has none. A layer is cloudy
    when it holds more than CLOUD_Q of cloud water, cloud ice and snow together, or when it is the convective cloud
    bottom CNVBOT or above it. CNVBOT counts layers from the ground (1 the lowest); a value outside 1..LM means no
    convective cloud since the last output."""
    ph = _get(s, "PH")
    q = [a for a in (_get(s, "QCLOUD"), _get(s, "QICE"), _get(s, "QSNOW")) if a is not None]
    cnv = _get(s, "CNVBOT")
    if ph is None or (not q and cnv is None):
        return None
    z = ph / np.float32(G)
    zmid = 0.5 * (z[1:] + z[:-1])
    lm = zmid.shape[0]
    none = np.int64(lm)
    if q:
        cloudy = sum(q) > CLOUD_Q
        k = np.where(cloudy.any(axis=0), np.argmax(cloudy, axis=0), none)
    else:
        k = np.full(zmid.shape[1:], none, np.int64)
    if cnv is not None:
        kc = np.rint(cnv).astype(np.int64)
        k = np.minimum(k, np.where((kc >= 1) & (kc <= lm), kc - 1, none))
    h = np.take_along_axis(zmid, np.minimum(k, lm - 1)[None], axis=0)[0]
    return np.where(k < lm, h, NO_CLOUD).astype(np.float32)


# --- precipitation type -------------------------------------------------------------------------------------------

def _ptype(code_name):
    def f(s):
        from namgpu.products import ptype as PT
        return PT.flag(s, getattr(PT, code_name))
    return f


NEAREST = "nearest"
CARRIERS = {
    "NAM_TSOIL": (_layers("TSLB"), NEAREST),
    "NAM_SOILW": (_layers("SMOIS"), NEAREST),
    "NAM_SOILL": (_layers("SH2O"), NEAREST),
    "NAM_TSOIL3M": (_layers("TG"), NEAREST),
    "NAM_MSTAV": (mstav, NEAREST),
    "NAM_SOILM": (soilm, NEAREST),
    "NAM_CNWAT": (_scaled("CMC", RHOWATER), NEAREST),
    "NAM_WEASD": (_scaled("SNO"), None),
    "NAM_SNOWC": (snowc, NEAREST),
    "NAM_SNOD": (_scaled("SI", 1.0e-3), None),
    "NAM_SMDRY": (_soil_param("DRYSMC"), NEAREST),
    "NAM_POROS": (_soil_param("MAXSMC"), NEAREST),
    "NAM_CSNOW": (_ptype("SN"), NEAREST),
    "NAM_CICEP": (_ptype("IP"), NEAREST),
    "NAM_CFRZR": (_ptype("ZR"), NEAREST),
    "NAM_CRAIN": (_ptype("RA"), NEAREST),
    "NAM_SFCR": (_layers("Z0"), NEAREST),
    "NAM_FRICV": (_layers("USTAR"), None),
    "NAM_CD": (cd, None),
    "NAM_VEG": (_scaled("VEGFRC", 100.0), NEAREST),
    "NAM_VGTYP": (_class("IVGTYP"), NEAREST),
    "NAM_SOTYP": (_class("ISLTYP"), NEAREST),
    "NAM_RLYRS": (rlyrs, NEAREST),
    "NAM_WILT": (_soil_param("WLTSMC"), NEAREST),
    "NAM_SMREF": (_soil_param("REFSMC"), NEAREST),
    "NAM_CLDBASE": (cldbase, None),
    "NAM_LAND": (land, NEAREST),
    "NAM_ICEC": (icec, NEAREST),
    "NAM_ALBDO": (_scaled("ALBEDO", 100.0), NEAREST),
    "NAM_SHTFL": (_neg("TWBS"), None),
    "NAM_LHTFL": (_neg("QWBS"), None),
    "NAM_TCDC": (tcdc, None),
    "NAM_DSWRF": (dswrf, None),
    "NAM_DLWRF": (dlwrf, None),
    "NAM_USWRF": (uswrf, None),
    "NAM_ULWRF": (ulwrf, None),
}
SOIL = ("NAM_TSOIL", "NAM_SOILW", "NAM_SOILL")


def compute(history: str, name: str):
    """Worker side: (native-grid array or None, regrid mode) for one carrier of one history file."""
    import netCDF4
    fn, mode = CARRIERS[name]
    with netCDF4.Dataset(history) as s:
        s.set_auto_mask(False)
        return fn(s), mode
