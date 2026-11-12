"""phys-ruc-driver: the RUC LSM (9 soil levels, snow and frozen soil on) in place of NAM's Noah on land points.

Selected with NAMGPU_LSM=ruc (default: noah, NAM's own scheme, unchanged bit for bit). The column physics is
ArWen's port of WRF v4.6.1 LSMRUC (gpuwm.core.ruc.ruc_land_surface_step with its CUDA sfctmp leaves and stages,
the device-resident set of gpuwm.core.ruc_runtime.ruc_device_sfctmp_sets); this module is the NMMB seam:

  - columns: every B1 land point without sea ice (XLAND < 1.5, SICE <= 0.5). Water and sea-ice points stay with
    NAM's Noah driver (its kernel skips the RUC points), so NAM's ice physics is unchanged.
  - forcing from TURBL's hand-over: lowest-layer DZ, mid pressure PHMID, T, Q (specific -> mixing ratio), QC,
    density RR, downward LW RLW_DN_SFC, absorbed SW RSW_NET_SFC, the JSFC exchange coefficient CHS (m/s) as
    FLHC = cp_m rho CHS and FLQC = rho CHS MAVAIL (WRF's non-MYJ coupling), RAINBL (mm per physics step) with
    Ferrier's SR as the frozen fraction (resolved liquid RAINBL (1-SR), resolved frozen RAINBL SR as snow).
  - outputs to the NMMB names Noah writes: TSFC, TWBS (HFX), QWBS (QFX), ELFLX (LH), GRNFLX, QSH (surface
    mixing ratio qvg, Noah's QSFC convention), SNOW, SNOWH, SNOWC, ALBEDO, EPSR, SMSTAV, SMSTOT, SSROFF, BGROFF,
    ACSNOW, ACSNOM, CHKLOWQ, and CHS2/CQS2 as Noah's driver sets them. The nine-level RUC state lives in RUC_*
    arrays; SMC/STC/SH2O (NAM's four Noah layers 0-10-40-100-200 cm, read by SOILTB and the products) are the
    layer means of the nine-level profile after every call.
  - start: the nine-level profile is built at the first call from the start's four Noah layers (linear in depth
    through the layer centres, skin TSFC at 0 m, deep TG at 8 m), as WRF's init_soil_3_real builds RUC levels
    from a Noah source; then, where NAMGPU_RUC_HRRR_SOIL names an HRRR analysis (namgpu.init.ruc_hrrr_soil),
    every land point inside HRRR's grid takes HRRR's own nine-level TSOIL/SOILW (bilinear over HRRR land
    neighbours, on the GPU; level 0 stays the model skin), soil moisture capped at the RUC porosity of the
    point's class; then ruclsminit (gpuwm ruc cold start) derives SH2O, SMFR3D, MAVAIL.
  - 2 m diagnostics stay NAM's (phys-noah-sfcdiags from TWBS/QWBS/TSFC/QSH, CHS2, CQS2), so a T2/Td2 change is
    the land model's, not a different diagnostic.
Not written by RUC on its land points (kept at their start values): CMC, POTEVP, SNOPCX, Z0 (JSFC keeps NAM's).
Bands: the seam runs on the whole domain only (the runner sets NAMGPU_PHYS_BAND_ROWS=0); a band state raises.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

UNIT = "phys_ruc_driver"
NZ = 9
ZS = np.array([0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00], np.float32)
NOAH_EDGES = np.array([0.0, 0.1, 0.4, 1.0, 2.0])
NOAH_MID = np.array([0.05, 0.25, 0.70, 1.50])
TG_DEPTH = 8.0                    # Noah ZBOT: TG is the temperature at 8 m
CP_D = 1004.6
MMINLU = "MODIFIED_IGBP_MODIS_NOAH"

STATE_2D = ("RUC_QVG", "RUC_QCG", "RUC_QSG", "RUC_DEW", "RUC_SOILT1", "RUC_TSNAV", "RUC_MAVAIL", "RUC_SFCEXC",
            "RUC_LAI", "RUC_Z0", "RUC_ZNT", "RUC_CANWAT", "RUC_SFCEVP", "RUC_ACRUNOFF", "RUC_RHOSNF",
            "RUC_PRECIPFR", "RUC_SNOWFALLAC", "RUC_QSFC")
STATE_3D = ("RUC_SMOIS", "RUC_SH2O", "RUC_TSLB", "RUC_SMFR3D", "RUC_KEEPFR")


def urban_on_noah() -> bool:
    """NAMGPU_RUC_URBAN=noah (default): urban points (MODIS 13, 31-33) stay on NAM Noah's urban treatment; ruc: RUC owns them.
    Breakage it prevents: the RUC port has no urban physics (only a snow-cover cap), so a city cell evaporated like natural
    soil at the cell's green fraction: SE ASOS (mostly urban cells) latent heat 2-3x NAM Noah's, 2 m dewpoint +3..+9 F moist
    and the cold-air-damming wedge too cold in the afternoon (wedge lane, 2026-11-09)."""
    return os.environ.get("NAMGPU_RUC_URBAN", "noah").strip().lower() != "ruc"


def noah_skip_mode() -> int:
    return 2 if urban_on_noah() else 1


def selected() -> bool:
    return os.environ.get("NAMGPU_LSM", "noah").strip().lower() == "ruc"


def _layer_weights():
    """W[4, 9]: the mean over each Noah layer of the piecewise-linear RUC profile."""
    z = np.linspace(0.0, 2.0, 20001)
    w = np.zeros((4, NZ))
    for k in range(4):
        zz = z[(z >= NOAH_EDGES[k]) & (z <= NOAH_EDGES[k + 1])]
        for n in range(NZ):
            e = np.zeros(NZ)
            e[n] = 1.0
            yy = np.interp(zz, ZS, e)
            w[k, n] = float(np.sum(0.5 * (yy[1:] + yy[:-1]) * np.diff(zz))) / (zz[-1] - zz[0])
    return w.astype(np.float32)


def _profile_from_noah(xp, lay, top, bottom, bottom_depth):
    """[9, N] from four layer values [4, N] at the layer centres, top value at 0 m, bottom value at bottom_depth."""
    zs = np.concatenate([[0.0], NOAH_MID, [bottom_depth]])
    out = xp.empty((NZ, lay.shape[1]), xp.float32)
    for n, z in enumerate(ZS):
        k = int(np.searchsorted(zs, z, side="right")) - 1
        k = min(max(k, 0), len(zs) - 2)
        a = float((z - zs[k]) / (zs[k + 1] - zs[k]))
        lo = top if k == 0 else lay[k - 1]
        hi = bottom if k + 1 == len(zs) - 1 else lay[k]
        out[n] = (1.0 - a) * lo + a * hi
    return out


def _hrrr_sample(xp, path, lat_deg, lon_deg):
    """HRRR nine-level TSOIL/SOILW at the given points: bilinear over HRRR land neighbours. Returns (t, w, ok)."""
    z = np.load(path)
    R = float(z["radius"])
    phi1 = np.radians(float(z["latin1"]))
    lov = float(z["lov"])
    n = np.sin(phi1)
    F = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** n / n

    def xy(lat, lon):
        rho = R * F / xp.tan(np.pi / 4 + xp.radians(lat) / 2) ** n
        th = n * xp.radians(((lon - lov + 180.0) % 360.0) - 180.0)
        return rho * xp.sin(th), -rho * xp.cos(th)

    x0, y0 = xy(xp.asarray(float(z["lat1"])), xp.asarray(float(z["lon1"])))
    x, y = xy(lat_deg.astype(xp.float64), lon_deg.astype(xp.float64))
    fi = (x - x0) / float(z["dx"])
    fj = (y - y0) / float(z["dy"])
    land = xp.asarray(z["land"]) > 0.5
    ny, nx = land.shape
    i0 = xp.floor(fi).astype(xp.int64)
    j0 = xp.floor(fj).astype(xp.int64)
    inside = (i0 >= 0) & (j0 >= 0) & (i0 < nx - 1) & (j0 < ny - 1)
    i0c = xp.clip(i0, 0, nx - 2)
    j0c = xp.clip(j0, 0, ny - 2)
    ax = (fi - i0c).astype(xp.float32)
    ay = (fj - j0c).astype(xp.float32)
    tsoil = xp.asarray(z["tsoil"])
    soilw = xp.asarray(z["soilw"])
    wsum = xp.zeros(fi.shape, xp.float32)
    tacc = xp.zeros((NZ,) + fi.shape, xp.float32)
    wacc = xp.zeros_like(tacc)
    for dj, di, wgt in ((0, 0, (1 - ax) * (1 - ay)), (0, 1, ax * (1 - ay)), (1, 0, (1 - ax) * ay),
                        (1, 1, ax * ay)):
        jj, ii = j0c + dj, i0c + di
        m = land[jj, ii].astype(xp.float32) * wgt
        wsum += m
        tacc += tsoil[:, jj, ii] * m
        wacc += soilw[:, jj, ii] * m
    ok = inside & (wsum > 1e-3)
    d = xp.where(ok, wsum, 1.0)
    return (tacc / d).astype(xp.float32), (wacc / d).astype(xp.float32), ok


def _root():
    from pathlib import Path
    return Path(os.environ.get("NAMGPU_ROOT") or ".")


def _auto_hrrr_soil(ctx, v):
    """NAMGPU_RUC_HRRR_SOIL=auto (suite rows): the HRRR f00 nine-level soil analysis at the start time, fetched once per
    cycle into $NAMGPU_ROOT/hrrr/ruc_soil_<YYYYMMDDHH>.npz (namgpu.init.ruc_hrrr_soil). A start HRRR's archive does not
    have is refused by the fetch (the run says which file), never run on a silent remap."""
    if v != "auto":
        return v
    from namgpu.init import ruc_hrrr_soil
    cyc = f"{ctx.start:%Y%m%d%H}"
    out = _root() / "hrrr" / f"ruc_soil_{cyc}.npz"
    if not out.is_file():
        import urllib.error
        out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        try:
            ruc_hrrr_soil.fetch(cyc, out)
        except (urllib.error.HTTPError, SystemExit) as e:
            # HRRR's AWS archive starts 2014-07-30 (and early files may lack a soil record): such a start has no HRRR
            # soil analysis, so RUC starts everywhere the way it starts outside HRRR's footprint on every run (the
            # CDF-matched source soil remapped to 9 levels). Loud, and recorded in the run receipt.
            if isinstance(e, urllib.error.HTTPError) and e.code not in (403, 404):
                raise
            msg = f"no HRRR soil analysis archived for {cyc} ({e}); RUC starts from the source soil everywhere"
            print(f"{UNIT}: {msg}", flush=True)
            ctx.tables["ruc_hrrr_soil_missing"] = msg
            return ""
        print(f"{UNIT}: HRRR soil analysis {cyc} fetched -> {out} ({time.time() - t:.1f} s)", flush=True)
    return str(out)


def _auto_soil_static(ctx, v):
    """NAMGPU_RUC_SOIL_STATIC=auto: HRRR static soil classes on this grid, built once per grid on the GPU
    (namgpu.init.ruc_hrrr_soilclass, BNU top soil from NAMGPU_GEOG) next to the domain's static file."""
    if v != "auto":
        return v
    from namgpu import config
    from namgpu.init import ruc_hrrr_soilclass
    st = config.static_npz(ctx.cfg, base=True)   # soil classes do not change by month: keyed on the configfile static, not the month file (static-race fix 2026-11-09)
    out = _root() / "static" / ((Path(st).stem if st else "nam12") + "_hrrrsoil.npz")
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        cfgfile = os.environ.get("NAMGPU_CONFIGFILE")
        ruc_hrrr_soilclass.main((["--config", cfgfile] if cfgfile else []) + ["--out", str(out)]
                                + (["--geog", os.environ["NAMGPU_GEOG"]] if os.environ.get("NAMGPU_GEOG") else []))
    return str(out)


class _Seam:
    def __init__(self, ctx):
        import cupy as cp
        from gpuwm.core import ruc
        from gpuwm.core.ruc_runtime import ruc_device_sfctmp_sets
        st, g = ctx.st, ctx.grid
        self.im, self.jm = int(g.im), int(g.jm)
        if tuple(st["TSFC"].shape) != (self.jm, self.im):
            raise RuntimeError("phys-ruc-driver: called inside a physics band; run with NAMGPU_PHYS_BAND_ROWS=0")
        self.ruc = ruc
        self.params = ruc.load_ruc_parameters()
        self.zs, _ = ruc.ruc_soil_geometry(NZ)
        self.leaves, self.stages, self.arrays = ruc_device_sfctmp_sets()
        self.W = cp.asarray(_layer_weights())
        self.ktau = 0
        self.seconds = 0.0
        t0 = time.time()
        jm, im = self.jm, self.im
        interior = cp.zeros((jm, im), bool)
        interior[1:jm - 1, 1:im - 1] = True
        mask = interior & (st["XLAND"] < 1.5) & (st["SICE"] <= 0.5)
        if urban_on_noah():
            iv = st["IVGTYP"]
            urban = (iv == 13) | (iv == 31) | (iv == 32) | (iv == 33)
            print(f"{UNIT}: {int((mask & urban).sum())} urban land points stay on NAM Noah (NAMGPU_RUC_URBAN=noah)", flush=True)
            mask = mask & ~urban
        self.idx = cp.flatnonzero(mask.ravel())
        nland = int(self.idx.size)
        self.isl = self._soil_class(st)
        self.ivg = self._veg_class(st)
        if ctx.tables.get("ruc_ktau") is not None and "RUC_TSLB" in st.fields:
            # resumed from a restart set: the nine-level state and the call count come back with the state
            self.ktau = int(ctx.tables["ruc_ktau"])
            print(f"{UNIT}: RUC LSM on {nland} land points, resumed at call {self.ktau}", flush=True)
            return
        for n in STATE_2D:
            if n not in st.fields:
                st.add(n, (jm, im))
        for n in STATE_3D:
            if n not in st.fields:
                st.add(n, (NZ, jm, im))

        def g2(name):
            return st[name].ravel()[self.idx]

        smc = st["SMC"].reshape(4, -1)[:, self.idx]
        stc = st["STC"].reshape(4, -1)[:, self.idx]
        tsfc = g2("TSFC")
        tg = g2("TG")
        tslb = _profile_from_noah(cp, stc, tsfc, tg, TG_DEPTH)
        smois = _profile_from_noah(cp, smc, smc[0], smc[3], float(ZS[-1]))
        isl, ivg = self.isl, self.ivg
        msg = "start Noah layers remapped to 9 RUC levels everywhere"
        hrrr = _auto_hrrr_soil(ctx, os.environ.get("NAMGPU_RUC_HRRR_SOIL", "").strip())
        self.hrrr_points = 0
        if hrrr:
            lat = cp.asarray(np.degrees(np.asarray(g.glat_h, np.float64)).reshape(jm, im)).ravel()[self.idx]
            lon = cp.asarray(np.degrees(np.asarray(g.glon_h, np.float64)).reshape(jm, im)).ravel()[self.idx]
            ht, hw, ok = _hrrr_sample(cp, hrrr, lat, lon)
            scls = _auto_soil_static(ctx, os.environ.get("NAMGPU_RUC_SOIL_STATIC", "").strip())
            if scls:
                # HRRR static soil categories (BNU, static_build --soil hrrr) where the HRRR analysis is used, so the
                # HRRR soil moisture is read against the soil classes HRRR's own RUC cycled it with
                sot = cp.asarray(np.load(scls)["SOTYP"]).astype(cp.int32).reshape(jm, im).ravel()[self.idx]
                use = ok & (sot > 0) & (sot != 14) & (self.ivg != 15)   # land ice keeps its class
                self.isl = cp.where(use, sot, self.isl).astype(cp.int32)
                isl = self.isl
                print(f"{UNIT}: HRRR static soil classes ({scls}) at {int(use.sum())} RUC land points, "
                      f"{int((sot[use] != self._soil_class(st)[use]).sum())} differ from the start's", flush=True)
            por = self._maxsmc_per_class()
            cap = cp.asarray(por)[cp.clip(isl, 0, por.size - 1)]
            hw = cp.minimum(hw, cap[None, :])
            okb = ok[None, :]
            tslb = cp.where(okb, ht, tslb).astype(cp.float32)
            tslb[0] = tsfc                      # level 0 is the skin: the model's own TSFC
            smois = cp.where(okb, hw, smois).astype(cp.float32)
            self.hrrr_points = int(ok.sum())
            top_before = float(cp.mean(_profile_from_noah(cp, smc, smc[0], smc[3], 3.0)[0][ok]))
            top_after = float(cp.mean(smois[0][ok]))
            msg = (f"HRRR nine-level soil analysis at {self.hrrr_points} of {nland} RUC land points "
                   f"({hrrr}; top-level soil moisture mean there {top_before:.3f} start -> {top_after:.3f} HRRR), "
                   "the rest remapped from the start")
            print(f"{UNIT}: RUC land lon {float(lon.min()):.1f}..{float(lon.max()):.1f} "
                  f"lat {float(lat.min()):.1f}..{float(lat.max()):.1f}", flush=True)
        xice = g2("SICE")
        cold = ruc.ruc_initialize_cold_start(cp.asnumpy(tslb), cp.asnumpy(smois), cp.asnumpy(isl),
                                             cp.asnumpy(ivg), cp.asnumpy(xice), mminlu=MMINLU,
                                             parameters=self.params)
        self._scatter3(st["RUC_TSLB"], tslb)
        self._scatter3(st["RUC_SMOIS"], smois)
        self._scatter3(st["RUC_SH2O"], cp.asarray(np.asarray(cold.sh2o, np.float32)))
        self._scatter3(st["RUC_SMFR3D"], cp.asarray(np.asarray(cold.smfr3d, np.float32)))
        self._scatter(st["RUC_MAVAIL"], cp.asarray(np.asarray(cold.mavail, np.float32)))
        self._scatter(st["RUC_ZNT"], cp.asarray(np.asarray(cold.znt, np.float32)))
        self._scatter(st["RUC_Z0"], cp.asarray(np.asarray(cold.znt, np.float32)))
        self._scatter(st["RUC_SOILT1"], tsfc)
        self._scatter(st["RUC_TSNAV"], cp.minimum(tsfc, 273.15).astype(cp.float32))
        qsh = g2("QSH")
        self._scatter(st["RUC_QVG"], qsh)
        self._scatter(st["RUC_QSFC"], (qsh / (1 + qsh)).astype(cp.float32))
        self._scatter(st["RUC_RHOSNF"], cp.full(nland, -1000.0, cp.float32))
        self._to_noah_layers(st)
        ctx.tables["ruc_ktau"] = 0
        print(f"{UNIT}: RUC LSM on {nland} land points (water and sea ice stay on NAM Noah); {msg}; "
              f"init {time.time() - t0:.1f} s", flush=True)

    def _maxsmc_per_class(self):
        """Porosity by soil class index (0..N), from the RUC soil table the column physics reads."""
        v = np.asarray(self.params.soil.column("MAXSMC"), np.float32)      # STAS-RUC, classes 1..19
        return np.concatenate([[1.0], v]).astype(np.float32)                # index 0 unused (classes are 1-based)

    def _soil_class(self, st):
        import cupy as cp
        s = st["ISLTYP"].ravel()[self.idx].astype(cp.int32)
        return cp.where((s == 14) | (s <= 0), 6, s).astype(cp.int32)    # water class on a land point -> loam

    def _veg_class(self, st):
        import cupy as cp
        v = st["IVGTYP"].ravel()[self.idx].astype(cp.int32)
        return cp.where((v <= 0) | (v == 17), 10, v).astype(cp.int32)   # water class on a land point -> grassland

    def _scatter(self, dst, val):
        dst.ravel()[self.idx] = val

    def _scatter3(self, dst, val):
        dst.reshape(dst.shape[0], -1)[:, self.idx] = val

    def _to_noah_layers(self, st):
        for src, dst in (("RUC_SMOIS", "SMC"), ("RUC_TSLB", "STC"), ("RUC_SH2O", "SH2O")):
            prof = st[src].reshape(NZ, -1)[:, self.idx]
            self._scatter3(st[dst], self.W @ prof)

    def step(self, ctx):
        import cupy as cp
        st = ctx.st
        f32 = cp.float32
        t0 = time.perf_counter()
        self.ktau += 1
        ctx.tables["ruc_ktau"] = self.ktau
        nphs = int(getattr(ctx.cfg, "nphs", 2)) if ctx.cfg is not None else 2
        dt = float(np.float32(nphs) * np.float32(ctx.dt))
        lm = int(ctx.grid.lm)
        idx = self.idx

        def g2(name):
            return st[name].ravel()[idx]

        def low(name):
            return st[name][lm - 1].ravel()[idx]

        def p3(name):
            return st[name].reshape(NZ, -1)[:, idx]

        q = low("Q")
        qv = (q / (1.0 - q)).astype(f32)
        rho = low("RR")
        chs = g2("CHS")
        mav = g2("RUC_MAVAIL")
        cpm = (CP_D * (1.0 + 0.8 * qv)).astype(f32)
        rain = cp.maximum(g2("RAINBL"), 0.0).astype(f32)
        sr = cp.clip(g2("SR"), 0.0, 1.0).astype(f32)
        vgf = g2("VGFRCK")
        v = dict(z3d=low("DZ"), p8w=low("PHMID"), t3d=low("T"), qv3d=qv, qc3d=low("QC"), rho3d=rho,
                 rainbl=rain, frzfrac=sr, glw=g2("RLW_DN_SFC"), gsw=g2("RSW_NET_SFC"), chs=chs,
                 flqc=rho * chs * mav, flhc=cpm * rho * chs,
                 albbck=g2("ALBASE"), xland=g2("XLAND"), xice=g2("SICE"), tbot=g2("TG"),
                 shdmin=vgf, shdmax=vgf, vegfra=vgf,
                 rainncv=rain, snowncv=rain * sr, graupelncv=cp.zeros_like(rain),
                 lakemask=cp.zeros_like(rain))
        v.update(snow=g2("SNOW"), snowh=g2("SNOWH"), snowc=g2("SNOWC"), canwat=g2("RUC_CANWAT"),
                 snoalb=g2("MXSNAL"), alb=g2("ALBEDO"), emiss=g2("EPSR"), lai=g2("RUC_LAI"), mavail=mav,
                 sfcexc=g2("RUC_SFCEXC"), z0=g2("RUC_Z0"), znt=g2("RUC_ZNT"), soilt=g2("TSFC"),
                 hfx=g2("TWBS"), qfx=g2("QWBS"), lh=g2("ELFLX"), sfcevp=g2("RUC_SFCEVP"),
                 sfcrunoff=g2("SSROFF"), udrunoff=g2("BGROFF"), acrunoff=g2("RUC_ACRUNOFF"),
                 grdflx=g2("GRNFLX"), acsnow=g2("ACSNOW"), snom=g2("ACSNOM"), qvg=g2("RUC_QVG"),
                 qcg=g2("RUC_QCG"), dew=g2("RUC_DEW"), qsfc=g2("RUC_QSFC"), qsg=g2("RUC_QSG"),
                 chklowq=g2("CHKLOWQ"), soilt1=g2("RUC_SOILT1"), tsnav=g2("RUC_TSNAV"),
                 smavail=g2("SMSTAV"), smmax=g2("SMSTOT"), rhosnf=g2("RUC_RHOSNF"),
                 precipfr=g2("RUC_PRECIPFR"), snowfallac=g2("RUC_SNOWFALLAC"))
        v.update(soilmois=p3("RUC_SMOIS"), sh2o=p3("RUC_SH2O"), tso=p3("RUC_TSLB"), smfr3d=p3("RUC_SMFR3D"),
                 keepfr3dflag=p3("RUC_KEEPFR"))
        v = {k: cp.ascontiguousarray(a, dtype=f32) for k, a in v.items()}
        qsh_old = g2("QSH")
        cqs2 = g2("CQS2")
        r = self.ruc.ruc_land_surface_step(
            v, dt=dt, ktau=self.ktau, zs=self.zs, ivgtyp=self.ivg, isltyp=self.isl, myj=False, em_core=1,
            lakemodel=1, frpcpn=True, rdlai2d=False, mosaic_lu=0, mosaic_soil=0, xice_threshold=0.5,
            ilnb=1, ilnb_chain=False, c1sn=0.026, c2sn=21.0, mminlu=MMINLU, parameters=self.params,
            leaves=self.leaves, stages=self.stages, arrays=self.arrays)

        def A(name):
            return cp.asarray(getattr(r, name), dtype=f32)

        for dst, name in (("TSFC", "soilt"), ("TWBS", "hfx"), ("QWBS", "qfx"), ("ELFLX", "lh"),
                          ("GRNFLX", "grdflx"), ("SNOW", "snow"), ("SNOWH", "snowh"), ("SNOWC", "snowc"),
                          ("ALBEDO", "alb"), ("EPSR", "emiss"), ("SMSTAV", "smavail"), ("SMSTOT", "smmax"),
                          ("SSROFF", "sfcrunoff"), ("BGROFF", "udrunoff"),
                          ("ACSNOM", "snom"), ("CHKLOWQ", "chklowq"),
                          ("RUC_CANWAT", "canwat"), ("RUC_LAI", "lai"), ("RUC_MAVAIL", "mavail"),
                          ("RUC_SFCEXC", "sfcexc"), ("RUC_Z0", "z0"), ("RUC_ZNT", "znt"),
                          ("RUC_SFCEVP", "sfcevp"), ("RUC_ACRUNOFF", "acrunoff"), ("RUC_QVG", "qvg"),
                          ("RUC_QCG", "qcg"), ("RUC_DEW", "dew"), ("RUC_QSFC", "qsfc"), ("RUC_QSG", "qsg"),
                          ("RUC_SOILT1", "soilt1"), ("RUC_TSNAV", "tsnav"), ("RUC_RHOSNF", "rhosnf"),
                          ("RUC_PRECIPFR", "precipfr"), ("RUC_SNOWFALLAC", "snowfallac"), ("QSH", "qvg")):
            self._scatter(st[dst], A(name))
        # ACSNOW (NMMB: snowfall water, kg m-2, emptied by drv-bucket-reset) as Noah accumulates it on its points
        # (phys_noah_driver.cu: ACSNOW += FFROZP PRCP DT): the step frozen precipitation RAINBL*SR (mm = kg m-2).
        # The ArWen RUC column does not accumulate it, so writing its acsnow back left ACSNOW 0 on every RUC land
        # point (nam-modern 2022-12-22 12Z history, winter-scoring lane 2026-11-09), and the hiresf WEASD
        # accumulation product read 0 over land. It was added in m before 2026-11-11, 1000 times smaller than
        # Noah's.
        self._scatter(st["ACSNOW"], (v["acsnow"] + rain * sr).astype(f32))
        for dst, name in (("RUC_SMOIS", "soilmois"), ("RUC_SH2O", "sh2o"), ("RUC_TSLB", "tso"),
                          ("RUC_SMFR3D", "smfr3d"), ("RUC_KEEPFR", "keepfr3dflag")):
            self._scatter3(st[dst], A(name))
        # CHS2/CQS2 as Noah's driver sets them (CHS2 = CQS2; CQS2 = CHS where the new QSFC exceeds the old)
        self._scatter(st["CHS2"], cqs2)
        self._scatter(st["CQS2"], cp.where(A("qsfc") > qsh_old / (1.0 + qsh_old), chs, cqs2).astype(f32))
        self._to_noah_layers(st)
        self.seconds += time.perf_counter() - t0
        if self.ktau in (1, 2, 10) or self.ktau % 360 == 0:
            print(f"{UNIT}: call {self.ktau}: host launch time {self.seconds / self.ktau:.3f} s/call", flush=True)


_SEAMS = {}       # per state object; kept out of ctx.tables so the restart walker sees only arrays and ints


def run(ctx):
    seam = _SEAMS.get(id(ctx.st))
    if seam is None:
        seam = _Seam(ctx)
        _SEAMS[id(ctx.st)] = seam
    seam.step(ctx)
