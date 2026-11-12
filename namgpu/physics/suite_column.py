"""phys-column-suite: an ArWen physics suite on the NMMB columns (NAM modernisation, 2026-11-08).

One plan unit runs a whole physics suite (a parm/physics_suites.json row, e.g. hrrrv5) on every NMMB H column,
on the GPU, through two seams that already exist:
  * namgpu.physics.arwen_adapter.ArwenColumn: NMMB state <-> ArWen column views (bottom-up, mixing ratios,
    mass-point winds; only changed values go back; interior minus one row, where NMMB TURBL writes);
  * gpuwm.core.mpas_column_suite.run_mpas_column_physics: ArWen's generic two-phase column seam, the one
    gpuwm-hex runs ArWen physics through on MPAS columns. Phase 1 = radiation, surface layer, land surface, PBL,
    drag (and cumulus when selected) as held raw tendencies; phase 2 = microphysics in place.

Order inside one call (every NPHS steps, dt_phys = NPHS*DT, NMMB's DTPHS): gather -> per tile: phase 1, apply
its tendencies (th, qv, the six masses, the scalars it forces, u, v), phase 2 -> scatter. NMMB's own physics
applies its turbulence increments before its microphysics the same way.

Tiles. The 12 km grid is 796,590 columns. ArWen's MYNN keeps ~67 KB of per-call work arrays per column in the
seam state's scratch dict; equal-size tiles (NAMGPU_SUITE_TILE columns, default 100,000) share the first tile's
work arrays (NAMGPU_SUITE_SHARE=0 turns it off), so that memory is one tile's, not the domain's. What a scheme
keeps between calls (soil, snow, qke, radiation rates, TEMPO aerosols, accumulators) lives in the driver fields
and tendencies, per tile.

Surface hand-back (each call, to the NMMB names the history and products read): TSHLTR (2 m theta), QSHLTR
(2 m specific humidity), PSHLTR, U10/V10 (H points, grid-relative; io-wind-rotate rotates them as before),
TH10/Q10, TSKIN, THS, PBLH, USTAR, TWBS, QWBS, SNO, SI, SNOWC, Z0, RLWIN, RSWIN, ALBEDO, EPSR; precipitation into
PREC, ACPREC, ACPREC_TOT (m), SR, ACSNOW; REFL_10CM (TEMPO's own) on the call before each history write.
CW/F_ICE/F_RAIN are kept as the condensate total and fractions, so history and post read the TEMPO species.

Not in the TDFI: while the filter runs (ctx.filter_active) the unit does nothing (adiabatic filter), because
the filter rewinds the NMMB state but not the schemes' own column state.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import numpy as np

UNIT_ID = "phys-column-suite"
G = np.float32(9.80665)
KAPPA = np.float32(287.04 / 1004.6)
RUC_DEPTHS = np.array([0.0, 0.01, 0.04, 0.10, 0.30, 0.60, 1.00, 1.60, 3.00])   # m, WRF RUC soil levels
NOAH_MID = np.array([0.05, 0.25, 0.70, 1.50])                                   # NAM Noah layer centres (m)
TG_DEPTH = 8.0                    # NMMB Noah ZBOT: TG is the deep soil temperature at 8 m
LS_STATS = {"VARLS": "HSTDV", "CONLS": "HCNVX", "OA1LS": "HASYW", "OA2LS": "HASYS", "OA3LS": "HASYSW",
            "OA4LS": "HASYNW", "OL1LS": "HLENW", "OL2LS": "HLENS", "OL3LS": "HLENSW", "OL4LS": "HLENNW"}
SS_STATS = ("VARSS", "CONSS", "OA1SS", "OA2SS", "OA3SS", "OA4SS", "OL1SS", "OL2SS", "OL3SS", "OL4SS")
EXTRA_SLOT = {"nc": 8, "ng": 9, "volg": 10, "nwfa": 11, "nifa": 12}   # TEMPO water slots past Thompson's 7


def to_ruc(noah, top=None, deep=None):
    """Noah's four layers onto RUC's nine levels: linear in depth through the layer centres; for temperature the
    skin is the value at 0 m, so the 1 and 4 cm levels lie between the skin and the 5 cm layer centre (as WRF's
    init_soil_3_real and the lsm-ruc lane build RUC levels from a Noah source); for moisture the first layer holds
    up to 5 cm; below 1.5 m toward the deep temperature at TG_DEPTH.

    Defect this replaces (2026-11-09, dump of the first call): the 1 and 4 cm levels took the 0-10 cm layer value
    outright, an 11 K step in the top centimetre under snow on the cold Plains; RUC's ground heat flux was ~700
    W/m2 into the skin, the snow skin warmed at night (sensible heat 150 W/m2 upward), and 2 m T and Td ran warm
    and moist (+4.5/+5.6 F Td2 bias against ASOS at +6/+12 h)."""
    z = NOAH_MID
    out = np.empty((9, noah.shape[1]), np.float32)
    for k, d in enumerate(RUC_DEPTHS):
        if top is not None and d == 0.0:
            out[k] = top
        elif top is not None and d < z[0]:
            w = d / z[0]
            out[k] = (1.0 - w) * top + w * noah[0]
        elif d <= z[0]:
            out[k] = noah[0]
        elif d >= z[-1]:
            if deep is not None:
                w = (d - z[-1]) / (TG_DEPTH - z[-1])
                out[k] = (1.0 - w) * noah[-1] + w * deep
            else:
                out[k] = noah[-1]
        else:
            i = int(np.searchsorted(z, d)) - 1
            w = (d - z[i]) / (z[i + 1] - z[i])
            out[k] = (1.0 - w) * noah[i] + w * noah[i + 1]
    return out


class SuiteColumn:
    """Built on the first call; one per run (ctx.tables['suite_column'])."""

    def __init__(self, ctx):
        import cupy as cp
        from gpuwm.core import mpas_column_suite as seam
        from namgpu.physics import suites
        from namgpu.physics.arwen_adapter import ArwenColumn

        t0 = time.time()
        self.cp, self.ctx = cp, ctx
        cfg, g, st = ctx.cfg, ctx.grid, ctx.st
        self.name = suites.selected(cfg.raw)
        self.row = suites.row(self.name)
        self.sel = dict(self.row["arwen_selection"])
        self.im, self.jm, self.nz = int(g.im), int(g.jm), int(g.lm)
        self.N = self.im * self.jm
        self.dt = float(np.float32(cfg.dt) * np.float32(cfg.nphs))
        rad_s = float(cfg.dt) * float(max(int(cfg.nrads), int(cfg.nphs)))
        self.rad_s = self.dt * max(1, round(rad_s / self.dt))
        self.receipt = {"suite": self.name, "selection": self.sel, "dt_phys": self.dt, "radiation_s": self.rad_s,
                        "declared": list(self.row.get("declared", ()))}
        self.col = ArwenColumn(
            ctx, read=("p", "p8w", "th", "pii", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr",
                       "u", "v", "w8w", "z8w", "alt"),
            write=("th", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "u", "v"), name=self.name)
        tile = int(os.environ.get("NAMGPU_SUITE_TILE", "100000"))
        nt = max(1, -(-self.N // tile))
        size = -(-self.N // nt)
        self.tiles = [(a, min(self.N, a + size)) for a in range(0, self.N, size)]
        self.share = os.environ.get("NAMGPU_SUITE_SHARE", "1") != "0"
        self.receipt["tiles"] = {"count": len(self.tiles), "columns": size, "shared_work": self.share}
        # ---- per-column inputs the seams are built from (host, once, at the first call) ----
        def flat(a):
            return cp.asnumpy(a).reshape(-1)
        lat = np.degrees(flat(st["GLAT"])).astype(np.float32)
        lon = np.degrees(flat(st["GLON"])).astype(np.float32)
        lon = np.where(lon > 180.0, lon - 360.0, lon).astype(np.float32)
        hgt = (flat(st["FIS"]) / G).astype(np.float32)
        dxh = np.asarray(g.dxh, np.float64)[:, None] * np.ones((1, self.im))
        area = (dxh * float(g.dyh)).reshape(-1)
        dxc = np.sqrt(area).astype(np.float32)
        sm = flat(st["SM"])
        sice = flat(st["SICE"])
        water = (sm > 0.5) & (sice < 0.5)
        self.water = water
        tsk = flat(st["TSKIN"]).astype(np.float32)
        tg = flat(st["TG"]).astype(np.float32)
        surf = dict(
            landmask=(~water).astype(np.float32), xland=np.where(water, 2.0, 1.0).astype(np.float32),
            tsk=tsk, tmn=tg, sst=flat(st["SST"]).astype(np.float32), xice=sice.astype(np.float32),
            ivgtyp=flat(st["IVGTYP"]).astype(np.int32), isltyp=flat(st["ISLTYP"]).astype(np.int32),
            vegfra=(100.0 * flat(st["VEGFRC"])).astype(np.float32), snow=flat(st["SNO"]).astype(np.float32),
            snow_depth=(flat(st["SI"]) * 1.0e-3).astype(np.float32),
            canwat=(1.0e3 * flat(st["CMC"])).astype(np.float32))
        stc = cp.asnumpy(st["STC"]).reshape(4, -1)
        smc = cp.asnumpy(st["SMC"]).reshape(4, -1)
        sh2o = cp.asnumpy(st["SH2O"]).reshape(4, -1)
        surf["soil_temperature"] = to_ruc(stc, tsk, tg)
        surf["soil_moisture"] = to_ruc(smc)
        surf["liquid_moisture"] = to_ruc(sh2o)
        hrrr = os.environ.get("NAMGPU_RUC_HRRR_SOIL", "").strip()
        if hrrr and int(self.sel.get("sf_surface_physics", 0)) == 3:
            self._hrrr_soil(surf, hrrr, lat, lon, water | (sice >= 0.5), tsk)
        else:
            self.receipt["ruc_soil_start"] = "start's four Noah layers remapped to nine RUC levels (TG at 8 m)"
        for k in ("soil_moisture", "liquid_moisture"):
            surf[k][:, water] = 1.0                     # WRF's value over water
        stats = {k: flat(st[v]).astype(np.float32) for k, v in LS_STATS.items()}
        ss = os.environ.get("NAMGPU_GSL_SS")
        if ss:
            z = np.load(ss)
            for k in SS_STATS:
                stats[k] = np.asarray(z[k], np.float32).reshape(-1)
            self.receipt["gsl_small_scale"] = ss
        else:
            # The twenty RRFS ugwp_oro_data statistics (large and small scale) built for this grid by the gwd-ugwp
            # lane's init_ugwp_oro (Rust, every core, cached per grid). It refuses (no zero-statistics fallback)
            # when its binary or terrain is missing.
            from namgpu.init import init_ugwp_oro as oro
            a20 = oro.build(g)
            for k, v in enumerate(oro.VARIABLES):
                stats[v.upper().replace("VAR2D", "VAR")] = np.ascontiguousarray(a20[k], np.float32).reshape(-1)
            self.receipt["gsl_small_scale"] = f"init_ugwp_oro ({oro.METHOD}): large + small scale, 20 statistics"
        zi = np.asarray(self._nominal_interfaces(), np.float64)
        start = ctx.start if isinstance(ctx.start, datetime) else datetime(2000, 1, 1)
        self.seams = []
        shared = None
        size0 = self.tiles[0][1] - self.tiles[0][0]
        for a, b in self.tiles:
            sl = slice(a, b)
            s = seam.run_mpas_column_physics(
                physics=self.sel, n_levels=self.nz, n_columns=b - a, dt=self.dt,
                radiation_seconds=self.rad_s, surface_pbl_seconds=self.dt, start_time=start,
                latitude_deg=lat[sl], longitude_deg=lon[sl], terrain_height_m=hgt[sl],
                z_interface_nominal_m=zi, p_top_pa=float(g.pt), dx_m=float(np.median(dxc)),
                dx_column_m=dxc[sl], cell_area_m2=area[sl],
                surface={k: (v[:, sl] if v.ndim == 2 else v[sl]) for k, v in surf.items()},
                statics={k: v[sl] for k, v in stats.items()})
            if self.share and shared is not None and (b - a) == size0:
                s._state._scratch = _SharedScratch(shared, s._state._scratch)
            self.seams.append(s)
            if shared is None:
                shared = s._state._scratch      # the first tile fills it on its first call
        self.species = self.seams[0].species
        self.receipt["species"] = list(self.species)
        indx = ctx.tables.get("indx") or {}
        wstart = int(indx.get("INDX_WATER_START", 4))
        self.extra = {n: wstart - 1 + p - 1 for n, p in EXTRA_SLOT.items() if n in self.species}
        self.first = True
        self.calls = 0
        self.receipt["build_s"] = round(time.time() - t0, 1)

    # ------------------------------------------------------------------ helpers
    def _hrrr_soil(self, surf, path, lat, lon, notland, tsk):
        """Every land point inside HRRR's grid takes HRRR's own nine-level TSOIL/SOILW (bilinear over HRRR land
        neighbours; level 0 stays the model skin), moisture capped at the RUC porosity of the point's class; with
        NAMGPU_RUC_SOIL_STATIC, those points also take HRRR's static soil classes (BNU), so HRRR's moisture is read
        against the classes HRRR cycled it with. The lsm-ruc lane's start (namgpu.init.ruc_hrrr_soil)."""
        from gpuwm.core import ruc as _ruc
        ht, hw, ok = _hrrr_sample(path, lat.astype(np.float64), lon.astype(np.float64))
        ok &= ~notland
        isl = surf["isltyp"].copy()
        scls = os.environ.get("NAMGPU_RUC_SOIL_STATIC", "").strip()
        if scls:
            sot = np.asarray(np.load(scls)["SOTYP"], np.int32).reshape(-1)
            use = ok & (sot > 0) & (sot != 14) & (surf["ivgtyp"] != 15)
            isl[use] = sot[use]
            surf["isltyp"] = isl
            self.receipt["ruc_soil_classes"] = f"HRRR static soil (BNU) at {int(use.sum())} points ({scls})"
        por = np.concatenate([[1.0], np.asarray(_ruc.load_ruc_parameters().soil.column("MAXSMC"), np.float32)])
        cap = por[np.clip(isl, 0, por.size - 1)]
        hw = np.minimum(hw, cap[None, :])
        before = float(surf["soil_moisture"][0][ok].mean()) if ok.any() else float("nan")
        for k, v in (("soil_temperature", ht), ("soil_moisture", hw), ("liquid_moisture", hw)):
            surf[k][:, ok] = v[:, ok]
        surf["soil_temperature"][0] = tsk                     # level 0 is the skin
        self.receipt["ruc_soil_start"] = (f"HRRR nine-level soil analysis at {int(ok.sum())} land points ({path}; "
                                          f"top-level moisture there {before:.3f} -> {float(hw[0][ok].mean()):.3f}); "
                                          "the rest remapped from the start (TG at 8 m)")

    def _nominal_interfaces(self):
        """Nominal interface heights (m) of the NMMB levels over standard sea-level pressure, bottom-up."""
        g = self.ctx.grid
        p = (np.asarray(g.sg2, np.float64) * (101325.0 - float(g.pt) - float(g.pdtop))
             + np.asarray(g.psg1, np.float64))[::-1]
        t0, l0 = 288.15, 0.0065
        z = np.where(p > 22632.1, t0 / l0 * (1.0 - (p / 101325.0) ** 0.190263),
                     11000.0 + 6341.62 * np.log(22632.1 / np.maximum(p, 1.0)))
        z[0] = 0.0
        return np.maximum.accumulate(z + np.arange(z.size) * 1e-3)

    def _flat3(self, a):
        return a.reshape(a.shape[0], self.N)

    def _tracer(self, slot):
        """TRACERS slot as a bottom-up (nz, N) view (writes go to the state)."""
        return self.ctx.st["TRACERS"][slot][::-1].reshape(self.nz, self.N)

    def _first_contact(self, col):
        """Species from the start's condensate (CW, F_ICE, F_RAIN), numbers from the masses with Thompson's own
        relations, aerosols from the Thompson-Eidhammer monthly climatology (the file WRF and RRFS read). Once."""
        cp, st = self.cp, self.ctx.st
        from gpuwm.ingest import wif_climatology as wif
        from gpuwm.ingest.handover_numbers import _thompson_number_per_kg as num
        f = self._flat3
        cw, fi, fr = f(st["CW"][::-1]), f(st["F_ICE"][::-1]), f(st["F_RAIN"][::-1])
        have = float(cp.abs(cp.stack([col.qc, col.qr, col.qi, col.qs, col.qg])).max())
        self.receipt["start_species_max"] = have
        if have == 0.0:
            liq = cw * (1.0 - fi)
            f(col.qc)[...] = liq * (1.0 - fr)
            f(col.qr)[...] = liq * fr
            f(col.qs)[...] = cw * fi
            self.receipt["species_from"] = "the start's CW, F_ICE, F_RAIN"
        res = wif.acquire_wif_climatology(requested_by="woof-nam hrrrv5 suite (TEMPO aerosols)")
        clim = wif.load_wif_climatology(res.path)
        latd = np.degrees(cp.asnumpy(st["GLAT"])).astype(np.float64)
        lond = np.degrees(cp.asnumpy(st["GLON"])).astype(np.float64)
        lond = np.where(lond > 180.0, lond - 360.0, lond)
        pb = cp.asnumpy(col.p).astype(np.float32)
        phb = (cp.asnumpy(col.z8w) * G).astype(np.float32)
        w, wrec = wif.wif_fields_for_grid(clim, latd, lond, self.ctx.start.strftime("%Y-%m-%d_%H:%M:%S"), pb, phb)
        self.receipt["aerosol_ingest"] = {k: wrec.get(k) for k in ("schema", "engine", "temporal")}
        for n in ("nwfa", "nifa"):
            self._tracer(self.extra[n])[...] = cp.asarray(np.asarray(w[n], np.float32).reshape(self.nz, self.N))
        got = float(self.ctx.st["TRACERS"][self.extra["nwfa"]].max())
        if not got > 0.0:
            raise RuntimeError("phys-column-suite: DEFECT: the nwfa climatology did not reach TRACERS (view is a copy); "
                               "TEMPO would run with zero water-friendly aerosol")
        self.receipt["aerosol"] = f"Thompson-Eidhammer monthly climatology ({res.path}); TRACERS nwfa max {got:.4g}"
        rho = cp.asnumpy(1.0 / col.alt)
        t = cp.asnumpy(col.th / col.pii)
        for mass, number, kind in ((col.qr, col.nr, "rain"), (col.qi, col.ni, "ice")):
            q = cp.asnumpy(mass)
            number[...] = cp.asarray(np.where(q > 1e-12, num(kind, q, rho, t), 0.0).astype(np.float32))
        qc = cp.asnumpy(col.qc)
        nwfa = cp.asnumpy(self._tracer(self.extra["nwfa"])).reshape(qc.shape)
        xl = np.broadcast_to(np.where(self.water, 2.0, 1.0).astype(np.float32).reshape(qc.shape[1:]), qc.shape)
        nc = np.where(qc > 1e-12, num("cloud", qc, rho, t, nwfa=nwfa, xland=xl), 0.0).astype(np.float32)
        self._tracer(self.extra["nc"])[...] = cp.asarray(nc.reshape(self.nz, self.N))

    def _views(self, col):
        f = self._flat3
        v = {n: f(getattr(col, n)) for n in ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")}
        for n, slot in self.extra.items():
            v[n] = self._tracer(slot)
        return {n: v[n] for n in self.species}

    def _hist_due(self) -> bool:
        c, n = self.ctx.cfg, int(self.ctx.ntimestep)
        hs = max(1, round(float(c.raw.get("minutes_history", 60)) * 60.0 / float(c.dt)))
        return ((n // hs + 1) * hs) - n <= int(c.nphs)

    # ------------------------------------------------------------------ the call
    def run(self):
        cp, st = self.cp, self.ctx.st
        col = self.col.gather()
        if self.first:
            self._first_contact(col)
        f = self._flat3
        th, p, p8w, z8w, w8w = f(col.th), f(col.p), f(col.p8w), f(col.z8w), f(col.w8w)
        pii, u, v = f(col.pii), f(col.u), f(col.v)
        rho_d = 1.0 / f(col.alt)
        spc = self._views(col)
        dt = np.float32(self.dt)
        refl_due = self._hist_due()
        rain = cp.zeros(self.N, cp.float32)
        sr = cp.zeros(self.N, cp.float32)
        snowp = cp.zeros(self.N, cp.float32)
        crain = cp.zeros(self.N, cp.float32)
        refl = cp.full((self.nz, self.N), -35.0, cp.float32) if refl_due else None
        for (a, b), s in zip(self.tiles, self.seams):
            sl = (slice(None), slice(a, b))
            # The seam takes C-contiguous (nz, ncol) arrays: a column tile of an (nz, N) view is strided, and the
            # TRACERS views run top-down, so each tile works on contiguous copies that go back after phase 2.
            tc = cp.ascontiguousarray
            th_t, u_t, v_t = tc(th[sl]), tc(u[sl]), tc(v[sl])
            p_t, p8_t, z_t, w_t, r_t, pi_t = (tc(x[sl]) for x in (p, p8w, z8w, w8w, rho_d, pii))
            sp = {n: tc(x[sl]) for n, x in spc.items()}
            drv = s._driver
            rc0 = getattr(drv, "rainc", None)
            rc0 = None if rc0 is None else rc0.reshape(-1).copy()
            t1 = s.run_phase1(dt=self.dt, u=u_t, v=v_t, theta=th_t, pressure=p_t,
                              pressure_interface=p8_t, z_interface=z_t, w=w_t, rho_dry=r_t,
                              species=sp, exner=pi_t)
            rc1 = getattr(drv, "rainc", None)
            if rc1 is not None:                 # cumulus (GF) rain this call, mm: the bucket's change
                crain[a:b] = rc1.reshape(-1) - (rc0 if rc0 is not None else 0.0)
            th_t += dt * t1.dtheta
            u_t += dt * t1.du
            v_t += dt * t1.dv
            for n, r in (("qv", t1.dqv), ("qc", t1.dqc), ("qr", t1.dqr), ("qi", t1.dqi), ("qs", t1.dqs),
                         ("qg", t1.dqg)):
                if n in sp:
                    sp[n] += dt * r
            for n, r in (t1.scalar_rates or {}).items():
                sp[n] += dt * r
            for n in sp:
                cp.maximum(sp[n], 0.0, out=sp[n])
            self._surface(s, a, b)
            rec = s.run_phase2(theta=th_t, pressure=p_t, z_interface=z_t, w=w_t, rho_dry=r_t,
                               species=sp, refl_10cm_due=refl_due)
            th[sl] = th_t
            u[sl] = u_t
            v[sl] = v_t
            for n, x in sp.items():
                spc[n][sl] = x
            rain[a:b] = rec["rainncv"]
            if "sr" in rec:
                sr[a:b] = rec["sr"]
            for k in ("snowncv", "graupelncv"):
                if k in rec:
                    snowp[a:b] += rec[k]
            if refl_due:
                refl[:, a:b] = rec["refl_10cm"]
        col.scatter()
        self._condensate(spc)
        self._precip(rain + crain, sr, snowp, crain)
        if refl is not None:
            st.add("REFL_10CM", (self.nz, self.jm, self.im))[...] = refl[::-1].reshape(self.nz, self.jm, self.im)
        self.first = False
        self.calls += 1
        dump = os.environ.get("NAMGPU_SUITE_DUMP")
        if dump and self.calls in {int(c) for c in os.environ.get("NAMGPU_SUITE_DUMP_CALLS", "1,10").split(",")}:
            self._dump(f"{dump}-call{self.calls}.npz")
        if self.calls in (1, 2) or self.calls % 72 == 0:     # host bookkeeping only, no device sync
            mp = cp.get_default_memory_pool()
            print(f"phys-column-suite: call {self.calls} (step {self.ctx.ntimestep}): pool used "
                  f"{mp.used_bytes() / 2**30:.2f} GiB, held {mp.total_bytes() / 2**30:.2f} GiB", flush=True)

    def _dump(self, path):
        """Diagnostic only (NAMGPU_SUITE_DUMP): every 1-D/2-D driver field of every tile, one host copy, one file."""
        cp = self.cp
        out = {}
        for k in sorted(self.seams[0]._driver.fields):
            parts = []
            for s in self.seams:
                v = s._driver.fields.get(k)
                if v is None or not hasattr(v, "shape") or v.size == 0:
                    break
                parts.append(cp.asnumpy(v).reshape(v.shape[0], -1) if v.ndim == 3 else cp.asnumpy(v).reshape(1, -1))
            else:
                try:
                    out[k] = np.concatenate(parts, axis=1)
                except ValueError:
                    pass
        out["_lat"] = np.degrees(cp.asnumpy(self.ctx.st["GLAT"])).reshape(-1)
        out["_lon"] = np.degrees(cp.asnumpy(self.ctx.st["GLON"])).reshape(-1)
        np.savez(path, **out)
        print(f"phys-column-suite: dumped {len(out)} driver fields to {path}", flush=True)

    def _put2(self, name, src, a, b):
        st = self.ctx.st
        if name not in st.fields:
            st.add(name, (self.jm, self.im))
        st[name].reshape(-1)[a:b] = src.reshape(-1)

    def _surface(self, s, a, b):
        cp = self.cp
        f = s._driver.fields
        q2 = f["q2"].reshape(-1)
        psfc = f["psfc"].reshape(-1)
        qs2 = q2 / (cp.float32(1.0) + q2)
        for nm, src in (("TSHLTR", f["th2"]), ("QSHLTR", qs2), ("PSHLTR", psfc), ("U10", f["u10"]),
                        ("V10", f["v10"]), ("TH10", f["th2"]), ("Q10", qs2)):
            self._put2(nm, src, a, b)
        tsk = f["tsk"].reshape(-1)
        self._put2("TSKIN", tsk, a, b)
        self._put2("THS", tsk * (cp.float32(1.0e5) / psfc) ** KAPPA, a, b)
        for nm, key in (("TWBS", "hfx"), ("QWBS", "lh")):    # NMMB: positive downward; ArWen: positive up
            if key in f:
                self._put2(nm, -f[key].reshape(-1), a, b)
        for nm, key in (("PBLH", "pblh"), ("USTAR", "ust"), ("SNO", "snow"),
                        ("SNOWC", "snowc"), ("Z0", "znt"), ("RLWIN", "glw"), ("RSWIN", "swdown"),
                        ("ALBEDO", "albedo"), ("EPSR", "emiss")):
            if key in f:
                self._put2(nm, f[key], a, b)
        if "snowh" in f:
            self._put2("SI", f["snowh"].reshape(-1) * cp.float32(1000.0), a, b)

    def _condensate(self, spc):
        """CW = total condensate, F_ICE = frozen share, F_RAIN = rain share of the liquid (what history reads)."""
        cp, st = self.cp, self.ctx.st
        qc, qr, qi, qs, qg = (spc[n] for n in ("qc", "qr", "qi", "qs", "qg"))
        liq = qc + qr
        ice = qi + qs + qg
        cw = liq + ice
        eps = cp.float32(1.0e-12)
        f = self._flat3
        f(st["CW"][::-1])[...] = cw
        f(st["F_ICE"][::-1])[...] = cp.where(cw > eps, ice / cp.maximum(cw, eps), 0.0)
        f(st["F_RAIN"][::-1])[...] = cp.where(liq > eps, qr / cp.maximum(liq, eps), 0.0)

    def _precip(self, rain_mm, sr, snow_mm, conv_mm=None):
        st = self.ctx.st
        m = (rain_mm * np.float32(1.0e-3)).reshape(self.jm, self.im)
        if conv_mm is not None and self.sel.get("cu_physics"):
            c = (conv_mm * np.float32(1.0e-3)).reshape(self.jm, self.im)
            for nm in ("CUPREC", "CUPREC_TOT"):
                if nm in st.fields:
                    st[nm][...] += c
        for nm in ("PREC", "ACPREC", "ACPREC_TOT", "SR", "ACSNOW"):
            if nm not in st.fields:
                st.add(nm, (self.jm, self.im))
        st["PREC"][...] = m
        st["ACPREC"][...] += m
        st["ACPREC_TOT"][...] += m
        st["SR"][...] = sr.reshape(self.jm, self.im)
        st["ACSNOW"][...] += snow_mm.reshape(self.jm, self.im)     # kg m-2, as Noah's ACSNOW


def _hrrr_sample(path, lat_deg, lon_deg):
    """HRRR nine-level TSOIL/SOILW at the points: bilinear over HRRR land neighbours on HRRR's Lambert grid.
    Returns (t [9, N], w [9, N], ok [N]). The lsm-ruc lane's sampler (phys_ruc_driver._hrrr_sample), host side, once."""
    z = np.load(path)
    R = float(z["radius"])
    phi1 = np.radians(float(z["latin1"]))
    lov = float(z["lov"])
    n = np.sin(phi1)
    F = np.cos(phi1) * np.tan(np.pi / 4 + phi1 / 2) ** n / n

    def xy(la, lo):
        rho = R * F / np.tan(np.pi / 4 + np.radians(la) / 2) ** n
        th = n * np.radians(((lo - lov + 180.0) % 360.0) - 180.0)
        return rho * np.sin(th), -rho * np.cos(th)

    x0, y0 = xy(float(z["lat1"]), float(z["lon1"]))
    x, y = xy(lat_deg, lon_deg)
    fi, fj = (x - x0) / float(z["dx"]), (y - y0) / float(z["dy"])
    land = np.asarray(z["land"]) > 0.5
    ny, nx = land.shape
    i0, j0 = np.floor(fi).astype(np.int64), np.floor(fj).astype(np.int64)
    inside = (i0 >= 0) & (j0 >= 0) & (i0 < nx - 1) & (j0 < ny - 1)
    i0c, j0c = np.clip(i0, 0, nx - 2), np.clip(j0, 0, ny - 2)
    ax, ay = (fi - i0c).astype(np.float32), (fj - j0c).astype(np.float32)
    tsoil, soilw = np.asarray(z["tsoil"]), np.asarray(z["soilw"])
    wsum = np.zeros(fi.shape, np.float32)
    tacc = np.zeros((9,) + fi.shape, np.float32)
    wacc = np.zeros_like(tacc)
    for dj, di, wgt in ((0, 0, (1 - ax) * (1 - ay)), (0, 1, ax * (1 - ay)), (1, 0, (1 - ax) * ay), (1, 1, ax * ay)):
        jj, ii = j0c + dj, i0c + di
        m = land[jj, ii].astype(np.float32) * wgt
        wsum += m
        tacc += tsoil[:, jj, ii] * m
        wacc += soilw[:, jj, ii] * m
    ok = inside & (wsum > 1e-3)
    d = np.where(ok, wsum, 1.0)
    return (tacc / d).astype(np.float32), (wacc / d).astype(np.float32), ok


class _SharedScratch(dict):
    """A tile's scratch dict that hands out the first tile's per-call work arrays (MYNN's *_work, *_stage,
    *_level2, ... buffers, rewritten on every call) and keeps everything else (the microphysics surface
    accumulators mp_rainnc/..., anything a scheme carries between calls) the tile's own."""

    OWN_PREFIXES = ("mp_",)

    def __init__(self, shared, own):
        super().__init__(own)
        self._shared = shared

    def get(self, key, default=None):
        if key in self:
            return dict.__getitem__(self, key)
        if not key.startswith(self.OWN_PREFIXES) and key in self._shared:
            return self._shared[key]
        return default


def run(ctx):
    if getattr(ctx, "filter_active", False):
        return
    sc = ctx.tables.get("suite_column")
    if sc is None:
        sc = ctx.tables["suite_column"] = SuiteColumn(ctx)
        print(f"phys-column-suite: {sc.name}: {sc.receipt['tiles']} built in {sc.receipt['build_s']} s; "
              f"species {sc.receipt['species']}; small-scale drag: {sc.receipt['gsl_small_scale']}", flush=True)
    if ctx.stream is not None:
        with ctx.stream:
            sc.run()
    else:
        sc.run()
