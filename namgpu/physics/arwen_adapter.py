"""arwen-adapter: the ONE column adapter between the NMMB state and ArWen (gpuwm) physics schemes.

NAM modernisation (2026-11-08). Every ArWen scheme lane (MYNN, YSU, Thompson, TEMPO, P3, RUC, RRTMG/RRTMGP,
Grell-Freitas, Tiedtke, GWDO, ...) runs on NMMB through this module and nothing else. The NAM dycore stays NMMB.

Layouts
-------
NMMB (CONTRACT.md):  A[l, j, i], l = 0 the MODEL TOP; B grid, U/V at V points; Q specific humidity;
                     water species in TRACERS slots (mixing ratios); hydrostatic layer pressure
                     PLYR = SGML2*PD + PSGML1 and interfaces PHINT = SG2*PD + PSG1 (what NMMB's own TURBL uses).
ArWen (gpuwm/core):  a[k, j, i] = (nz, ny, nx), k = 0 the LOWEST layer; everything at mass points.
2-D surface fields are [j, i] in both, so they pass through untouched (``col.sfc("SM")``).

Fields a scheme can ask for (ArWen names; all float32, device-resident, allocated once and reused):
  p (Pa, layer)        p8w (Pa, nz+1 interfaces, p8w[0] = surface)   t (K)    th (K, theta)   pii (Exner)
  qv (kg/kg MIXING ratio, = Q/(1-Q))   qc qr qi qs qg (kg/kg)   ni nr (1/kg)   qke (m2/s2, NMMB Q2 = 2 TKE, the MYJ/MYNN qke)   [only the species the NMMB
  microphysics carries: fer_hires has qc qr qs; thompson has all seven]
  u v (m/s, the four-point mean of the surrounding V points; grid-relative on the rotated lat-lon grid)
  w (m/s, NMMB W at layers)   w8w (nz+1, interface mean, 0 at the surface)
  z (m ASL, layer middle)   z8w (m ASL, nz+1, z8w[0] = FIS/g)   dz8w (m, NMMB's hypsometric DZ, TURBL formula)
  rho (kg/m3 moist, WRF phy_prep rho_d*(1+qv))   alt (m3/kg, 1/rho_d)

Use (one scheme call, inside the step, no host sync, no Python loop over columns):

    col = ArwenColumn(ctx, read=("p", "p8w", "th", "pii", "qv", "qc", "z", "dz8w", "rho", "u", "v"),
                      write=("th", "qv", "qc", "u", "v"), name="mynn")      # once (allocates)
    if clock.due("pbl", ctx.ntimestep):
        col.gather()                         # NMMB -> ArWen views (+ snapshots of the writable fields)
        my_arwen_scheme(col.th, col.qv, ..., dt=clock.dt_phys("pbl"))      # in place on the views
        col.scatter()                        # only changed values go back: T, Q, water, CW/F_ICE/F_RAIN, U/V
    # or, for a scheme that returns tendencies (ArWen RTHBLTEN-style, per second, uncoupled):
        col.gather(); col.apply_tendencies(clock.dt_phys("pbl"), th=rthblten, qv=rqvblten, u=rublten, v=rvblten)

Write-back rules (namgpu/kernels/arwen_column_adapter.cu):
  * only cells whose value changed are written, so gather -> scatter with no physics is BITWISE identity
    (tests/test_arwen_adapter.py on a real NMMB state);
  * writes go to the interior minus one row (ITS_B1..ITE_B1, JTS_B1..JTE_B1), where NMMB TURBL writes; the
    outer row stays the boundary scheme's;
  * T from th (T = th*pii) if th changed, else from t if t changed; Q = qv/(1+qv);
  * if any of qc qr qi qs qg changed, CW, F_ICE, F_RAIN are re-coupled exactly as module_TURBULENCE.F90:1336-1350
    (graupel left out of CW under Ferrier, as NMMB does); F_RIMEF is left alone;
  * a mass-point wind change du goes back to the V points as the four-point mean (module_H_TO_V.F90 stencil),
    the same way NMMB returns DUDT/DVDT; a zero mean leaves the V value's bits untouched.

Time step: ``PhysicsClock`` gives each family's cadence and physics dt from the NAM config (nphs for
pbl/sfc/lsm/gwd, nprecip for mp/cu, nrads/nradl for sw/lw), and ``substeps`` splits a physics dt for a scheme
with a stability limit (e.g. an explicit sedimentation step).

Multi-card row slabs: pass ``joff``/``jmg`` (ctx.slab) and the interior test uses global rows, like every other
kernel (nmmb_common.cuh NMMB_SLAB_EDGE).
"""
from __future__ import annotations

import math

import numpy as np

from namgpu import kernels

UNIT = "arwen_column_adapter"

BITS = {n: 1 << b for b, n in enumerate((
    "p", "p8w", "t", "th", "pii", "qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr",
    "u", "v", "w", "w8w", "z", "z8w", "dz8w", "rho", "alt", "qke"))}
SPECIES = ("qc", "qr", "qi", "qs", "qg", "ni", "nr", "qke")      # TRACERS-backed fields (kernel order)
INTERFACE = {"p8w", "w8w", "z8w"}                       # nz + 1 levels
WRITABLE = {"t", "th", "qv", *SPECIES, "u", "v"}        # what scatter can return to NMMB
DERIVED_FOR = {"th": ("pii",)}                          # T = th * pii needs the gathered pii

# ArWen surface name -> NMMB 2-D field (same [j, i] layout, passed through). A lane that needs another one
# adds it here; col.sfc(nmmb_name) reads any NMMB 2-D field directly.
SURFACE = {
    "xland_sea": "SM",       # NMMB SM = 1 over water (WRF xland = 1 + SM)
    "xice": "SICE", "tsk": "TSKIN", "sst": "SST", "znt": "Z0", "ust": "USTAR", "pblh": "PBLH",
    "hfx": "TWBS", "qfx": "QWBS", "snow": "SNO", "snowh": "SI", "vegfra": "VEGFRC",
    "isltyp": "ISLTYP", "ivgtyp": "IVGTYP", "glw": "RLWIN", "swdown": "RSWIN", "coszen": "CZEN",
}


def _slots(ctx) -> dict:
    tab = getattr(ctx, "tables", {}) or {}
    if "indx" in tab:
        indx = tab["indx"]
    else:
        from namgpu.steps.drv_tracer_index import host_index  # noqa: PLC0415
        indx = host_index(ctx.cfg.microphysics, int(ctx.cfg.raw.get("num_tracers_chem", 0) or 0))
    key = {s: f"INDX_{s.upper()}" for s in SPECIES}
    key["qke"] = "INDX_Q2"                                    # NMMB Q2 = 2 TKE = MYJ/MYNN qke
    return {s: int(indx[key[s]]) - 1 for s in SPECIES}       # 0-based slot, -1 = not carried


class ArwenColumn:
    """NMMB state -> ArWen column views -> changes back. See the module docstring."""

    def __init__(self, ctx, read=(), write=(), name="adp", joff: int | None = None, jmg: int | None = None):
        import cupy as cp
        self.cp = cp
        self.ctx = ctx
        self.name = name
        g = ctx.grid
        self.im, self.jm, self.lm = int(g.im), int(g.jm), int(g.lm)
        bad = [f for f in (*read, *write) if f not in BITS]
        if bad:
            raise KeyError(f"arwen-adapter: unknown field(s) {bad}; known: {sorted(BITS)}")
        nowrite = [f for f in write if f not in WRITABLE]
        if nowrite:
            raise ValueError(f"arwen-adapter: {nowrite} cannot be written back to NMMB (dynamics or derived "
                             f"fields); writable: {sorted(WRITABLE)}")
        self.slots = _slots(ctx)
        missing = [s for s in (*read, *write) if s in SPECIES and self.slots[s] < 0]
        if missing:
            raise ValueError(f"arwen-adapter: the NMMB microphysics {ctx.cfg.microphysics!r} carries no "
                             f"{missing} (TRACERS has no slot); configure a microphysics that does "
                             f"(drv_tracer_index.MICROPHYSICS_WATER) or do not ask for them")
        self.write = tuple(dict.fromkeys(write))
        want = set(read) | set(write)
        for f in list(want):
            want.update(DERIVED_FOR.get(f, ()) if f in self.write else ())
        self.read = tuple(f for f in BITS if f in want)
        self.need = sum(BITS[f] for f in self.read)
        self.wr = sum(BITS[f] for f in self.write)
        nz, ny, nx = self.lm, self.jm, self.im
        self.a: dict[str, object] = {}
        for f in self.read:
            self.a[f] = cp.empty((nz + 1 if f in INTERFACE else nz, ny, nx), np.float32)
        self.snap = {f: cp.empty((nz, ny, nx), np.float32) for f in self.write}
        self._dummy = cp.zeros(1, np.float32)
        wind = [f for f in ("u", "v") if f in self.write]
        self._dh = {f: cp.zeros((nz, ny, nx), np.float32) for f in wind}
        self._slot_dev = cp.asarray(np.array([self.slots[s] for s in SPECIES], np.int32))
        self._hyd = cp.asarray(np.array([self._ptr(self.a.get(s)) for s in SPECIES], np.uint64))
        self._hyd_s = cp.asarray(np.array([self._ptr(self.snap.get(s)) for s in SPECIES], np.uint64))
        vc = (g.sg2, g.psg1, g.sgml2, g.psgml1)
        self._vc = tuple(cp.asarray(np.asarray(x, np.float32)) for x in vc)
        slab = getattr(ctx, "slab", None)
        # namgpu/slab.py: JOFF = jms - 1 (jms 1-based); no ctx.slab = one card (JOFF 0, JMG = grid.jm)
        self.joff = int(joff if joff is not None else (int(slab.jms) - 1 if slab is not None else 0))
        self.jmg = int(jmg if jmg is not None else (int(slab.jmg) if slab is not None else self.jm))
        mp = str(getattr(ctx.cfg, "microphysics", "")).strip()
        self.fer_mic = int(mp.startswith("fer"))
        self.couple_cw = int(all(k in ctx.st.fields for k in ("CW", "F_ICE", "F_RAIN")))

    # ArWen views as attributes: col.th, col.qv, ...
    def __getattr__(self, name):
        a = self.__dict__.get("a", {})
        if name in a:
            return a[name]
        raise AttributeError(name)

    def _ptr(self, arr):
        return int((arr if arr is not None else self._dummy).data.ptr)

    def _arr(self, f):
        return self.a.get(f, self._dummy)

    def sfc(self, name):
        """An NMMB 2-D field ([j, i], identical layout), by NMMB name or by an ArWen name in SURFACE."""
        st = self.ctx.st
        key = SURFACE.get(name, name)
        if key not in st.fields:
            raise KeyError(f"arwen-adapter: NMMB has no 2-D field {key!r} in this state")
        return st.fields[key]

    def hgt(self):
        """Terrain height (m) = FIS / g, [j, i]."""
        return self.ctx.st["FIS"] / np.float32(9.8060226)

    def gather(self):
        cp = self.cp
        st = self.ctx.st
        im, jm, lm = self.im, self.jm, self.lm
        k = kernels.get(UNIT, "adapter_gather")
        trac = st["TRACERS"]
        args = (st["T"], st["Q"], st["PD"], st["FIS"], st["U"], st["V"], st["W"], trac, self._slot_dev,
                *self._vc,
                *(self._arr(f) for f in ("p", "p8w", "t", "th", "pii", "qv")), self._hyd,
                *(self._arr(f) for f in ("u", "v", "w", "w8w", "z", "z8w", "dz8w", "rho", "alt")),
                np.uint32(self.need), np.int32(im), np.int32(jm), np.int32(lm))
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL, args)
        for f, s in self.snap.items():
            cp.copyto(s, self.a[f])
        return self

    def apply_tendencies(self, dt: float, **tend):
        """x += dt * tend[x] on the views (tendencies per second, ArWen layout), then scatter()."""
        k = kernels.get(UNIT, "adapter_add_tend")
        for f, ten in tend.items():
            if f not in self.write:
                raise ValueError(f"arwen-adapter: tendency for {f!r}, which this adapter does not write back")
            x = self.a[f]
            n = int(x.size)
            k(((n + 255) // 256,), (256,), (x, ten, np.float32(dt), np.uint64(n)))
        return self.scatter()

    def scatter(self):
        st = self.ctx.st
        im, jm, lm = self.im, self.jm, self.lm
        snap = self.snap
        d = self._dummy
        cw = st.fields.get("CW", d)
        fi = st.fields.get("F_ICE", d)
        fr = st.fields.get("F_RAIN", d)
        k = kernels.get(UNIT, "adapter_scatter")
        k(kernels.grid_cols(im, jm), kernels.BLOCK_COL,
          (st["T"], st["Q"], st["TRACERS"], self._slot_dev, cw, fi, fr,
           np.int32(self.couple_cw), np.int32(self.fer_mic),
           self._arr("th"), snap.get("th", d), self._arr("t"), snap.get("t", d), self._arr("pii"),
           self._arr("qv"), snap.get("qv", d), self._hyd, self._hyd_s,
           self._arr("u"), snap.get("u", d), self._arr("v"), snap.get("v", d),
           self._dh.get("u", d), self._dh.get("v", d),
           np.uint32(self.wr), np.int32(im), np.int32(jm), np.int32(lm), np.int32(self.joff), np.int32(self.jmg)))
        if self._dh:
            h = kernels.get(UNIT, "adapter_h2v_add")
            h(kernels.grid_2d(im, jm, lm), kernels.BLOCK_2D,
              (st["U"], self._dh.get("u", d), st["V"], self._dh.get("v", d),
               np.int32("u" in self._dh), np.int32("v" in self._dh), np.int32(im), np.int32(jm)))
        return self

    def nbytes(self) -> int:
        return sum(int(x.nbytes) for x in (*self.a.values(), *self.snap.values(), *self._dh.values()))

    def release(self):
        self.a.clear()
        self.snap.clear()
        self._dh.clear()


class PhysicsClock:
    """When each ArWen physics family runs inside the NMMB step, and with which dt.

    NMMB cadences (parm/nam_configfile; 12 km CONUS: dt 25 s, nphs 2, nprecip 2, radiation every 48 steps):
      pbl, sfc, lsm, gwd -> every nphs steps     (TURBL, CALL_TURBULENCE = MOD(NTIMESTEP,NPHS)==0)
      mp, cu             -> every nprecip steps  (CALL_PRECIP = MOD(NTIMESTEP,NPRECIP)==0)
      sw / lw            -> every nrads / nradl steps
    dt_phys = dt * nsteps, as NMMB's DTPHS. A scheme with its own cadence (ArWen bldt/cudt/radt in minutes)
    passes ``every_s`` and the clock rounds it to a whole number of NMMB steps (at least the family's).
    """

    FAMILY = {"pbl": "nphs", "sfc": "nphs", "lsm": "nphs", "gwd": "nphs",
              "mp": "nprecip", "cu": "nprecip", "sw": "nrads", "lw": "nradl"}

    def __init__(self, cfg, every_s: dict | None = None):
        self.dt = float(cfg.dt)
        self.n = {}
        for fam, key in self.FAMILY.items():
            n = max(1, int(getattr(cfg, key)))
            if every_s and fam in every_s and every_s[fam]:
                n = max(n, int(round(float(every_s[fam]) / self.dt)))
            self.n[fam] = n

    def due(self, family: str, ntimestep: int) -> bool:
        return int(ntimestep) % self.n[family] == 0

    def dt_phys(self, family: str) -> float:
        return float(np.float32(self.dt * self.n[family]))     # single precision, as DTPHS

    def substeps(self, family: str, dt_max: float) -> tuple[int, float]:
        """(n, dt_sub): split this family's physics dt into n equal pieces no longer than dt_max."""
        dtp = self.dt_phys(family)
        n = max(1, math.ceil(dtp / float(dt_max) - 1e-9))
        return n, float(np.float32(dtp / n))
