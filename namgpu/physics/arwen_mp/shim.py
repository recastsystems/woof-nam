"""ShimState: a row band of NMMB (B-grid, top-down) columns presented as the ArWen DomainState the gpuwm
microphysics adapters read (bottom-up, WRF names). Modernise lane mp-thompson-p3.

ArWen fields set here (all bottom-up, k = 0 the lowest layer, shape (nz, nb, nx) for a band of nb rows):
  p          full pressure at mass levels (NMMB P_PHY = PHMID)
  thb, thp   thb = 0 (1-D), thp = full potential temperature (NMMB TH_PHY = T/PI_PHY); thb + thp is the full theta
  phb, php   phb = 0 (1-D, nz+1), php = g * z at the layer interfaces (nz+1), z built from the surface height FIS/g
             and the NMMB layer depths DZ8W, so diff(php)/g is the NMMB DZ and z[0] is the terrain height
  alt        1/RHO_PHY (NMMB moist density; WRF passes dry density, a difference below 1 %)
  w          vertical velocity at the interfaces (nz+1): NMMB W on its levels, top interface 0
  qv         mixing ratio QV = Q/(1-Q) (GSMDRIVE USE_QV)
  qc qr qi qs qg ni nr and, by scheme, nc nwfa nifa (mp 28), qir qib nc (P3): the TRACERS water slots
  th_old qv_old (P3): band views are loaded from persistent whole-domain NMMB-layout arrays P3_TH_OLD/P3_QV_OLD
The scheme is column-local, so a band computes exactly what the whole-domain call computes for its columns.
Write-back (store): the rows the band owns, B1 interior only (rows and columns 1..N-2), the GSMDRIVE tile.
"""
from __future__ import annotations

import numpy as np

_SPECIES_ATTR = {"qc": "QC", "qi": "QI", "qr": "QR", "qs": "QS", "qg": "QG", "ni": "NI", "nr": "NR",
                 "nc": "NC", "nwfa": "NWFA", "nifa": "NIFA", "qir": "QIR", "qib": "QIB"}
CONDENSATE = ("qc", "qr", "qi", "qs", "qg")


class _Driver:
    """The one-frame REFL_10CM handoff the ArWen adapters stash on (gpuwm.core.refl.stash_refl_10cm)."""

    def __init__(self, state):
        self.state = state
        self.refl_10cm = None


class ShimState:
    use_theta_m = 0
    radar_tten_forcing = None

    def __init__(self, ctx, nz: int, nb: int, nx: int, mp_physics: int):
        import cupy as cp
        from gpuwm.core import constants as c
        self._ctx = ctx
        self._cp = cp
        self._g = float(c.G)
        self.mp_physics = int(mp_physics)
        self.nz, self.ny, self.nx = nz, nb, nx
        self._scratch = {}
        self.physics = _Driver(self)
        f = np.float32
        z3 = lambda: cp.zeros((nz, nb, nx), dtype=f)  # noqa: E731
        self.p = z3()
        self.thb = cp.zeros((nz,), dtype=f)
        self.thp = z3()
        self.phb = cp.zeros((nz + 1,), dtype=f)
        self.php = cp.zeros((nz + 1, nb, nx), dtype=f)
        self.alt = z3()
        self.w = cp.zeros((nz + 1, nb, nx), dtype=f)
        self.h_diabatic = z3()
        self.effc, self.effi, self.effs = z3(), z3(), z3()
        self.ht = cp.zeros((nb, nx), dtype=f)
        tab = ctx.tables.get("indx")
        if tab is None:
            from namgpu.steps.drv_tracer_index import host_index
            tab = host_index(ctx.cfg.microphysics, ctx.cfg.raw.get("num_tracers_chem", 0))
        self._slots = {}
        for a, n in _SPECIES_ATTR.items():
            s = int(tab.get(f"INDX_{n}", 0) or 0)
            if s > 0:
                self._slots[a] = s
                setattr(self, a, z3())
            else:
                setattr(self, a, None)
        if self.mp_physics in (8, 28):          # Thompson carries snow and graupel; P3 has neither
            for a in ("qs", "qg"):
                if getattr(self, a) is None:
                    raise RuntimeError(f"phys-mp-arwen: Thompson needs a TRACERS slot for {a}")
        self.qv = z3()
        if self.mp_physics == 28:
            self.nwfa2d = cp.zeros((nb, nx), dtype=f)
            self.nifa2d = cp.zeros((nb, nx), dtype=f)
        self._p3 = self.mp_physics in (50, 51, 52, 53)
        if self._p3:
            self.th_old = z3()
            self.qv_old = z3()
            self.p3_itimestep = 0
            self.elapsed_seconds = 0.0
            if self.mp_physics == 50:
                self.nc = None
        self.rows = (0, nb)

    # --- the DomainState surface the adapters use ---
    def scratch(self, shape, name, dtype=np.float32):
        shape = tuple(int(s) for s in shape)
        key = (name, shape, np.dtype(dtype).str)
        a = self._scratch.get(key)
        if a is None:
            a = self._cp.zeros(shape, dtype=dtype)
            self._scratch[key] = a
        return a

    def existing_scratch(self, name, *_a, **_k):
        for (n, _s, _d), a in self._scratch.items():
            if n == name:
                return a
        return None

    def total_theta(self):
        return self.thb[:, None, None] + self.thp

    # --- NMMB -> shim, rows [j0, j1) ---
    def load(self, j0, j1, *, dz8w, rho_phy, p_phy, th_phy, p3_first=False, itimestep=0):
        cp = self._cp
        st = self._ctx.st
        nz = self.nz
        J = (slice(None), slice(j0, j1), slice(None))
        self.rows = (j0, j1)
        self.p[...] = p_phy[J][::-1]
        self.thp[...] = th_phy[J][::-1]
        self.alt[...] = 1.0 / rho_phy[J][::-1]
        q = st["Q"][J]
        self.qv[...] = (q / (1.0 - q))[::-1]
        if "FIS" in st.fields:
            self.ht[...] = st["FIS"][j0:j1] / self._g
        z = self.php
        z[0] = self.ht
        cp.cumsum(dz8w[J][::-1], axis=0, out=z[1:])
        z[1:] += self.ht
        z *= self._g
        w = st["W"] if "W" in st.fields else None
        if w is not None:
            if w.shape[0] == nz + 1:
                self.w[...] = w[J][::-1]
            else:
                self.w[:nz] = w[J][::-1]
                self.w[nz] = 0.0
        tr = st["TRACERS"]
        for a, s in self._slots.items():
            arr = getattr(self, a)
            if arr is not None:
                arr[...] = tr[s - 1][J][::-1]
        if self._p3:
            tho, qvo = self._p3_old()
            if p3_first:
                tho[J] = th_phy[J]
                qvo[J] = q / (1.0 - q)
            self.th_old[...] = tho[J][::-1]
            self.qv_old[...] = qvo[J][::-1]
            self.p3_itimestep = int(itimestep)
            self.elapsed_seconds = float(self._ctx.ntimestep) * float(self._ctx.dt)

    def _p3_old(self):
        st = self._ctx.st
        s3 = (self.nz,) + tuple(st["T"].shape[1:])
        return st.add("P3_TH_OLD", s3), st.add("P3_QV_OLD", s3)

    def store_aerosol(self, own):
        """After the domain-start CCN/IN fill: the profile goes into the TRACERS slots of the owned rows (every column)."""
        tr = self._ctx.st["TRACERS"]
        j0 = self.rows[0]
        a0, a1 = own
        for a in ("nwfa", "nifa"):
            s = self._slots.get(a)
            if s:
                tr[s - 1][:, a0:a1, :] = getattr(self, a)[::-1][:, a0 - j0:a1 - j0, :]

    # --- shim -> NMMB: owned rows [a0, a1), B1 interior ---
    def store(self, own, *, th_phy, diag, rainncv, rainnc):
        st = self._ctx.st
        j0 = self.rows[0]
        jm = int(st["T"].shape[1])
        a0, a1 = max(own[0], 1), min(own[1], jm - 1)
        if a1 <= a0:
            self.physics.refl_10cm = None
            return
        D = (slice(None), slice(a0, a1), slice(1, -1))            # destination (whole-domain rows)
        S = (slice(None), slice(a0 - j0, a1 - j0), slice(1, -1))  # source (band rows)
        th_phy[D] = self.thp[::-1][S]
        qv = self.qv[::-1][S]
        st["Q"][D] = qv / (1.0 + qv)
        tr = st["TRACERS"]
        cw = None
        for a, s in self._slots.items():
            arr = getattr(self, a)
            if arr is None:
                continue
            v = arr[::-1][S]
            tr[s - 1][D] = v
            if a in CONDENSATE:
                cw = v.copy() if cw is None else cw + v
        if cw is not None:
            st["CW"][D] = cw
        S2 = (slice(a0 - j0, a1 - j0), slice(1, -1))
        D2 = (slice(a0, a1), slice(1, -1))
        if diag is not None:
            for name, dst in (("rainncv", rainncv), ("rainnc", rainnc)):
                v = getattr(diag, name, None)
                if v is not None:
                    dst[D2] = v[S2]
            sr = getattr(diag, "sr", None)
            if sr is not None:
                st.add("SR", tuple(st["T"].shape[1:]))[D2] = sr[S2]
        # Effective radii for progcld8 (np3d 8): ArWen hands them over in microns (WRF driver-side contract);
        # the NMMB kernel reads metres in RE_CLOUD/RE_ICE/RE_SNOW (model layout). A zero radius (P3 has no
        # snow) keeps the Fortran initial value in the kernel.
        s3 = tuple(st["T"].shape)
        for name, arr in (("RE_CLOUD", self.effc), ("RE_ICE", self.effi), ("RE_SNOW", self.effs)):
            st.add(name, s3)[D] = arr[::-1][S] * np.float32(1.0e-6)
        refl = self.physics.refl_10cm
        self.physics.refl_10cm = None
        if refl is not None:
            st.add("REFL_10CM", tuple(st["T"].shape))[D] = refl[::-1][S]
        if self._p3:
            tho, qvo = self._p3_old()
            tho[D] = self.th_old[::-1][S]
            qvo[D] = self.qv_old[::-1][S]

    def zero_ring(self):
        """Hydrometeor species (mass and number) on the outer ring, outside the GSMDRIVE B1 tile, set to zero.
        Breakage it prevents: with spec_adv on, condensate advected into the ring has no microphysics sink and piles
        up on the outflow edge, then rains out of the first interior column (mp-tempo smoke 2026-11-08: 19 g/kg,
        19 mm in 2 h). Zero is a GFS start's boundary condensate (NPS do_clouds false). Aerosol numbers (nwfa,
        nifa) keep their profile: the ring is their inflow value."""
        tr = self._ctx.st["TRACERS"]
        for a, s in self._slots.items():
            if a in ("nwfa", "nifa"):
                continue
            t = tr[s - 1]
            t[:, 0, :] = 0.0
            t[:, -1, :] = 0.0
            t[:, :, 0] = 0.0
            t[:, :, -1] = 0.0
