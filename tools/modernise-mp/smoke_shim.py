"""Synthetic smoke of the ArWen MP shim: a 64 x 48 x 60 NMMB-layout (top-down) moist sounding with a saturated,
rising layer, through ShimState + gpuwm.core.microphysics.apply for each scheme. Prints condensate, precip, theta
change and finiteness. Usage: python smoke_shim.py <site> <src> [schemes]"""
import sys
import time
import types

site, src = sys.argv[1], sys.argv[2]
schemes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["thompson", "thompson_aero", "p3"]
sys.path[:0] = [site, src]
import numpy as np  # noqa: E402
import cupy as cp  # noqa: E402

from namgpu.steps.drv_tracer_index import host_index  # noqa: E402
from namgpu.physics.arwen_mp.shim import ShimState  # noqa: E402
from gpuwm.core import microphysics as amp  # noqa: E402
from gpuwm.config import RunConfig  # noqa: E402

lm, jm, im = 60, 48, 64
f = np.float32
# bottom-up sounding then flip to NMMB top-down
pb = np.linspace(100000.0, 5000.0, lm).astype(f)
z = (-7000.0 * np.log(pb / 100000.0)).astype(f)
tb = np.maximum(288.0 - 0.0065 * z, 215.0).astype(f)
es = 611.2 * np.exp(17.67 * (tb - 273.15) / (tb - 29.65))
qs = 0.622 * es / (pb - es)
rh = np.where(z < 9000.0, 1.03, 0.3)        # supersaturated below 9 km: forces condensation and ice
qvb = (rh * qs).astype(f)
dzb = np.diff(np.concatenate([[0.0], z, [z[-1] + (z[-1] - z[-2])]]))[1:].astype(f)
dzb = np.gradient(z).astype(f)


def col3(v):
    return cp.asarray(np.broadcast_to(v[::-1, None, None], (lm, jm, im)).copy())


p_phy = col3(pb)
pi = (p_phy * 1e-5) ** 0.2857
t = col3(tb)
th_phy = t / pi
qv = col3(qvb)
q = qv / (1 + qv)
rho = p_phy / (287.04 * t * (1 + 0.608 * q))
dz8w = col3(dzb)

for mp in schemes:
    tab = host_index(mp, 0)
    st = types.SimpleNamespace()
    fields = {"Q": q.copy(), "FIS": cp.zeros((jm, im), dtype=f), "W": cp.full((lm, jm, im), 0.5, dtype=f),
              "TRACERS": cp.zeros((tab["NUM_TRACERS_TOTAL"], lm, jm, im), dtype=f),
              "CW": cp.zeros((lm, jm, im), dtype=f), "SR": cp.zeros((jm, im), dtype=f),
              "REFL_10CM": cp.zeros((lm, jm, im), dtype=f)}
    st.fields = fields
    st.__getitem__ = None

    class St(dict):
        @property
        def fields(self):
            return self
    S = St(fields)
    S.add = lambda n, shape, dtype="float32": S.setdefault(n, cp.zeros(shape, dtype=dtype))
    S["T"] = t
    ctx = types.SimpleNamespace(st=S, tables={"indx": tab}, cfg=types.SimpleNamespace(microphysics=mp, raw={}),
                                ntimestep=0, dt=25.0)
    mpn = {"thompson": 8, "thompson_aero": 28, "p3": 50}[mp]
    acfg = RunConfig(nx=im, ny=jm, nz=lm, dx=12000.0, dy=12000.0, ztop=20000.0, dt=50.0, run_seconds=3600.0,
                     mp_physics=mpn, moist=True)
    sh = ShimState(ctx, lm, jm, im, mpn)
    th = th_phy.copy()
    t0 = time.time()
    rainncv = cp.zeros((jm, im), dtype=f)
    rainnc = cp.zeros((jm, im), dtype=f)
    for it in range(20):
        ctx.ntimestep = it * 2
        sh.load(0, jm, dz8w=dz8w, rho_phy=rho, p_phy=p_phy, th_phy=th, p3_first=(it == 0), itimestep=it + 1)
        if it == 0:
            rec = amp.microphysics_init(sh, acfg)
            if mpn == 28:
                sh.store_aerosol((0, jm))
            print(mp, "init", rec)
        d = amp.apply(sh, acfg, 50.0, refl_10cm_due=True)
        sh.store((0, jm), th_phy=th, diag=d, rainncv=rainncv, rainnc=rainnc)
    cp.cuda.Device().synchronize()
    tr = S["TRACERS"]
    out = {k: float(tr[v - 1][:, 1:-1, 1:-1].max()) for k, v in tab.items()
           if k.startswith("INDX_") and k not in ("INDX_Q", "INDX_CW", "INDX_Q2", "INDX_WATER_START", "INDX_WATER_END") and v > 0}
    refl = sh.existing_scratch("refl_10cm")
    print(mp, f"{time.time() - t0:.1f}s", "dTH max", float(cp.abs(th - th_phy).max()),
          "rainnc max mm", float(rainnc.max()), "CW max", float(S.get("CW", cp.zeros(1)).max()) if "CW" in S else None,
          "finite", bool(cp.isfinite(tr).all() and cp.isfinite(th).all()),
          "refl max", None if refl is None else float(refl.max()), out, flush=True)
