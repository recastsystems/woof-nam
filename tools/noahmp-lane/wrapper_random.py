"""Unit DONE check (CONTRACT.md): phys-noahmp-driver runs on a full-size (835 x 954) state built from the real
static file and plausible random surface values, two calls (cold start + a step), no NaN in what it writes."""
import types, datetime, sys
import numpy as np, cupy as cp
from namgpu.physics.noahmp import phys_noahmp_driver as m

jm, im, lm = 835, 954, 3
z = np.load(sys.argv[1])
rng = np.random.default_rng(0)


class St:
    def __init__(self):
        self.fields = {}

    def __getitem__(self, k):
        return self.fields[k]

    def add(self, k, shape, dtype="float32"):
        self.fields[k] = cp.zeros(shape, dtype=dtype)
        return self.fields[k]


st = St()
land = z["LAND"] > 0.5
f = lambda a: cp.asarray(np.asarray(a, np.float32))
st.fields.update({
    "XLAND": f(np.where(land, 1.0, 2.0)), "SICE": f(np.zeros((jm, im))),
    "IVGTYP": cp.asarray(np.where(land, np.clip(z["VGTYP"], 1, 20), 17).astype(np.int32)),
    "ISLTYP": cp.asarray(np.where(land, np.clip(z["SOTYP"], 1, 16), 14).astype(np.int32)),
    "VGFRCK": f(np.clip(z["VEG"], 0, 100)), "TG": f(np.full((jm, im), 283.0)),
    "RSW_DN_SFC": f(rng.uniform(0, 600, (jm, im))), "RLW_DN_SFC": f(rng.uniform(200, 350, (jm, im))),
    "RAINBL": f(rng.uniform(0, 0.5, (jm, im))), "SR": f(rng.uniform(0, 1, (jm, im))),
    "SNOW": f(rng.uniform(0, 30, (jm, im)) * (rng.random((jm, im)) < 0.3)), "CMC": f(np.zeros((jm, im))),
    "TSFC": f(rng.uniform(260, 290, (jm, im))), "ALBEDO": f(np.full((jm, im), 0.2)), "EPSR": f(np.full((jm, im), 0.95)),
    "CHS2": f(np.full((jm, im), 0.01)), "CQS2": f(np.full((jm, im), 0.01)), "QSH": f(np.full((jm, im), 0.003)),
    "STC": f(rng.uniform(270, 285, (4, jm, im))), "SMC": f(rng.uniform(0.15, 0.4, (4, jm, im))),
    "T": f(rng.uniform(260, 290, (lm, jm, im))), "Q": f(rng.uniform(0.001, 0.006, (lm, jm, im))),
    "U_PHY": f(rng.uniform(-10, 10, (lm, jm, im))), "V_PHY": f(rng.uniform(-10, 10, (lm, jm, im))),
    "DZ": f(np.full((lm, jm, im), 40.0)),
    "PHINT": f(np.stack([np.full((jm, im), p) for p in (50000.0, 90000.0, 99000.0, 99500.0)])),
    "GLAT": f(np.deg2rad(np.linspace(15, 60, jm))[:, None] * np.ones((1, im))),
    "GLON": f(np.deg2rad(np.linspace(-140, -55, im))[None, :] * np.ones((jm, 1))),
    "TSHLTR": f(np.full((jm, im), 280.0)), "QSHLTR": f(np.full((jm, im), 0.003)), "PSHLTR": f(np.full((jm, im), 99000.0)),
})
st.fields["SNOWH"] = st.fields["SNOW"] * cp.float32(0.005)
st.fields["SH2O"] = st.fields["SMC"].copy()
grid = types.SimpleNamespace(im=im, jm=jm, lm=lm, dxh=np.full(jm, 12000.0, np.float32))
ctx = types.SimpleNamespace(st=st, grid=grid, tables={}, cfg=types.SimpleNamespace(nphs=2, dt=25.0, land_surface="noah"),
                            dt=25.0, ntimestep=1, start=datetime.datetime(2026, 1, 23, 12), start_time=None)
import os
os.environ["NAMGPU_LSM"] = "noahmp"
assert m.selected(ctx)
import time
for n in (1, 2, 4, 6, 8):
    ctx.ntimestep = n
    cp.cuda.Device().synchronize(); t0 = time.perf_counter()
    c = m.run(ctx)
    cp.cuda.Device().synchronize(); print("call ms", round((time.perf_counter() - t0) * 1e3, 1))
    print("call ntsd", n, "census", c)
bad = []
for k in ("TSFC", "TWBS", "QWBS", "ELFLX", "GRNFLX", "QSH", "STC", "SMC", "SH2O", "SNOW", "SNOWH", "ALBEDO", "TSHLTR", "QSHLTR"):
    a = st.fields[k]
    L = cp.asarray(land) if a.ndim == 2 else cp.asarray(land)[None]
    sel = a[cp.broadcast_to(L, a.shape)]
    nan = int(cp.isnan(a).sum())
    print(f"{k:7s} nan={nan} land min={float(sel.min()):.4g} mean={float(sel.mean()):.4g} max={float(sel.max()):.4g}")
    if nan:
        bad.append(k)
print("NaN in", bad if bad else "nothing")
import hashlib
h = hashlib.sha256()
for k in sorted(st.fields):
    h.update(k.encode()); h.update(cp.asnumpy(st.fields[k]).tobytes())
print("state digest", h.hexdigest())
