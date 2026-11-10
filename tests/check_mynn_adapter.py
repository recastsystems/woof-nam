"""Small-shape GPU check of namgpu.physics.mynn.phys_mynn_edmf: synthetic NMMB-ordered columns (L=1 top), a daytime
convective land half and a stable-ish water half, 3 calls of sfc -> post -> pbl. Prints ranges; fails on non-finite."""
import types
import numpy as np
import cupy as cp

from namgpu.physics.mynn import phys_mynn_edmf as M


class St:
    def __init__(self):
        self.fields = {}

    def add(self, name, shape, dtype="float32"):
        if name not in self.fields:
            self.fields[name] = cp.zeros(shape, dtype=dtype)
        return self.fields[name]

    def __getitem__(self, k):
        return self.fields[k]


def main():
    lm, jm, im = 60, 40, 50
    st = St()
    # interface pressures top (L=1) to bottom (LM+1): 2 hPa .. 1000 hPa
    pint = np.linspace(200.0, 100000.0, lm + 1).astype(np.float32)
    pmid = 0.5 * (pint[1:] + pint[:-1])
    ts = np.full((jm, im), 300.0, np.float32); ts[:, im // 2:] = 288.0          # hot land west, cool water east
    T = np.empty((lm, jm, im), np.float32)
    for k in range(lm):
        T[k] = np.maximum(295.0 * (pmid[k] / 1e5) ** 0.19, 210.0)
    Q = (0.012 * (pmid / 1e5) ** 3).astype(np.float32)[:, None, None] * np.ones((1, jm, im), np.float32)
    ex = (pmid / 1e5) ** M.CAPPA
    f = st.fields
    f["T"] = cp.asarray(T); f["Q"] = cp.asarray(Q); f["QC"] = cp.zeros_like(f["T"]); f["QS"] = cp.zeros_like(f["T"])
    f["TH"] = f["T"] / cp.asarray(ex.astype(np.float32))[:, None, None]
    f["EXNER"] = cp.asarray(np.broadcast_to(ex.astype(np.float32)[:, None, None], (lm, jm, im)).copy())
    f["PHMID"] = cp.asarray(np.broadcast_to(pmid.astype(np.float32)[:, None, None], (lm, jm, im)).copy())
    f["PHINT"] = cp.asarray(np.broadcast_to(pint[:, None, None], (lm + 1, jm, im)).copy())
    f["RR"] = f["PHMID"] / (M.R_D * f["T"])
    dp = cp.asarray(np.diff(pint).astype(np.float32))[:, None, None]
    f["DZ"] = (M.R_D * f["T"] * dp / (f["PHMID"] * M.G)).astype(cp.float32)
    f["U_PHY"] = cp.full((lm, jm, im), 8.0, cp.float32); f["V_PHY"] = cp.full((lm, jm, im), 2.0, cp.float32)
    f["W"] = cp.zeros((lm, jm, im), cp.float32); f["Q2"] = cp.full((lm, jm, im), 0.02, cp.float32)
    f["TSFC"] = cp.asarray(ts); xl = np.ones((jm, im), np.float32); xl[:, im // 2:] = 2.0
    f["XLAND"] = cp.asarray(xl); f["MAVAIL"] = cp.ones((jm, im), cp.float32); f["SNOWH"] = cp.zeros((jm, im), cp.float32)
    f["IVGTYP"] = cp.full((jm, im), 10, cp.int32); f["VEGFRC"] = cp.full((jm, im), 0.3, cp.float32); f["SNOWC"] = cp.zeros((jm, im), cp.float32)
    f["Z0"] = cp.asarray(np.where(xl == 1, 0.1, 1e-4).astype(np.float32)); f["USTAR"] = cp.full((jm, im), 0.1, cp.float32)
    f["QSH"] = cp.full((jm, im), 0.012, cp.float32); f["PBLH"] = cp.full((jm, im), 500.0, cp.float32)
    for n in ("CHS", "CHS2", "CQS2", "CPM", "QGH", "TWBS", "QWBS", "T2", "Q2X", "RMOL"):
        f[n] = cp.zeros((jm, im), cp.float32)
    for n in ("DUDT", "DVDT", "RTHBLTEN", "RQBLTEN", "RQCBLTEN", "RQIBLTEN", "EXCH_H"):
        f[n] = cp.zeros((lm, jm, im), cp.float32)
    ctx = types.SimpleNamespace(st=st, tables={}, grid=types.SimpleNamespace(dxh=np.full(jm, 12000.0, np.float32)))
    for step in range(3):
        M.run_sfc(ctx)
        # stand-in for NOAHLSM + SFCDIAGS: land keeps a 300 W m-2 / 1e-4 kg m-2 s-1 flux, water keeps MYNN's
        land = f["XLAND"] < 1.5
        f["TWBS"][...] = cp.where(land, 300.0, f["TWBS"]); f["QWBS"][...] = cp.where(land, 1e-4, f["QWBS"])
        rho = f["PHINT"][lm] / (M.R_D * f["TSFC"])
        f["T2"][...] = f["TSFC"] - f["TWBS"] / (rho * 1004.6 * cp.maximum(f["CHS2"], 1e-3))
        f["Q2X"][...] = f["QSH"]
        M.post_sfcdiags(ctx)
        M.run_pbl(ctx, 50.0)
        cp.cuda.Device().synchronize()
        print(f"call {step + 1}:")
        for n in ("USTAR", "CHS", "TWBS", "QWBS", "PBLH", "TSHLTR", "U10"):
            a = f[n][1:-1, 1:-1]
            print(f"  {n:9s} land {float(a[:, :im//2-1].mean()):10.4g} water {float(a[:, im//2:].mean()):10.4g}")
        for n in ("RTHBLTEN", "RQBLTEN", "DUDT", "EXCH_H", "Q2"):
            a = f[n][:, 1:-1, 1:-1]
            print(f"  {n:9s} min {float(a.min()):11.4g} max {float(a.max()):11.4g} bottom-land {float(a[-1, :, :im//2-1].mean()):11.4g}")
        bad = [n for n, a in f.items() if a.dtype.kind == "f" and not bool(cp.isfinite(a).all())]
        if bad:
            raise SystemExit(f"non-finite: {bad}")
    print("PASS")


if __name__ == "__main__":
    main()
