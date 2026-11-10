"""Build a case's real initial NMMB state and save what the ArWen column adapter reads/writes, for its test.

    python tools/adapter_dump_state.py --case <case dir> --source native --out <file>.npz [--managed]

Runs the real init chain (init-case-driver, as `woof-nam init`) on the full grid, then writes T, Q, PD, FIS, U, V,
W, TRACERS, CW, F_ICE, F_RAIN, F_RIMEF, the vertical coordinate and the tracer slots to one .npz.
tests/test_arwen_adapter.py loads it (NAMGPU_ADAPTER_STATE=<file>.npz).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

FIELDS = ("T", "Q", "PD", "FIS", "U", "V", "W", "TRACERS", "CW", "F_ICE", "F_RAIN", "F_RIMEF", "SM", "SICE")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--source", default="native")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--domain", default=None)
    ap.add_argument("--managed", action="store_true", help="CUDA managed memory (a shared or small card)")
    a = ap.parse_args(argv)
    import cupy as cp
    if a.managed:   # after devmem.setup (State calls it again and returns early), so the managed pool stays
        from namgpu import devmem
        devmem.setup()
        cp.cuda.set_allocator(cp.cuda.MemoryPool(cp.cuda.malloc_managed).malloc)
    from namgpu import config, domains, grid as gridmod, paths, state as statemod
    from namgpu.driver import Ctx
    domains.select(a.domain, a.source)
    cdir = paths.case_dir(a.case)
    rs = domains.ensure_static(cdir, domains.start_of(cdir), a.source)
    print(f"static: {rs['action']} ({rs['seconds']} s)", flush=True)
    cfg = config.load()
    cfg.raw["init_source"] = a.source
    g = gridmod.build(cfg)
    st = statemod.State(g)
    ctx = Ctx(cfg=cfg, grid=g, st=st, stream=cp.cuda.Stream(non_blocking=True))
    ctx.case_dir = cdir
    ctx.start = datetime.strptime(cdir.name[:14], "%Y-%m-%d_%Hz")
    ctx.out_dir = a.out.parent
    ctx.init_source = a.source
    ctx.forecast_hours = 6
    from namgpu.init import init_case_driver  # noqa: PLC0415
    t0 = time.perf_counter()
    with ctx.stream:
        init_case_driver.run(ctx)
    ctx.stream.synchronize()
    print(f"init wall {time.perf_counter() - t0:.1f} s", flush=True)
    out = {k: cp.asnumpy(st.fields[k]) for k in FIELDS if k in st.fields}
    indx = ctx.tables.get("indx", {})
    out["indx_names"] = np.array(list(indx.keys()))
    out["indx_values"] = np.array([int(v) for v in indx.values()], np.int64)
    for k in ("sg2", "sg1", "sgml2", "sgml1"):
        out["vc_" + k] = np.asarray(getattr(g, k), np.float32)
    out["vc_pt_pdtop"] = np.array([g.pt, g.pdtop], np.float64)
    out["meta"] = np.array([str(cfg.microphysics), a.source, cdir.name, str(cfg.dt), str(cfg.nphs),
                            str(cfg.nprecip), str(cfg.nrads), str(cfg.nradl)])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print("wrote", a.out, sum(v.nbytes for v in out.values()) / 2**30, "GiB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
