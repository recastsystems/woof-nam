"""drv-timing-receipt: per-stage wall-clock stamps for the run receipt, with no added device syncs.

Mode "new": no NAM Fortran counterpart (driver glue). Kernel: namgpu/kernels/drv_timing_receipt.cu
(timing_stamp, one thread, one launch).

Two clocks per stage, both recorded at the call site:
    host    time.perf_counter_ns() when the host reaches the stage (launch side).
    device  %globaltimer (ns) written by timing_stamp when the step stream reaches the stamp.
            This is the GPU's own time, so it shows where the stream actually waits.

Wiring: Driver.step() stamps every plan entry it runs (label "<unit>/<cadence key>"), so a stamp
marks the START of that stage; device_ns_to_next is the stage's GPU time. Driver.run() calls
drain() at its hourly synchronisation point (the stream is already synchronised there, so the
copy waits for nothing) and write_receipt() when the forecast ends.

API:
    stamp(ctx, label)        record one stage boundary. Host stamp, plus one 1-thread kernel enqueued
                             on ctx.stream. No synchronisation, no copy.
    drain(ctx)               copy the device stamps taken since the last drain back to the host.
                             Called once per forecast hour, at the hourly sync, never per step.
    receipt(ctx)             drains, then returns the receipt dict (per-stage summary and totals).
    write_receipt(ctx, dir)  writes timing_receipt.json (summary) and timing_stamps.npz (every stamp).
    run(ctx)                 a Driver-protocol entry point: stamps "step <ntimestep>".

Storage: ctx.timings["_rec"] (labels, host stamps, device stamps, counters) and ctx.tables["timing_dev"]
(a list of uint64 device blocks of CAPACITY slots: the first allocated on the first stamp, one more whenever the
stamps since the last drain fill the blocks held; drain copies them all back).

Capacity: no stamp is dropped. With one fixed block of CAPACITY slots, the 300 start-up filter steps of the 12 km
parent (no drain until the first forecast hour) filled it before the first drain, and every forecast-hour stamp
after them was dropped (profile, 2026-11-09). A new block is a pool allocation and a memset, no sync.
"dropped" stays in the receipt (always 0).
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path

import numpy as np

from namgpu import kernels

UNIT = "drv_timing_receipt"
KERNEL = "timing_stamp"
CAPACITY = 16384


def _rec(ctx) -> dict:
    t = ctx.timings
    if "_rec" not in t:
        t["_rec"] = {"labels": [], "host_ns": [], "device_ns": [], "pending": 0,
                     "dropped": 0, "drains": 0}
    return t["_rec"]


def _dev_buf(ctx, slot: int = 0):
    """The device block holding `slot` (stamps since the last drain), allocated when first needed."""
    import cupy as cp
    blocks = ctx.tables.get("timing_dev")
    if not isinstance(blocks, list):
        blocks = []
        ctx.tables["timing_dev"] = blocks
    while len(blocks) <= slot // CAPACITY:
        blocks.append(cp.zeros((CAPACITY,), dtype=cp.uint64))
    return blocks[slot // CAPACITY]


def _scope(ctx):
    return ctx.stream if ctx.stream is not None else contextlib.nullcontext()


def stamp(ctx, label: str) -> None:
    rec = _rec(ctx)
    slot = rec["pending"]
    rec["labels"].append(str(label))
    rec["host_ns"].append(time.perf_counter_ns())
    rec["device_ns"].append(None)            # filled by drain()
    rec["pending"] = slot + 1
    k = kernels.get(UNIT, KERNEL)
    with _scope(ctx):
        k((1,), (1,), (_dev_buf(ctx, slot), np.int32(slot % CAPACITY)))


def drain(ctx) -> None:
    """Copy the device stamps taken since the last drain to the host. Call at the hourly sync only."""
    rec = _rec(ctx)
    n = rec["pending"]
    if n == 0:
        return
    import cupy as cp
    with _scope(ctx):
        blocks = ctx.tables["timing_dev"]
        vals = []
        for b in range((n + CAPACITY - 1) // CAPACITY):
            vals += [int(v) for v in cp.asnumpy(blocks[b][:min(CAPACITY, n - b * CAPACITY)])]
    start = len(rec["labels"]) - n            # the pending stamps are the last n labels
    rec["device_ns"][start:] = vals
    rec["pending"] = 0
    rec["drains"] += 1


def receipt(ctx) -> dict:
    """Drains, then summarises. Not on the step path: called when the run receipt is written."""
    drain(ctx)
    rec = _rec(ctx)
    labels, host, dev = rec["labels"], rec["host_ns"], rec["device_ns"]
    n = len(labels)
    per = {}
    for i, label in enumerate(labels):
        s = per.setdefault(label, {"count": 0, "host_ns_to_next": 0, "device_ns_to_next": 0,
                                   "timed": 0})
        s["count"] += 1
        if i + 1 < n:
            s["host_ns_to_next"] += host[i + 1] - host[i]
            s["timed"] += 1
            if dev[i] is not None and dev[i + 1] is not None:
                s["device_ns_to_next"] += dev[i + 1] - dev[i]
    stages = [{"label": k, **v} for k, v in per.items()]
    stages.sort(key=lambda s: -s["device_ns_to_next"])
    device_ok = n > 1 and all(v is not None for v in dev)
    return {
        "unit": UNIT,
        "capacity_per_drain": CAPACITY,
        "drain_point": "hourly, at the existing hourly stream synchronisation",
        "drains": rec["drains"],
        "stamped": n,
        "dropped": rec["dropped"],
        "complete": rec["dropped"] == 0,
        "host_span_ns": host[-1] - host[0] if n > 1 else 0,
        "device_span_ns": dev[-1] - dev[0] if device_ok else None,
        "stages_by_device_time": stages,
    }


def write_receipt(ctx, out_dir) -> dict:
    """Write timing_receipt.json (summary) and timing_stamps.npz (every stamp) into out_dir."""
    rec = receipt(ctx)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "timing_receipt.json").write_text(json.dumps(rec, indent=1))
    r = _rec(ctx)
    np.savez(out / "timing_stamps.npz",
             labels=np.array(r["labels"], dtype=str),
             host_ns=np.array(r["host_ns"], dtype=np.int64),
             device_ns=np.array([-1 if v is None else v for v in r["device_ns"]], dtype=np.int64))
    return rec


def run(ctx):
    stamp(ctx, f"step {ctx.ntimestep}")
