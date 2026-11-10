"""Multi-card runtime: one process per card (rank), an NCCL communicator over cupy.cuda.nccl.

MULTICARD-PLAN.md section 6. Started by tools/run_forecast.py --cards N: every rank pins CUDA_VISIBLE_DEVICES to its
card, rank 0 writes the NCCL unique id to a file, the others read it. With one card ctx.comm is None and nothing here
runs, so the one-card plan, shapes and arithmetic are unchanged.

Every collective is enqueued on the caller's stream (ctx.stream): no host sync. barrier() is the one exception
(host sync, used at init and at the end only).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path


def _dt(nccl, dtype):
    import numpy as np
    d = np.dtype(dtype)
    return {np.dtype(np.float32): nccl.NCCL_FLOAT32, np.dtype(np.float64): nccl.NCCL_FLOAT64,
            np.dtype(np.int32): nccl.NCCL_INT32, np.dtype(np.uint32): nccl.NCCL_UINT32,
            np.dtype(np.int64): nccl.NCCL_INT64, np.dtype(np.uint64): nccl.NCCL_UINT64,
            np.dtype(np.uint8): nccl.NCCL_UINT8}[d]


class Comm:
    def __init__(self, rank: int, nranks: int, idfile, timeout_s: float = 600.0):
        from cupy.cuda import nccl
        self.rank, self.nranks = int(rank), int(nranks)
        self._nccl = nccl
        idfile = Path(idfile)
        if self.rank == 0:
            uid = nccl.get_unique_id()
            tmp = idfile.with_name(idfile.name + ".tmp")
            # bytes in current CuPy (a tuple of ints in older ones): keep the type through the file
            tmp.write_text(uid.hex() if isinstance(uid, (bytes, bytearray)) else json.dumps(list(uid)))
            os.replace(tmp, idfile)
        else:
            t0 = time.time()
            while not idfile.exists():
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"parallel: rank {rank} waited {timeout_s} s for the NCCL id {idfile}")
                time.sleep(0.05)
            txt = idfile.read_text().strip()
            uid = tuple(json.loads(txt)) if txt.startswith("[") else bytes.fromhex(txt)
        self.nccl = nccl.NcclCommunicator(self.nranks, uid, self.rank)

    # ---- point to point -----------------------------------------------------------------------------------
    def sendrecv(self, pairs, stream):
        """pairs: [(sendbuf or None, recvbuf or None, peer)], all in one NCCL group on `stream`."""
        nccl, s = self._nccl, stream.ptr
        nccl.groupStart()
        try:
            for snd, rcv, peer in pairs:
                if snd is not None:
                    self.nccl.send(snd.data.ptr, snd.size, _dt(nccl, snd.dtype), int(peer), s)
                if rcv is not None:
                    self.nccl.recv(rcv.data.ptr, rcv.size, _dt(nccl, rcv.dtype), int(peer), s)
        finally:
            nccl.groupEnd()

    # ---- collectives --------------------------------------------------------------------------------------
    def allgather(self, sendbuf, recvbuf, stream):
        self.nccl.allGather(sendbuf.data.ptr, recvbuf.data.ptr, sendbuf.size, _dt(self._nccl, sendbuf.dtype),
                            stream.ptr)

    def bcast(self, buf, root: int, stream):
        self.nccl.bcast(buf.data.ptr, buf.size, _dt(self._nccl, buf.dtype), int(root), stream.ptr)

    def allreduce(self, sendbuf, recvbuf, op: str, stream):
        nccl = self._nccl
        o = {"sum": nccl.NCCL_SUM, "max": nccl.NCCL_MAX, "min": nccl.NCCL_MIN}[op]
        self.nccl.allReduce(sendbuf.data.ptr, recvbuf.data.ptr, sendbuf.size, _dt(nccl, sendbuf.dtype), o,
                            stream.ptr)

    def barrier(self, stream) -> int:
        """Host barrier (init and end only): a 1-int allreduce, then a stream sync. Returns nranks."""
        import cupy as cp
        with stream:
            x = cp.ones((1,), dtype=cp.int32)
            self.allreduce(x, x, "sum", stream)
        stream.synchronize()
        return int(x.get()[0])


class HostComm:
    """Several ranks on ONE card, for the N-rank referee on a one-card machine: NCCL refuses two ranks on one device
    ("Duplicate GPU detected"), so every exchange is staged through files in a shared folder (normally /dev/shm).
    Selected only by NAMGPU_COMM=host; never a product path (each call synchronises the stream and copies through the
    host, so it is slow). The answers are the same bytes NCCL moves: send/recv buffers are copied whole, reductions
    are done in the buffer's own dtype in rank order (sum of float values would be order-dependent; the forecast
    reduces only integer flags and max/min, which are order-free).

    Message files: p<src>-<dst>-<n> for the n-th point-to-point message from src to dst (deleted by the receiver),
    c<k>-<rank> for this rank's part of the k-th collective (every rank calls the collectives in the same order;
    removed by its writer two collectives later, when every rank has read it)."""

    def __init__(self, rank: int, nranks: int, idfile, timeout_s: float = 600.0):
        import collections
        self.rank, self.nranks = int(rank), int(nranks)
        self.dir = Path(str(idfile) + ".hostcomm")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = float(timeout_s)
        self._sent = collections.Counter()
        self._got = collections.Counter()
        self._col = 0

    def _put(self, name: str, host) -> None:
        tmp = self.dir / f".{name}.tmp{self.rank}"
        host.tofile(tmp)
        os.replace(tmp, self.dir / name)

    def _get(self, name: str, dtype, unlink: bool):
        import numpy as np
        p = self.dir / name
        t0 = time.time()
        while not p.exists():
            if time.time() - t0 > self.timeout_s:
                raise TimeoutError(f"parallel.HostComm: rank {self.rank} waited {self.timeout_s} s for {p}")
            time.sleep(0.0002)
        a = np.fromfile(p, dtype=dtype)
        if unlink:
            p.unlink()
        return a

    def sendrecv(self, pairs, stream):
        import cupy as cp
        stream.synchronize()
        for snd, _rcv, peer in pairs:
            if snd is not None:
                k = (self.rank, int(peer))
                self._put(f"p{k[0]}-{k[1]}-{self._sent[k]}", cp.asnumpy(snd).ravel())
                self._sent[k] += 1
        for _snd, rcv, peer in pairs:
            if rcv is not None:
                k = (int(peer), self.rank)
                h = self._get(f"p{k[0]}-{k[1]}-{self._got[k]}", rcv.dtype, True)
                self._got[k] += 1
                if h.size != rcv.size:
                    raise ValueError(f"parallel.HostComm: rank {self.rank} got {h.size} values from {peer}, "
                                     f"expected {rcv.size}")
                rcv.set(h.reshape(rcv.shape), stream=stream)
        stream.synchronize()

    def _gather_all(self, sendbuf, stream):
        import cupy as cp
        stream.synchronize()
        k = self._col
        self._col += 1
        self._put(f"c{k}-{self.rank}", cp.asnumpy(sendbuf).ravel())
        parts = [self._get(f"c{k}-{r}", sendbuf.dtype, False) for r in range(self.nranks)]
        old = self.dir / f"c{k - 2}-{self.rank}"
        if k >= 2 and old.exists():
            old.unlink()
        return parts

    def allgather(self, sendbuf, recvbuf, stream):
        import numpy as np
        parts = self._gather_all(sendbuf, stream)
        recvbuf.set(np.concatenate(parts).reshape(recvbuf.shape), stream=stream)
        stream.synchronize()

    def bcast(self, buf, root: int, stream):
        parts = self._gather_all(buf, stream)
        buf.set(parts[int(root)].reshape(buf.shape), stream=stream)
        stream.synchronize()

    def allreduce(self, sendbuf, recvbuf, op: str, stream):
        import numpy as np
        parts = self._gather_all(sendbuf, stream)
        f = {"sum": np.add, "max": np.maximum, "min": np.minimum}[op]
        acc = parts[0].copy()
        for p in parts[1:]:
            acc = f(acc, p).astype(acc.dtype)
        recvbuf.set(acc.reshape(recvbuf.shape), stream=stream)
        stream.synchronize()

    def barrier(self, stream) -> int:
        import cupy as cp
        with stream:
            x = cp.ones((1,), dtype=cp.int32)
            self.allreduce(x, x, "sum", stream)
        stream.synchronize()
        return int(x.get()[0])


def make_comm(rank: int, nranks: int, idfile):
    """The communicator for rank `rank` of `nranks`: NCCL (Comm), or HostComm when NAMGPU_COMM=host (several ranks
    sharing one card, the N-rank referee on a one-card machine)."""
    kind = os.environ.get("NAMGPU_COMM", "nccl").strip().lower()
    if kind == "host":
        return HostComm(rank, nranks, idfile)
    if kind != "nccl":
        raise SystemExit(f"NAMGPU_COMM={kind!r}: nccl (default) or host")
    return Comm(rank, nranks, idfile)


def get(ctx):
    """The communicator when more than one card runs, else None (one card: every multi-card step is a no-op)."""
    c = getattr(ctx, "comm", None)
    return c if (c is not None and c.nranks > 1) else None
