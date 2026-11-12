"""io-wind-rotate on the CPU: the grid-to-earth angle, the H-point mean, and the kernel source itself.

The angle. With the map of namgpu.grid.rotated_to_geographic,
    sin(lat) = sin(phi0) cos(phi) cos(lam) + cos(phi0) sin(phi)
    cos(lat) sin(dlon) = cos(phi) sin(lam)
    cos(lat) cos(dlon) = cos(phi0) cos(phi) cos(lam) - sin(phi0) sin(phi),
moving along grid i (d lam, length cos(phi) d lam on the unit sphere) changes the latitude by
    cos(lat) d lat = -sin(phi0) cos(phi) sin(lam) d lam,
so the north component of the grid-i unit vector is -sin(phi0) sin(lam) / cos(lat). Its east component,
cos(lat) d lon / (cos(phi) d lam) worked out from the last two lines, is
    (cos(phi0) cos(phi) - sin(phi0) sin(phi) cos(lam)) / cos(lat).
The two have unit length together, and the grid-j vector is the grid-i vector turned 90 degrees
anticlockwise (the map is a rotation of the sphere: conformal and orientation-preserving). The tests below
check that against a finite-difference tangent of the map itself, so a sign or a factor out of place fails.

Breakage each test prevents:
  - test_vectors_match_the_map: winds turned the wrong way or by the wrong angle (history U10/V10, U/V, maps,
    products all read them).
  - test_inverse_of_the_start_turn: a start's 10 m winds and its boundary winds not coming back to the source's
    winds through history (the start turns earth to grid with init-gfs-wind-rotate's vectors).
  - test_h_mean_edges: edge H points reading the unused last V row or column, or dividing by four where they
    have fewer neighbours.
  - test_kernel_source_on_cpu: the .cu computing something other than the reference (index or argument order).
Run: python -m pytest -q tests/test_io_wind_rotate.py   (CPU only; the kernel test needs a C++ compiler)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from namgpu.grid import rotated_to_geographic  # noqa: E402
from namgpu.io import io_wind_rotate as iwr  # noqa: E402

DTR = np.pi / 180.0


def _fd_vectors(tlm, tph, tlm0, tph0, h=1.0e-6):
    """Unit (east, north) tangents of grid i and grid j by central differences of the map."""
    _, lat = rotated_to_geographic(tlm, tph, tlm0, tph0)

    def d(dl, dp):
        lop, lap = rotated_to_geographic(tlm + dl, tph + dp, tlm0, tph0)
        lom, lam = rotated_to_geographic(tlm - dl, tph - dp, tlm0, tph0)
        dlon = np.arctan2(np.sin(lop - lom), np.cos(lop - lom))
        e, n = np.cos(lat) * dlon, lap - lam
        m = np.hypot(e, n)
        return e / m, n / m

    return d(h, 0.0) + d(0.0, h)


def _grids():
    # NAM 12 km parent (CONTRACT.md section 5: tph0d 54, tlm0d -106, 954 x 835), a high-latitude
    # nest, a southern-hemisphere one and one on the equator
    return [(54.0, -106.0, -60.039, -45.036), (70.0, 30.0, -10.0, -8.0), (-35.0, 150.0, -12.0, -10.0),
            (0.0, 0.0, -20.0, -20.0)]


def test_vectors_match_the_map():
    rng = np.random.default_rng(7)
    worst = 0.0
    for tph0d, tlm0d, wbd, sbd in _grids():
        tlm = rng.uniform(wbd, -wbd, 4000) * DTR
        tph = rng.uniform(sbd, -sbd, 4000) * DTR
        got = iwr.grid_vectors(tlm, tph, tph0d * DTR)
        want = _fd_vectors(tlm, tph, tlm0d * DTR, tph0d * DTR)
        worst = max(worst, max(float(np.max(np.abs(a - b))) for a, b in zip(got, want)))
        ie_e, ie_n, je_e, je_n = got
        assert np.allclose(ie_e * ie_e + ie_n * ie_n, 1.0, atol=1e-12)
        assert np.allclose(ie_e * je_e + ie_n * je_n, 0.0, atol=1e-12)
        assert np.allclose(ie_e * je_n - ie_n * je_e, 1.0, atol=1e-12)     # j is i turned anticlockwise
    assert worst < 1.0e-6, worst
    # at the centre the grid axes are east and north
    c = iwr.grid_vectors(0.0, 0.0, 54.0 * DTR)
    assert np.allclose([float(x) for x in c], [1.0, 0.0, 0.0, 1.0], atol=1e-15)
    # east of the centre meridian in the northern hemisphere grid i points south of east
    ie_n = float(iwr.grid_vectors(20.0 * DTR, 0.0, 54.0 * DTR)[1])
    assert ie_n < 0.0


def _bgrid(tph0d=54.0, tlm0d=-106.0, im=40, jm=30, dlmd=0.126, dphd=0.108, wbd=-2.457, sbd=-1.566, **kw):
    g = SimpleNamespace(im=im, jm=jm, lm=3, dlm=dlmd * DTR, dph=dphd * DTR, wb=wbd * DTR, sb=sbd * DTR,
                        tph0=tph0d * DTR, tlm0=tlm0d * DTR, win_i0=0, win_j0=0, joff=0)
    g.__dict__.update(kw)
    return g


def test_inverse_of_the_start_turn():
    from namgpu.init import init_gfs_wind_rotate as wr
    for tph0d, tlm0d, _, _ in _grids():
        g = _bgrid(tph0d, tlm0d, wbd=-30.0 * 0.126, sbd=-20.0 * 0.108, im=60, jm=40)
        ce_e, ce_n, cn_e, cn_n = (a.astype(np.float64) for a in wr._unit_tangents(g))   # earth -> grid at V
        he_e, he_n, hj_e, hj_n = (a.astype(np.float64) for a in iwr.host_vectors(g, "v"))
        assert max(float(np.max(np.abs(a - b))) for a, b in
                   zip((ce_e, ce_n, cn_e, cn_n), (he_e, he_n, hj_e, hj_n))) < 2.0e-6
        rng = np.random.default_rng(3)
        ue, vn = rng.normal(0, 15, (2, g.jm, g.im))
        ug, vg = ue * ce_e + vn * ce_n, ue * cn_e + vn * cn_n
        ue2, vn2 = ug * he_e + vg * hj_e, ug * he_n + vg * hj_n
        assert float(np.max(np.abs(ue2 - ue))) < 2.0e-4 and float(np.max(np.abs(vn2 - vn))) < 2.0e-4
    # row slab and nest window: the local rows are the global rows shifted by joff (and the window)
    g = _bgrid()
    full = iwr.host_vectors(g, "h")
    part = iwr.host_vectors(_bgrid(jm=10, joff=12), "h")
    assert all(np.array_equal(a[12:22], b) for a, b in zip(full, part))
    moved = iwr.host_vectors(_bgrid(win_i0=3, win_j0=2, jm=10), "h")
    big = iwr.host_vectors(_bgrid(im=43, jm=12), "h")
    assert all(np.array_equal(a, b[2:, 3:]) for a, b in zip(moved, big))


def test_h_mean_edges():
    jm, im = 4, 5
    u = np.zeros((2, jm, im), np.float32)
    u[:, :, :] = np.arange(jm * im, dtype=np.float32).reshape(jm, im)
    u[:, jm - 1, :] = 1.0e6          # unused last V row and column: must never be read
    u[:, :, im - 1] = 1.0e6
    one = np.ones((jm, im))
    zero = np.zeros((jm, im))
    ue, ve = iwr.to_earth_h(u, u * 0, one, zero, zero, one)
    vgrid = np.arange(jm * im, dtype=np.float64).reshape(jm, im)
    assert ue[0, 0, 0] == vgrid[0, 0]                                           # corner: one neighbour
    assert ue[0, 0, 2] == (vgrid[0, 1] + vgrid[0, 2]) / 2                       # south edge: two
    assert ue[0, 2, 2] == (vgrid[1, 1] + vgrid[1, 2] + vgrid[2, 1] + vgrid[2, 2]) / 4
    assert ue[0, jm - 1, im - 1] == vgrid[jm - 2, im - 2]                        # north-east corner: one
    assert ue[0, 2, im - 1] == (vgrid[1, im - 2] + vgrid[2, im - 2]) / 2         # east edge: two
    assert np.all(np.abs(ue) < 1.0e5) and np.all(ve == 0.0)


HARNESS = r"""
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
using std::min; using std::max;
struct D3 { int x, y, z; };
static D3 blockIdx, threadIdx, blockDim;
#define __global__
#define __device__
#define __forceinline__ inline
#include "io_wind_rotate.cu"
static float* rd(const char* p, size_t n) { float* a = (float*)malloc(n * 4); FILE* f = fopen(p, "rb");
  if (!f || fread(a, 4, n, f) != n) { fprintf(stderr, "read %s\n", p); exit(2); } fclose(f); return a; }
static void wr(const char* p, const float* a, size_t n) { FILE* f = fopen(p, "wb"); fwrite(a, 4, n, f); fclose(f); }
int main(int argc, char** argv) {
  const int IM = atoi(argv[1]), JM = atoi(argv[2]), LM = atoi(argv[3]);
  const size_t n2 = (size_t)IM * JM, n3 = n2 * LM;
  float *ie_e = rd("ie_e", n2), *ie_n = rd("ie_n", n2), *je_e = rd("je_e", n2), *je_n = rd("je_n", n2);
  float *u = rd("u", n3), *v = rd("v", n3), *u10 = rd("u10", n2), *v10 = rd("v10", n2);
  float *ue = (float*)calloc(n3, 4), *ve = (float*)calloc(n3, 4), *ue10 = (float*)calloc(n2, 4), *ve10 = (float*)calloc(n2, 4);
  blockDim = {32, 8, 1};
  const int gx = (IM + 31) / 32, gy = (JM + 7) / 8;
  for (int z = 0; z < LM; ++z) for (int by = 0; by < gy; ++by) for (int bx = 0; bx < gx; ++bx)
    for (int ty = 0; ty < 8; ++ty) for (int tx = 0; tx < 32; ++tx) {
      blockIdx = {bx, by, z}; threadIdx = {tx, ty, 0};
      io_wind_rotate_3d(LM, IM, JM, ie_e, ie_n, je_e, je_n, u, v, ue, ve);
      if (z == 0) io_wind_rotate_10m(IM, JM, ie_e, ie_n, je_e, je_n, u10, v10, ue10, ve10);
    }
  wr("ue", ue, n3); wr("ve", ve, n3); wr("ue10", ue10, n2); wr("ve10", ve10, n2);
  return 0;
}
"""


def test_kernel_source_on_cpu():
    cxx = shutil.which("g++") or shutil.which("clang++")
    if cxx is None:
        import pytest
        pytest.skip("no C++ compiler")
    g = _bgrid(im=37, jm=19)
    lm = 3
    vec = iwr.host_vectors(g, "h")
    rng = np.random.default_rng(11)
    u, v = rng.normal(0, 20, (2, lm, g.jm, g.im)).astype(np.float32)
    u10, v10 = rng.normal(0, 8, (2, g.jm, g.im)).astype(np.float32)
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "h.cpp").write_text(HARNESS)
        kdir = ROOT / "namgpu" / "kernels"
        subprocess.run([cxx, "-std=c++17", "-O1", f"-I{kdir}", "-o", str(d / "h"), str(d / "h.cpp")],
                       check=True, capture_output=True)
        for name, a in zip(("ie_e", "ie_n", "je_e", "je_n", "u", "v", "u10", "v10"), vec + (u, v, u10, v10)):
            np.ascontiguousarray(a, np.float32).tofile(d / name)
        subprocess.run([str(d / "h"), str(g.im), str(g.jm), str(lm)], cwd=d, check=True, capture_output=True)
        out = {k: np.fromfile(d / k, np.float32) for k in ("ue", "ve", "ue10", "ve10")}
    want_u, want_v = iwr.to_earth_h(u, v, *(a.astype(np.float64) for a in vec))
    ie_e, ie_n, je_e, je_n = (a.astype(np.float64) for a in vec)
    assert np.max(np.abs(out["ue"].reshape(u.shape) - want_u)) < 1.0e-4
    assert np.max(np.abs(out["ve"].reshape(u.shape) - want_v)) < 1.0e-4
    assert np.max(np.abs(out["ue10"].reshape(u10.shape) - (u10 * ie_e + v10 * je_e))) < 1.0e-5
    assert np.max(np.abs(out["ve10"].reshape(u10.shape) - (u10 * ie_n + v10 * je_n))) < 1.0e-5


if __name__ == "__main__":
    for t in (test_vectors_match_the_map, test_inverse_of_the_start_turn, test_h_mean_edges,
              test_kernel_source_on_cpu):
        t()
        print("ok", t.__name__)
