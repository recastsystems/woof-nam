// WOOF-NAM unit io-wind-rotate: grid-relative winds of the rotated latitude-longitude B grid to
// earth-relative (east, north) winds at H points, for the history, the maps and the products.
//
// Mode "new". The angle between grid i and geographic east comes from the rotated-pole map
// (namgpu/grid.py rotated_to_geographic; WMO-No. 306 GRIB2 grid definition template 3.1); the
// wrapper io_wind_rotate.py builds the four unit-vector components per H point once and the
// derivation is in its docstring. The component convention is WMO-No. 306 GRIB2 code table 3.3,
// bit 5. The notice is in NOTICE.
//
// Math per H point (i0, j0), level-independent vectors:
//   ie = unit vector of increasing grid i, (ie_e, ie_n) in (east, north)
//   je = unit vector of increasing grid j, (je_e, je_n)
//   u_east  = u * ie_e + v * je_e
//   v_north = u * ie_n + v * je_n
// 3-D winds are V-point fields (V(i,j) north-east of H(i,j), CONTRACT.md section 1). H(i0, j0)
// takes the grid-relative mean of V(i0-1, j0-1), V(i0, j0-1), V(i0-1, j0), V(i0, j0) before the turn.
// The last V row and column are not part of the grid, so only V points with i0 <= IM-2 and
// j0 <= JM-2 enter; an edge H point averages the two neighbours it has, a corner H point one.
// 10 m winds are H-point fields and are turned in place.
//
// Grid: kernels.grid_2d(IM, JM, LM) for io_wind_rotate_3d (blockIdx.z = level), grid_2d(IM, JM, 1)
// for io_wind_rotate_10m. Every H point is written. The neighbour tests are plain local row tests
// (j0 - 1 >= 0, j0 <= JM - 2), which give the one-card answer on every owned row of a row slab.

#include "nmmb_common.cuh"

extern "C" __global__ void io_wind_rotate_3d(
    const int LM, const int IM, const int JM,
    const float* __restrict__ ie_e, const float* __restrict__ ie_n,   // [JM,IM] H-point grid-i vector
    const float* __restrict__ je_e, const float* __restrict__ je_n,   // [JM,IM] H-point grid-j vector
    const float* __restrict__ u, const float* __restrict__ v,         // [LM,JM,IM] V points, grid-relative
    float* __restrict__ ue, float* __restrict__ ve)                   // [LM,JM,IM] H points, earth-relative
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;

    float su = 0.0f, sv = 0.0f;
    int n = 0;
    for (int dj = -1; dj <= 0; ++dj) {
        const int jv = j0 + dj;
        if (jv < 0 || jv > JM - 2) continue;
        for (int di = -1; di <= 0; ++di) {
            const int iv = i0 + di;
            if (iv < 0 || iv > IM - 2) continue;
            const size_t k = IDX3(iv, jv, l0, IM, JM);
            su += u[k];
            sv += v[k];
            ++n;
        }
    }
    const size_t c3 = IDX3(i0, j0, l0, IM, JM);
    if (n == 0) {              // a grid of one row or one column has no V point
        ue[c3] = 0.0f;
        ve[c3] = 0.0f;
        return;
    }
    const float r = 1.0f / (float)n;
    const float ug = su * r, vg = sv * r;
    const size_t c2 = IDX2(i0, j0, IM);
    ue[c3] = ug * ie_e[c2] + vg * je_e[c2];
    ve[c3] = ug * ie_n[c2] + vg * je_n[c2];
}

extern "C" __global__ void io_wind_rotate_10m(
    const int IM, const int JM,
    const float* __restrict__ ie_e, const float* __restrict__ ie_n,
    const float* __restrict__ je_e, const float* __restrict__ je_n,
    const float* __restrict__ u10, const float* __restrict__ v10,     // [JM,IM] H points, grid-relative
    float* __restrict__ ue10, float* __restrict__ ve10)               // [JM,IM] H points, earth-relative
{
    POINT_IJL
    (void)l0;
    if (i0 >= IM || j0 >= JM) return;
    const size_t c2 = IDX2(i0, j0, IM);
    const float ug = u10[c2], vg = v10[c2];
    ue10[c2] = ug * ie_e[c2] + vg * je_e[c2];
    ve10[c2] = ug * ie_n[c2] + vg * je_n[c2];
}
