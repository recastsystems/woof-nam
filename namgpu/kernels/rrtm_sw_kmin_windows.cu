// rrtm_sw_kmin_windows.cu -- MINVAL(laytrop) groups of the SW taumol units when rrtm-call batches several
// radiation row windows into one launch set (namgpu/physics/rrtm/rrtm_call.py, sw_seam.window_kmins).
// SPDX-License-Identifier: Apache-2.0
//
// Why: taumol19 and taumol23 take MINVAL(laytrop) over groups of CHK = 8 consecutive day columns, and taumol25 and
// taumol28 over every day column of the launch (the sw_seam "declared divergence"). Those groups are formed from the
// day list of ONE rrtm-call row window (rrtm_chunk.windows(JM, ROWS)): the day columns of window W, in row-major
// order over its rows [w0, w1), are its day indices 0 .. nday_W - 1. When B consecutive windows run as one batch,
// the batch day list (grrad_sw: SW_DAY, SW_CSUM, SW_IDXDAY over the batch rows) holds the day columns of each window
// as one contiguous run [ws_W, we_W) in the same order, with ws_W = day columns in batch rows before w0 and we_W =
// day columns in batch rows before w1 (windows overlap by their halo rows, so their runs overlap too). A column keeps
// the groups of the window that OWNS its row (rrtm_chunk.windows: owned rows [a, b)), so every owned column gets the
// minimum that a launch of its window alone gives it: groups start at ws_W + 8 g and end at min(+8, we_W); the
// window minimum is over [ws_W, we_W). Columns of rows that no window of the batch owns (the batch halo rows) are
// computed and never written back; they get their own laytrop.
//
// Layout: one thread per day index t of the batch (0 .. ncols-1, ncols >= SW_NDAY; one SW chunk per batch).
//   laytrop   [ncols]  int32  setcoef laytrop per day column (live entries only are read)
//   idxday    [JMB*IM] int32  SW_IDXDAY: batch-local flat index j*IM + i of day column t
//   ndayp     [1]      int32  SW_NDAY
//   csum      [JMB*IM] int32  SW_CSUM: inclusive cumsum of SW_DAY over the batch rows
//   win_r0/r1 [nwin]   int32  batch-local first row / one past the last row of each window
//   row_owner [JMB]    int32  window (0 .. nwin-1) owning each batch row (batch halo rows: the nearest window)
// No host sync, no shared memory. The caller fills kwin with INT_MAX before rrtm_sw_kmin_win.

__device__ __forceinline__ int kmw_cum(const int* __restrict__ csum, int r, int im)
{
    return r > 0 ? csum[(size_t)r * (size_t)im - 1] : 0;    // day columns in batch rows [0, r)
}

extern "C" __global__ void rrtm_sw_kmin_win(
    const int* __restrict__ laytrop, const int* __restrict__ ndayp, const int* __restrict__ csum,
    const int* __restrict__ win_r0, const int* __restrict__ win_r1,
    const int nwin, const int im, const int ncols, int* __restrict__ kwin)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncols || t >= ndayp[0]) return;
    const int v = laytrop[t];
    for (int w = 0; w < nwin; ++w) {
        const int ws = kmw_cum(csum, win_r0[w], im);
        const int we = kmw_cum(csum, win_r1[w], im);
        if (t >= ws && t < we) atomicMin(&kwin[w], v);
    }
}

extern "C" __global__ void rrtm_sw_kmin_cols(
    const int* __restrict__ laytrop, const int* __restrict__ idxday, const int* __restrict__ ndayp,
    const int* __restrict__ csum, const int* __restrict__ win_r0, const int* __restrict__ win_r1,
    const int* __restrict__ row_owner, const int* __restrict__ kwin,
    const int im, const int ncols, const int nlay, const int chk,
    int* __restrict__ kmin8, int* __restrict__ kminw)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= ncols) return;
    if (t >= ndayp[0]) {                 // dead column: never written by swrad_post
        kmin8[t] = nlay;
        kminw[t] = nlay;
        return;
    }
    const int lt = laytrop[t];
    const int w = row_owner[idxday[t] / im];
    const int ws = kmw_cum(csum, win_r0[w], im);
    const int we = kmw_cum(csum, win_r1[w], im);
    if (t < ws || t >= we) {             // a batch halo row outside its nearest window: never written back
        kmin8[t] = lt;
        kminw[t] = lt;
        return;
    }
    const int g0 = ws + ((t - ws) / chk) * chk;
    const int g1 = min(g0 + chk, we);
    int m = lt;
    for (int s = g0; s < g1; ++s) m = min(m, laytrop[s]);
    kmin8[t] = m;
    kminw[t] = kwin[w];
}
