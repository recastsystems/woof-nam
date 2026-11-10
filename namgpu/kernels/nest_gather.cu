// WOOF-NAM concurrent nests on several cards: the parent values a nest's lateral boundary reads, gathered from the
// ranks that own them into one window array on every rank (namgpu/nest/coupled.py WindowCoupler).
//
// No NMMB counterpart: NMMB moves the same values with MPI between the parent's and the child's task sets
// (module_PARENT_CHILD_CPL_COMP.F90 PARENTS_SEND_CHILD_DATA / CHILDREN_RECV_PARENT_DATA, not ported). Here every rank
// packs the parent points it owns (nest_gather_pack), one allgather moves the packs, and every rank scatters all of
// them into the parent window (nest_gather_scatter), so kernels/nest_bc.cu reads the window exactly as it reads the
// whole parent on one card. Pure copies: the values are bit for bit the parent's.
//
// Layout (CuPy C order): parent 2-D [JM][IM], 3-D [LM][JM][IM] local memory rows; pack [NV][MAXN] with NV = 2 + 5*LM
// rows in the order PD, FIS, T(LM), Q(LM), CW(LM), U(LM), V(LM); window 2-D [JW][IW], 3-D [LM][JW][IW].

extern "C" __global__ void nest_gather_pack(
    const float* __restrict__ pd, const float* __restrict__ fis, const float* __restrict__ t,
    const float* __restrict__ q, const float* __restrict__ cw, const float* __restrict__ u,
    const float* __restrict__ v, long long plane, int lm, int maxn, const long long* __restrict__ idx,
    float* __restrict__ pack)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;                       // pack row
    if (k >= maxn) return;
    const long long i = idx[k];
    float x;
    if (r == 0) x = pd[i];
    else if (r == 1) x = fis[i];
    else {
        const int f = (r - 2) / lm, l = (r - 2) % lm;
        const float* src = f == 0 ? t : f == 1 ? q : f == 2 ? cw : f == 3 ? u : v;
        x = src[(long long)l * plane + i];
    }
    pack[(size_t)r * maxn + k] = x;
}

// recv [NR][NV][MAXN]; wpos [NR][MAXN] window flat index of each packed point (-1: padding)
extern "C" __global__ void nest_gather_scatter(
    const float* __restrict__ recv, const long long* __restrict__ wpos, int nr, int nv, int maxn, int lm,
    long long wplane, float* __restrict__ wpd, float* __restrict__ wfis, float* __restrict__ wt,
    float* __restrict__ wq, float* __restrict__ wcw, float* __restrict__ wu, float* __restrict__ wv)
{
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;                       // pack row
    const int s = blockIdx.z;                       // source rank
    if (k >= maxn) return;
    const long long w = wpos[(size_t)s * maxn + k];
    if (w < 0) return;
    const float x = recv[((size_t)s * nv + r) * maxn + k];
    if (r == 0) { wpd[w] = x; return; }
    if (r == 1) { wfis[w] = x; return; }
    const int f = (r - 2) / lm, l = (r - 2) % lm;
    float* dst = f == 0 ? wt : f == 1 ? wq : f == 2 ? wcw : f == 3 ? wu : wv;
    dst[(long long)l * wplane + w] = x;
}
