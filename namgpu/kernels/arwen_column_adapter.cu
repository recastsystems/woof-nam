// arwen_column_adapter.cu -- the ONE NMMB <-> ArWen physics column adapter (NAM modernisation, 2026-11-08).
//
// NMMB side (CONTRACT.md): A[l][j][i], l = 0 is the MODEL TOP, B grid, U/V at V points (V(i,j) is NE of H(i,j)),
//   Q = specific humidity (a TRACERS slot), water species = TRACERS slots (mixing ratio, as the NMMB physics
//   drivers treat them), hydrostatic layer pressure PLYR = SGML2*PD + PSGML1, interfaces PHINT = SG2*PD + PSG1
//   (module_TURBULENCE.F90:626-648, the same column prep NMMB's own TURBL does).
// ArWen side (gpuwm/core): a[k][j][i], (nz, ny, nx), k = 0 is the LOWEST layer, all fields at mass points,
//   qv = water-vapour MIXING RATIO, theta, Exner pii = (p/1e5)^(R/cp), z/z8w heights above sea level (m),
//   dz8w layer depth, rho = moist density (WRF phy_prep: rho_d*(1+qv)), alt = 1/rho_d.
//
// gather: one thread per column, levels bottom-up (the height recurrence). Writes only the outputs whose bit is
//   set in `need` (the rest point at a 1-element dummy and are never touched).
// scatter: one thread per column; writes back ONLY values the scheme changed (new != snapshot), so a round trip
//   with no physics is bitwise identity on every NMMB field. Writes are limited to ITS_B1..ITE_B1 / JTS_B1..JTE_B1
//   (the interior minus one row), exactly where NMMB TURBL writes T/Q/QC (module_TURBULENCE.F90:1325-1352).
//   Mass-point wind changes go to DUH/DVH; h2v_add then adds their four-point average to U/V (module_H_TO_V.F90).
#include "nmmb_common.cuh"

// need / write bits (keep in step with namgpu/physics/arwen_adapter.py BITS)
#define AD_P     (1u << 0)
#define AD_P8W   (1u << 1)
#define AD_T     (1u << 2)
#define AD_TH    (1u << 3)
#define AD_PII   (1u << 4)
#define AD_QV    (1u << 5)
#define AD_QC    (1u << 6)
#define AD_QR    (1u << 7)
#define AD_QI    (1u << 8)
#define AD_QS    (1u << 9)
#define AD_QG    (1u << 10)
#define AD_NI    (1u << 11)
#define AD_NR    (1u << 12)
#define AD_U     (1u << 13)
#define AD_V     (1u << 14)
#define AD_W     (1u << 15)
#define AD_W8W   (1u << 16)
#define AD_Z     (1u << 17)
#define AD_Z8W   (1u << 18)
#define AD_DZ8W  (1u << 19)
#define AD_RHO   (1u << 20)
#define AD_ALT   (1u << 21)
#define AD_QKE   (1u << 22)   // NMMB Q2 (= 2 TKE, the MYJ/MYNN qke), TRACERS slot INDX_Q2
#define AD_NSPEC 8           // qc qr qi qs qg ni nr qke
__device__ __forceinline__ unsigned ad_spbit(int s) { return s < 7 ? (AD_QC << s) : AD_QKE; }

extern "C" __global__ void adapter_gather(
    const float* __restrict__ T, const float* __restrict__ Q, const float* __restrict__ PD,
    const float* __restrict__ FIS, const float* __restrict__ U, const float* __restrict__ V,
    const float* __restrict__ W, const float* __restrict__ TRAC, const int* __restrict__ SLOT,
    const float* __restrict__ SG2, const float* __restrict__ PSG1,
    const float* __restrict__ SGML2, const float* __restrict__ PSGML1,
    float* p, float* p8w, float* t, float* th, float* pii, float* qv,
    const unsigned long long* __restrict__ HYD,
    float* u, float* v, float* w, float* w8w, float* z, float* z8w, float* dz8w, float* rho, float* alt,
    const unsigned need, const int IM, const int JM, const int LM)
{
    COLUMN_IJ(IM, JM);
    float* hyd[AD_NSPEC];
    int slot[AD_NSPEC];
    for (int s = 0; s < AD_NSPEC; ++s) { slot[s] = SLOT[s]; hyd[s] = (float*)HYD[s]; }
    const size_t ij = IDX2(i0, j0, IM);
    const size_t plane = (size_t)IM * JM;
    const float pd = PD[ij];
    float zb = FIS[ij] / NMMB_G;
    // V-point neighbours of H(i,j): V(i-1,j-1), V(i,j-1), V(i-1,j), V(i,j) (clamped on the outer row/column,
    // where NMMB itself never writes physics tendencies)
    const int im1 = i0 > 0 ? i0 - 1 : 0, jm1 = j0 > 0 ? j0 - 1 : 0;
    const size_t v00 = IDX2(im1, jm1, IM), v10 = IDX2(i0, jm1, IM), v01 = IDX2(im1, j0, IM), v11 = ij;
    float wprev = 0.0f;
    if (need & AD_Z8W) z8w[ij] = zb;
    if (need & AD_P8W) p8w[ij] = SG2[LM] * pd + PSG1[LM];
    if (need & AD_W8W) w8w[ij] = 0.0f;
    for (int k = 0; k < LM; ++k) {
        const int l0 = LM - 1 - k;
        const size_t n = (size_t)l0 * plane + ij;      // NMMB index
        const size_t a = (size_t)k * plane + ij;       // ArWen index
        const float plyr = SGML2[l0] * pd + PSGML1[l0];
        const float ptop = SG2[l0] * pd + PSG1[l0];
        const float pbot = SG2[l0 + 1] * pd + PSG1[l0 + 1];
        const float tl = T[n];
        const float qq = Q[n];
        const float ql = fmaxf(qq, NMMB_EPSQ);
        const float rxner = powf(1.0e5f / plyr, NMMB_CAPPA);
        const float dz = tl * (NMMB_P608 * ql + 1.0f) * NMMB_R_D * (pbot - ptop) / (plyr * NMMB_G);
        const float r = qq / (1.0f - qq);
        if (need & AD_P) p[a] = plyr;
        if (need & AD_P8W) p8w[a + plane] = ptop;
        if (need & AD_T) t[a] = tl;
        if (need & AD_TH) th[a] = tl * rxner;
        if (need & AD_PII) pii[a] = 1.0f / rxner;
        if (need & AD_QV) qv[a] = r;
        if (need & AD_DZ8W) dz8w[a] = dz;
        if (need & AD_Z) z[a] = zb + 0.5f * dz;
        zb = zb + dz;
        if (need & AD_Z8W) z8w[a + plane] = zb;
        if (need & (AD_RHO | AD_ALT)) {
            const float rmix = fmaxf(r, 0.0f);
            const float rhod = plyr / (NMMB_R_D * tl * (1.0f + (NMMB_R_V / NMMB_R_D) * rmix));
            if (need & AD_RHO) rho[a] = rhod * (1.0f + rmix);
            if (need & AD_ALT) alt[a] = 1.0f / rhod;
        }
        for (int s = 0; s < AD_NSPEC; ++s)
            if ((need & ad_spbit(s)) && slot[s] >= 0)
                hyd[s][a] = TRAC[(size_t)slot[s] * LM * plane + n];
        const size_t b = (size_t)l0 * plane;
        if (need & AD_U) u[a] = 0.25f * (U[b + v00] + U[b + v10] + U[b + v01] + U[b + v11]);
        if (need & AD_V) v[a] = 0.25f * (V[b + v00] + V[b + v10] + V[b + v01] + V[b + v11]);
        const float wk = W[n];
        if (need & AD_W) w[a] = wk;
        if ((need & AD_W8W) && k > 0) w8w[a] = 0.5f * (wprev + wk);
        wprev = wk;
    }
    if (need & AD_W8W) w8w[(size_t)LM * plane + ij] = wprev;
}

// Write back what the scheme changed. *_s are the snapshots taken right after gather.
// HYD / HYD_S: 8 device pointers each (qc, qr, qi, qs, qg, ni, nr, qke), passed as a uint64 array.
extern "C" __global__ void adapter_scatter(
    float* T, float* Q, float* TRAC, const int* __restrict__ SLOT,
    float* CW, float* F_ICE, float* F_RAIN, const int couple_cw, const int fer_mic,
    const float* __restrict__ th, const float* __restrict__ th_s,
    const float* __restrict__ t, const float* __restrict__ t_s,
    const float* __restrict__ pii,
    const float* __restrict__ qv, const float* __restrict__ qv_s,
    const unsigned long long* __restrict__ HYD, const unsigned long long* __restrict__ HYD_S,
    const float* __restrict__ u, const float* __restrict__ u_s,
    const float* __restrict__ v, const float* __restrict__ v_s,
    float* DUH, float* DVH,
    const unsigned wr, const int IM, const int JM, const int LM, const int JOFF, const int JMG)
{
    COLUMN_IJ(IM, JM);
    const int jg = j0 + JOFF;
    if (i0 < 1 || i0 > IM - 2 || jg < 1 || jg > JMG - 2) return;   // ITS_B1..ITE_B1, JTS_B1..JTE_B1
    int slot[AD_NSPEC];
    const float* hn[AD_NSPEC];
    const float* hs[AD_NSPEC];
    for (int s = 0; s < AD_NSPEC; ++s) {
        slot[s] = SLOT[s];
        hn[s] = (const float*)HYD[s];
        hs[s] = (const float*)HYD_S[s];
    }
    const size_t ij = IDX2(i0, j0, IM);
    const size_t plane = (size_t)IM * JM;
    const size_t tstride = (size_t)LM * plane;
    for (int k = 0; k < LM; ++k) {
        const int l0 = LM - 1 - k;
        const size_t n = (size_t)l0 * plane + ij;
        const size_t a = (size_t)k * plane + ij;
        if ((wr & AD_TH) && th[a] != th_s[a]) T[n] = th[a] * pii[a];
        else if ((wr & AD_T) && t[a] != t_s[a]) T[n] = t[a];
        if ((wr & AD_QV) && qv[a] != qv_s[a]) Q[n] = qv[a] / (1.0f + qv[a]);
        bool water = false;
        for (int s = 0; s < AD_NSPEC; ++s) {
            if (!(wr & ad_spbit(s)) || slot[s] < 0) continue;
            const float x = hn[s][a];
            if (x != hs[s][a]) {
                TRAC[(size_t)slot[s] * tstride + n] = x;
                water = water || s < 5;
            }
        }
        if (water && couple_cw) {        // module_TURBULENCE.F90:1336-1350: couple CWM, F_ICE, F_RAIN
            float q5[5];
            for (int s = 0; s < 5; ++s)
                q5[s] = slot[s] >= 0 ? TRAC[(size_t)slot[s] * tstride + n] : 0.0f;
            const float qcw = q5[0], qra = q5[1], qci = q5[2], qsn = q5[3];
            const float qgr = fer_mic ? 0.0f : q5[4];
            const float cwm = qcw + qra + qci + qsn + qgr;
            CW[n] = cwm;
            F_ICE[n] = cwm > NMMB_EPSQ ? (qci + qsn + qgr) / cwm : 0.0f;
            F_RAIN[n] = qra > NMMB_EPSQ ? qra / (qcw + qra) : 0.0f;
        }
        if (wr & AD_U) DUH[n] = u[a] - u_s[a];
        if (wr & AD_V) DVH[n] = v[a] - v_s[a];
    }
}

// V(i,j) += 0.25 * (DH(i,j) + DH(i+1,j) + DH(i,j+1) + DH(i+1,j+1)), i0 < IM-1, j0 < JM-1 (H_TO_V stencil).
// A zero average leaves the V value's bits untouched (no -0 + 0 flip), so no-physics is bitwise identity.
extern "C" __global__ void adapter_h2v_add(
    float* U, const float* __restrict__ DUH, float* V, const float* __restrict__ DVH,
    const int do_u, const int do_v, const int IM, const int JM)
{
    POINT_IJL;
    if (i0 >= IM - 1 || j0 >= JM - 1) return;
    const size_t plane = (size_t)IM * JM;
    const size_t a00 = (size_t)l0 * plane + IDX2(i0, j0, IM), a10 = a00 + 1, a01 = a00 + IM, a11 = a01 + 1;
    if (do_u) {
        const float d = 0.25f * (DUH[a00] + DUH[a10] + DUH[a01] + DUH[a11]);
        if (d != 0.0f) U[a00] += d;
    }
    if (do_v) {
        const float d = 0.25f * (DVH[a00] + DVH[a10] + DVH[a01] + DVH[a11]);
        if (d != 0.0f) V[a00] += d;
    }
}

// In-place tendency add on an ArWen-layout view: x += dt * ten (fused; the tendency path of the adapter).
extern "C" __global__ void adapter_add_tend(float* x, const float* __restrict__ ten, const float dt,
                                            const unsigned long long n)
{
    const unsigned long long idx = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) x[idx] += dt * ten[idx];
}
