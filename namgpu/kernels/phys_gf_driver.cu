// WOOF-NAM unit phys-gf-driver: the NMMB side of the Grell-Freitas cumulus option (convection: gf).
//
// NOTICE: written for WOOF-NAM (Apache-2.0). The scheme itself is unit phys-gf-gf
// (namgpu/kernels/phys_gf_gf.cu), ArWen's CUDA port of WRF v4.6.1 GFDRV (module_cu_gf_wrfdrv.F,
// module_cu_gf_deep.F, module_cu_gf_sh.F; WRF public-domain lineage, G. Grell and S. Freitas,
// NOAA ESRL/GSD). This file only maps an NMMB B-grid column onto GFDRV's column inputs and
// GFDRV's outputs back onto CUCNVC's arrays.
//
// Column map (WRF k bottom-up 1..LM <-> NMMB l0 top-down 0..LM-1: l0 = LM - k):
//   u, v      U_PHY, V_PHY (CUCNVC's 4-point H averages)     t      T
//   w         W (midlayer, m/s)                                qv     Q/(1-Q)  (GFDRV wants mixing ratio)
//   p         PHMID      pi  EXNER      rho  RR      dz8w  DZ      p8w  PHINT(l0+1) (the layer's lower interface)
//   forcing   tn = t + (rthften + rthraten + rthblten)*pi*dt, qo = q + (rqvften + rqvblten)*dt.
//             The NMMB physics applies its tendencies to T and Q in place, so the non-convective
//             forcing since the last convection step is measured as (T - GF_TSNAP)/dt_cu, GF_TSNAP
//             being T after the previous convection step's microphysics: dynamics + radiation + PBL,
//             no microphysics and no convection (WRF's forcing set). rthblten/rqvblten (which drive
//             the shallow closure through dhdt and tshall) are the surface fluxes spread over the PBL
//             depth; the advective lane carries the rest, so the three lanes still sum to the
//             measured total. rthraten is 0 (radiation is inside the measured total).
//   surface   ht = FIS/g, hfx = -TWBS, qfx = -QWBS/XLV (after TURBL, TWBS and QWBS are W m-2 positive toward
//             the surface: SFCEVP grows when QWBS < 0, phys_turbl_driver.cu; GFDRV wants upward fluxes),
//             xland = XLAND (1 land, 2 water),
//             dx = DXH(j), dt = NPRECIP*DT, kpbl = LM + 1 - LPBL.
// Outputs: RTHCUTEN (theta tendency), RQCUTEN (specific-humidity tendency), DUDT_PHY/DVDT_PHY
// (momentum, H points, applied by drv-h-to-v-tend), RAINCV (mm per model step: CUCNVC's post
// multiplies by NCNVC), CUTOP/CUBOT (bottom-up level, CUCNVC's convention), and the detrained
// condensate (rqccuten + rqicuten) added to CW with F_ICE/F_RAIN kept consistent (Ferrier-Aligo
// reads only CW, F_ICE, F_RAIN on the spec_adv=false route).
#include "nmmb_common.cuh"

#define GF_NIN_LEV 15
#define GF_NIN_SCA 6
#define GF_NLEV 16
#define GF_NSCA 9
#define GF_NISCA 6
#define GF_XLV 2.501e6f

extern "C" __global__ void gf_snapshot(const int n, const float* __restrict__ T, const float* __restrict__ Q,
                                       float* __restrict__ TS, float* __restrict__ QS)
{
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n) return;
    TS[i] = T[i];
    QS[i] = Q[i];
}

extern "C" __global__ void gf_pack(
    const int IM, const int JM, const int LM, const int lo, const int n, const int fvalid, const float dtc,
    const int ishallow, const int ichoice,
    const float* __restrict__ U_PHY, const float* __restrict__ V_PHY, const float* __restrict__ W,
    const float* __restrict__ T, const float* __restrict__ Q,
    const float* __restrict__ PHMID, const float* __restrict__ EXNER, const float* __restrict__ RR,
    const float* __restrict__ DZ, const float* __restrict__ PHINT,
    const float* __restrict__ TSNAP, const float* __restrict__ QSNAP,
    const float* __restrict__ FIS, const float* __restrict__ TWBS, const float* __restrict__ QWBS,
    const float* __restrict__ XLAND, const float* __restrict__ DXH, const int* __restrict__ KPBL,
    float* __restrict__ lvin, float* __restrict__ scin, int* __restrict__ iin)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const int col = lo + c;
    const int i0 = col % IM;
    const int j0 = col / IM;
    const size_t p2 = IDX2(i0, j0, IM);
    float* lv = lvin + (size_t)c * GF_NIN_LEV * LM;

    int kp = LM + 1 - KPBL[p2];
    kp = max(1, min(kp, LM - 1));
    const float psfc = PHINT[IDX3(i0, j0, LM, IM, JM)];
    const float ptop = PHINT[IDX3(i0, j0, LM - kp, IM, JM)];
    const float dp = fmaxf(psfc - ptop, 100.0f);
    const float hfx = -TWBS[p2];                 // upward sensible heat flux, W m-2
    const float qfx = -QWBS[p2] / GF_XLV;        // upward moisture flux, kg m-2 s-1
    const float dtdt_bl = NMMB_G * hfx / (NMMB_CP * dp);
    const float dqdt_bl = NMMB_G * qfx / dp;
    const float rdtc = 1.0f / dtc;

    for (int k = 1; k <= LM; ++k) {
        const int l0 = LM - k;
        const size_t pk = IDX3(i0, j0, l0, IM, JM);
        const float q = fminf(fmaxf(Q[pk], 0.0f), 0.1f);
        const float r = q / (1.0f - q);
        const float pi = EXNER[pk];
        const float rthbl = (k <= kp) ? dtdt_bl / pi : 0.0f;
        const float rqvbl = (k <= kp) ? dqdt_bl : 0.0f;
        float rthf = 0.0f, rqvf = 0.0f;
        if (fvalid) {
            const float qs = fminf(fmaxf(QSNAP[pk], 0.0f), 0.1f);
            const float rs = qs / (1.0f - qs);
            rthf = (T[pk] - TSNAP[pk]) * rdtc / pi - rthbl;
            rqvf = (r - rs) * rdtc - rqvbl;
        }
        const int kk = k - 1;
        lv[0 * LM + kk] = U_PHY[pk];
        lv[1 * LM + kk] = V_PHY[pk];
        lv[2 * LM + kk] = W[pk];
        lv[3 * LM + kk] = T[pk];
        lv[4 * LM + kk] = r;
        lv[5 * LM + kk] = PHMID[pk];
        lv[6 * LM + kk] = pi;
        lv[7 * LM + kk] = RR[pk];
        lv[8 * LM + kk] = DZ[pk];
        lv[9 * LM + kk] = PHINT[IDX3(i0, j0, l0 + 1, IM, JM)];
        lv[10 * LM + kk] = rthf;
        lv[11 * LM + kk] = rqvf;
        lv[12 * LM + kk] = 0.0f;
        lv[13 * LM + kk] = rthbl;
        lv[14 * LM + kk] = rqvbl;
    }
    float* sc = scin + (size_t)c * GF_NIN_SCA;
    sc[0] = FIS[p2] / NMMB_G;
    sc[1] = hfx;
    sc[2] = qfx;
    sc[3] = XLAND[p2];
    sc[4] = dtc;
    sc[5] = DXH[j0];
    int* ii = iin + (size_t)c * 3;
    ii[0] = kp;
    ii[1] = ishallow;
    ii[2] = ichoice;
}

extern "C" __global__ void gf_unpack(
    const int IM, const int JM, const int LM, const int lo, const int n, const float dt, const float dtc,
    const int momentum,
    const float* __restrict__ lev, const float* __restrict__ sca, const int* __restrict__ isca,
    const float* __restrict__ Q,
    float* __restrict__ RTHCUTEN, float* __restrict__ RQCUTEN,
    float* __restrict__ DUDT_PHY, float* __restrict__ DVDT_PHY,
    float* __restrict__ RAINCV, float* __restrict__ CUTOP, float* __restrict__ CUBOT,
    float* __restrict__ CW, float* __restrict__ F_ICE, float* __restrict__ F_RAIN,
    float* __restrict__ GF_DEEP, float* __restrict__ GF_SHAL)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    const int col = lo + c;
    const int i0 = col % IM;
    const int j0 = col / IM;
    const size_t p2 = IDX2(i0, j0, IM);
    const bool interior = (i0 >= 1 && i0 <= IM - 2 && j0 >= 1 && j0 <= JM - 2);
    if (!interior) return;
    const float* lv = lev + (size_t)c * GF_NLEV * LM;
    const float* sc = sca + (size_t)c * GF_NSCA;
    const int* is = isca + (size_t)c * GF_NISCA;

    for (int k = 1; k <= LM; ++k) {
        const int l0 = LM - k;
        const size_t pk = IDX3(i0, j0, l0, IM, JM);
        const int kk = k - 1;
        const float q = fminf(fmaxf(Q[pk], 0.0f), 0.1f);
        const float r = q / (1.0f - q);
        const float opr = 1.0f + r;
        RTHCUTEN[pk] = lv[0 * LM + kk];
        RQCUTEN[pk] = lv[1 * LM + kk] / (opr * opr);       // d(q)/dt = d(r)/dt / (1+r)^2
        if (momentum) {
            DUDT_PHY[pk] = lv[4 * LM + kk];
            DVDT_PHY[pk] = lv[5 * LM + kk];
        }
        const float dqc = lv[2 * LM + kk] * dtc;
        const float dqi = lv[3 * LM + kk] * dtc;
        if (dqc != 0.0f || dqi != 0.0f) {
            const float cw = fmaxf(CW[pk], 0.0f);
            const float fi = fminf(fmaxf(F_ICE[pk], 0.0f), 1.0f);
            const float fr = fminf(fmaxf(F_RAIN[pk], 0.0f), 1.0f);
            const float ice = fi * cw;
            const float liq = cw - ice;
            const float rain = fr * liq;
            const float cld = liq - rain;
            const float ice2 = fmaxf(ice + dqi, 0.0f);
            const float cld2 = fmaxf(cld + dqc, 0.0f);
            const float cw2 = ice2 + cld2 + rain;
            CW[pk] = cw2;
            if (cw2 > NMMB_EPSQ) F_ICE[pk] = ice2 / cw2;
            const float liq2 = cld2 + rain;
            if (liq2 > NMMB_EPSQ) F_RAIN[pk] = rain / liq2;
        }
    }
    const float pratec = sc[1];
    RAINCV[p2] = pratec * dt;                                  // mm per model step (CUCNVC post: * NCNVC)
    const int kbcon = is[4], ktop = is[5];
    const int kts = is[3], kbs = is[2];
    if (pratec > 0.0f && ktop > 0 && kbcon > 0) {
        CUTOP[p2] = (float)ktop;
        CUBOT[p2] = (float)kbcon;
        GF_DEEP[p2] += 1.0f;
    } else if (kts > 0 && kbs > 0 && kts > kbs) {
        CUTOP[p2] = (float)kts;
        CUBOT[p2] = (float)kbs;
        GF_SHAL[p2] += 1.0f;
    }
}
