// phys_rrtm_progcld8.cu -- WOOF-NAM unit phys-rrtm-progcld8: PROGCLD8, Thompson-family prognostic cloud
// optical properties for the RRTM radiation (np3d = 8), used by the TEMPO microphysics (modernise/mp-tempo).
//
// NOTICE: NOAA/NCEP NMMB source, US government work. Ported to CUDA from phys/radiation_clouds_nmmb.f:1329-1542
// (nam.v4.2.11, 83df4e13c1), SUBROUTINE progcld8, with gethml (:2167-2494) inlined exactly as in
// phys_rrtm_progcld2.cu (copied from that kernel).
// The one change from the Fortran: calc_effectRad (Thompson's radii) is replaced by the effective radii the
// TEMPO microphysics computed for the same species on its last call (re_cloud, re_ice, re_snow, metres),
// read from RE_CLOUD / RE_ICE / RE_SNOW. Where TEMPO has no radius the Fortran initial values 2.501, 10.01,
// 25.01 micron are kept, and the Fortran floors apply.
// Layout: clouds [NF_CLDS, NLAY, JM, IM], NLAY = LM + LTP (extra top layer at k0 = 0 when ivflip = 0); the RE_*
// state arrays are [LM, JM, IM] model layers, model layer l0 = k0 - LTP.
#include "nmmb_common.cuh"
#define PGC_NF_CLDS    9
#define PGC_MAXLAY     128
#define PGC_CON_PI     3.1415926535897931f
#define PGC_CON_G      9.80665f
#define PGC_CON_RD     287.05f
#define PGC_CON_RV     461.5f
#define PGC_CON_FVIRT  (PGC_CON_RV / PGC_CON_RD - 1.0f)
#define PGC_CON_T0C    273.15f
#define PGC_CON_TTP    273.16f
#define PGC_CCLIMIT     0.001f
#define PGC_CCLIMIT2    0.05f
#define PGC_CLIMIT      0.001f
#define PGC_OVCST       (1.0f - 1.0e-8f)
#define PGC_RECWAT_DEF  5.0f
#define PGC_RECICE_DEF  10.0f
#define PGC_RERAIN_DEF  100.0f
#define PGC_RE_50C      (1250.0f / 9.917f)
#define PGC_RE_40C      (1250.0f / 9.337f)
#define PGC_RE_30C      (1250.0f / 9.208f)
#define PGC_RE_20C      (1250.0f / 9.387f)
#define PGC_CEXP        (1.0f / 3.0f)
#define PGC_EPSQ        1.0e-12f
#define PGC_CN0R0       2511.54f
#define PGC_TNW         200.0f

extern "C" __global__ void phys_rrtm_progcld8_column(
    const int im, const int jm, const int nlay, const int lm,
    const int ivflip, const int llyr, const int iovr, const int cld_fraction,
    const float* __restrict__ plyr, const float* __restrict__ plvl, const float* __restrict__ tlyr,
    const float* __restrict__ qlyr, const float* __restrict__ qc, const float* __restrict__ qi,
    const float* __restrict__ qs, const float* __restrict__ cldfk1, const float* __restrict__ xlat,
    const float* __restrict__ re_c, const float* __restrict__ re_i, const float* __restrict__ re_s,
    float* __restrict__ clouds, float* __restrict__ clds, int* __restrict__ mtop, int* __restrict__ mbot)
{
    COLUMN_IJ(im, jm)
    #define CT1(k)  cldtot[(k) - 1]
    #define PL1(k)  plyr[IDX3(i0, j0, (k) - 1, im, jm)]
    const int ltp = nlay - lm;
    float ptop1[4];
    {
        const float ptopc1[4] = {1050.0f, 650.0f, 400.0f, 0.0f};
        const float ptopc2[4] = {1050.0f, 750.0f, 500.0f, 0.0f};
        const float tem2 = fmaxf(0.0f, 4.0f * fabsf(xlat[IDX2(i0, j0, im)]) / PGC_CON_PI - 1.0f);
        for (int id = 0; id < 4; ++id) ptop1[id] = ptopc1[id] + (ptopc2[id] - ptopc1[id]) * tem2;
    }
    float cldtot[PGC_MAXLAY];
    bool any = false;
    for (int k0 = 0; k0 < nlay; ++k0) {
        const size_t p = IDX3(i0, j0, k0, im, jm);
        cldtot[k0] = (cld_fraction == 0) ? 0.0f : cldfk1[p];
        for (int f = 0; f < PGC_NF_CLDS; ++f) clouds[IDX4(i0, j0, k0, f, im, jm, nlay)] = 0.0f;
        if (qc[p] + qi[p] + qs[p] > 1.0e-9f) { any = true; cldtot[k0] = 1.0f; }
    }
    if (any) {
        for (int k0 = 0; k0 < nlay; ++k0) {
            const size_t p = IDX3(i0, j0, k0, im, jm);
            const float t1 = tlyr[p], p1 = plyr[p] * 100.0f;
            const float qv1 = fmaxf(1.0e-8f, qlyr[p] / (1.0f - qlyr[p]));
            const int l0 = (ivflip == 0) ? (k0 - ltp) : (lm - 1 - k0);
            float rc = 2.501e-6f, ri = 10.01e-6f, rs = 25.01e-6f;
            if (l0 >= 0 && l0 < lm) {
                const size_t q = IDX3(i0, j0, l0, im, jm);
                if (re_c[q] > 0.0f) rc = re_c[q];
                if (re_i[q] > 0.0f) ri = re_i[q];
                if (re_s[q] > 0.0f) rs = re_s[q];
            }
            const float rho_k = 0.622f * p1 / (287.05f * t1 * (qv1 + 0.622f));
            const float cpath = fabsf(plvl[IDX3(i0, j0, k0 + 1, im, jm)] - plvl[p]) * (100000.0f / (rho_k * 9.8f));
            clouds[IDX4(i0, j0, k0, 0, im, jm, nlay)] = cldtot[k0];
            clouds[IDX4(i0, j0, k0, 1, im, jm, nlay)] = fmaxf(0.0f, qc[p]) * cpath;
            clouds[IDX4(i0, j0, k0, 2, im, jm, nlay)] = fmaxf(2.501f, rc * 1.0e6f);
            clouds[IDX4(i0, j0, k0, 3, im, jm, nlay)] = fmaxf(0.0f, qi[p]) * cpath;
            clouds[IDX4(i0, j0, k0, 4, im, jm, nlay)] = fmaxf(10.01f, ri * 1.0e6f);
            float csp = fmaxf(0.0f, qs[p]) * cpath, res = fmaxf(25.01f, rs * 1.0e6f);
            if (csp > 0.0f && res > 130.0f) {
                const float smf = (130.0e-6f / rs) * (130.0e-6f / rs);
                csp = smf * qs[p] * cpath;
                res = 130.0f;
            }
            clouds[IDX4(i0, j0, k0, 7, im, jm, nlay)] = csp;
            clouds[IDX4(i0, j0, k0, 8, im, jm, nlay)] = res;
        }
    }

    // ---- gethml (radiation_clouds_nmmb.f:2167-2494): cloud layer fractions and indices --------
    float cl[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};    // clds(i,1..5)
    int mt[4] = {0, 0, 0, 0};                         // mtop(i,1..3); slot 4 is a guard only
    int mb[4] = {0, 0, 0, 0};

    // total and bl clouds (gethml: random or max/ran overlap, layers from the top or surface).
    {
        float cl1 = 1.0f, cl2 = 1.0f;
        for (int kk = 0; kk < nlay; ++kk) {
            const int k = (ivflip == 0) ? (nlay - kk) : (kk + 1);   // Fortran 1-based
            const float ccur = fminf(PGC_OVCST, fmaxf(CT1(k), 0.0f));  // cldcnv = 0
            if (iovr == 0) {
                if (ccur >= PGC_CLIMIT) cl1 = cl1 * (1.0f - ccur);
                if (k == llyr) cl[4] = 1.0f - cl1;
            } else {
                if (ccur >= PGC_CLIMIT) {
                    cl2 = fminf(cl2, 1.0f - ccur);
                } else {
                    cl1 = cl1 * cl2;
                    cl2 = 1.0f;
                }
                if (k == llyr) cl[4] = 1.0f - cl1 * cl2;
            }
        }
        if (iovr == 0) {
            cl[3] = 1.0f - cl1;
        } else {
            cl[3] = 1.0f - cl1 * cl2;
        }
    }

    // high, mid, low clouds and their top and base layer indices.
    {
        float cl1 = 0.0f, cl2 = 0.0f;
        int kbt1, kbt2, kth1 = 0, kth2 = 0, idom = 1;
        if (ivflip == 0) {
            kbt1 = nlay; kbt2 = nlay;
            mb[0] = nlay; mt[0] = nlay;
            for (int k = nlay; k >= 1; --k) {
                int id = idom;
                int id1 = id + 1;
                const float pcur = PL1(k);
                const float ccur = fminf(PGC_OVCST, fmaxf(CT1(k), 0.0f));
                float pnxt, cnxt;
                if (k > 1) {
                    pnxt = PL1(k - 1);
                    cnxt = fminf(PGC_OVCST, fmaxf(CT1(k - 1), 0.0f));
                } else {
                    pnxt = -1.0f;
                    cnxt = 0.0f;
                }
                if (pcur < ptop1[id1 - 1]) {
                    id += 1;
                    id1 += 1;
                    idom = id;
                }
                if (ccur >= PGC_CLIMIT) {
                    if (kth2 == 0) kbt2 = k;
                    kth2 += 1;
                    if (iovr == 0) {
                        cl2 = cl2 + ccur - cl2 * ccur;
                    } else {
                        cl2 = fmaxf(cl2, ccur);
                    }
                    if (cnxt < PGC_CLIMIT || pnxt < ptop1[id1 - 1]) {
                        kbt1 = (int)roundf((cl1 * (float)kbt1 + cl2 * (float)kbt2) / (cl1 + cl2));
                        kth1 = (int)roundf((cl1 * (float)kth1 + cl2 * (float)kth2) / (cl1 + cl2));
                        cl1 = cl1 + cl2 - cl1 * cl2;
                        kbt2 = k - 1;
                        kth2 = 0;
                        cl2 = 0.0f;
                    }
                }
                if (pnxt < ptop1[id1 - 1]) {
                    cl[id - 1] = cl1;
                    mt[id - 1] = min(kbt1, kbt1 - kth1 + 1);
                    mb[id - 1] = kbt1;
                    cl1 = 0.0f;
                    kbt1 = k - 1;
                    kth1 = 0;
                    if (id1 <= 3) {
                        mb[id1 - 1] = kbt1;
                        mt[id1 - 1] = kbt1;
                    }
                }
            }
        } else {
            kbt1 = 1; kbt2 = 1;
            mb[0] = 1; mt[0] = 1;
            for (int k = 1; k <= nlay; ++k) {
                int id = idom;
                int id1 = id + 1;
                const float pcur = PL1(k);
                const float ccur = fminf(PGC_OVCST, fmaxf(CT1(k), 0.0f));
                float pnxt, cnxt;
                if (k < nlay) {
                    pnxt = PL1(k + 1);
                    cnxt = fminf(PGC_OVCST, fmaxf(CT1(k + 1), 0.0f));
                } else {
                    pnxt = -1.0f;
                    cnxt = 0.0f;
                }
                if (pcur < ptop1[id1 - 1]) {
                    id += 1;
                    id1 += 1;
                    idom = id;
                }
                if (ccur >= PGC_CLIMIT) {
                    if (kth2 == 0) kbt2 = k;
                    kth2 += 1;
                    if (iovr == 0) {
                        cl2 = cl2 + ccur - cl2 * ccur;
                    } else {
                        cl2 = fmaxf(cl2, ccur);
                    }
                    if (cnxt < PGC_CLIMIT || pnxt < ptop1[id1 - 1]) {
                        kbt1 = (int)roundf((cl1 * (float)kbt1 + cl2 * (float)kbt2) / (cl1 + cl2));
                        kth1 = (int)roundf((cl1 * (float)kth1 + cl2 * (float)kth2) / (cl1 + cl2));
                        cl1 = cl1 + cl2 - cl1 * cl2;
                        kbt2 = k + 1;
                        kth2 = 0;
                        cl2 = 0.0f;
                    }
                }
                if (pnxt < ptop1[id1 - 1]) {
                    cl[id - 1] = cl1;
                    mt[id - 1] = max(kbt1, kbt1 + kth1 - 1);
                    mb[id - 1] = kbt1;
                    cl1 = 0.0f;
                    kbt1 = k + 1;
                    kth1 = 0;
                    if (id1 <= 3) {
                        mb[id1 - 1] = kbt1;
                        mt[id1 - 1] = kbt1;
                    }
                }
            }
        }
    }

    for (int c = 0; c < 5; ++c) {
        clds[IDX3(i0, j0, c, im, jm)] = cl[c];
    }
    for (int c = 0; c < 3; ++c) {
        mtop[IDX3(i0, j0, c, im, jm)] = mt[c];
        mbot[IDX3(i0, j0, c, im, jm)] = mb[c];
    }

    #undef CT1
    #undef PL1
}
