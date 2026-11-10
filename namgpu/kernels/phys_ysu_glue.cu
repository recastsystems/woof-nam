// phys_ysu_glue.cu -- WOOF-NAM modernisation: the NMMB TURBL seam for the selectable
// YSU PBL + revised MM5 surface layer (configfile turbulence: ysu, sfc_layer: sfclayrev).
//
// The schemes themselves are phys_sfclayrev.cu and phys_ysu_column.cu (ArWen's WRF v4.6.1
// transcriptions). These kernels only move values between NMMB's TURBL names and the WRF
// argument names, in the WRF surface-driver order: SFCLAY -> NOAHLSM -> SFCDIAGS -> PBL.
// One thread per (i, j) column, NMMB layout [LM, JM, IM], level 0 = model top.
#include "nmmb_common.cuh"

// (1) before SFCLAY: lowest-level water vapour as a MIXING RATIO (NMMB Q is specific humidity; the
//     MM5 scheme forms its surface saturation value qs = ep2 es / (p - es) as a mixing ratio, so the
//     flux qs - qv must compare like with like).
extern "C" __global__ void sfclayrev_in(
    const float* __restrict__ Q,       // [LM,JM,IM] specific humidity
    float* __restrict__ QV1,           // [JM,IM] OUT lowest-level mixing ratio
    float* __restrict__ LAKE,          // [JM,IM] OUT zero (no lake model on the NMMB route)
    const int im, const int jm, const int lm)
{
    COLUMN_IJ(im, jm)
    const size_t c2 = IDX2(i0, j0, im);
    const float q = Q[IDX3(i0, j0, lm - 1, im, jm)];
    QV1[c2] = q / (1.0f - q);
    LAKE[c2] = 0.0f;
}

// (2) after SFCLAY, before NOAHLSM: the fluxes the LSM starts from (water points keep them), and the
//     MYJ-era exchange names the history and the max fields read.
extern "C" __global__ void sfclayrev_out(
    const float* __restrict__ SL_HFX,  // [JM,IM] sfclay HFX (W m-2, up)
    const float* __restrict__ SL_QFX,  // [JM,IM] sfclay QFX (kg m-2 s-1, up)
    const float* __restrict__ CHS,     // [JM,IM] sfclay CHS (m s-1)
    const float* __restrict__ UST,     // [JM,IM]
    const float* __restrict__ WSPD,    // [JM,IM]
    float* __restrict__ TWBS,          // [JM,IM] OUT HFX handed to NOAHLSM
    float* __restrict__ QWBS,          // [JM,IM] OUT QFX handed to NOAHLSM
    float* __restrict__ AKHS,          // [JM,IM] OUT heat exchange coefficient (m s-1)
    float* __restrict__ AKMS,          // [JM,IM] OUT momentum exchange coefficient (m s-1) = u*^2 / U
    const int im, const int jm)
{
    COLUMN_IJ(im, jm)
    // B1 tile only, as JSFC writes: the outer rim keeps the zero fluxes TURBL set (no LSM runs there, so a rim
    // SFCLAY flux against a stale skin temperature would otherwise show in the history).
    if (!(i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2)) return;
    const size_t c2 = IDX2(i0, j0, im);
    TWBS[c2] = SL_HFX[c2];
    QWBS[c2] = SL_QFX[c2];
    AKHS[c2] = CHS[c2];
    const float u = UST[c2];
    AKMS[c2] = u * u / fmaxf(WSPD[c2], 0.1f);
}

// (3) after SFCDIAGS: the 2 m and 10 m shelter fields the history writes (T2 = TSHLTR (PSHLTR/1e5)^kappa,
//     Q2 = QSHLTR as specific humidity), from SFCDIAGS as WRF's Noah route does; and the post-LSM fluxes
//     kept for the next SFCLAY call (WRF passes HFX/QFX INOUT from the LSM back into SFCLAY).
extern "C" __global__ void sfclayrev_diag(
    const float* __restrict__ TH2X,    // [JM,IM] SFCDIAGS 2 m theta
    const float* __restrict__ Q2X,     // [JM,IM] SFCDIAGS 2 m mixing ratio
    const float* __restrict__ PHINT,   // [LM+1,JM,IM]
    const float* __restrict__ TH,      // [LM,JM,IM] theta
    const float* __restrict__ Q,       // [LM,JM,IM] specific humidity
    const float* __restrict__ DZ,      // [LM,JM,IM]
    const float* __restrict__ TWBS,    // [JM,IM] post-LSM HFX
    const float* __restrict__ QWBS,    // [JM,IM] post-LSM QFX
    float* __restrict__ TSHLTR,        // [JM,IM] OUT 2 m theta
    float* __restrict__ QSHLTR,        // [JM,IM] OUT 2 m specific humidity
    float* __restrict__ PSHLTR,        // [JM,IM] OUT pressure used with TSHLTR (the surface pressure)
    float* __restrict__ TH10,          // [JM,IM] OUT 10 m theta (linear between 2 m and the lowest level)
    float* __restrict__ Q10,           // [JM,IM] OUT 10 m specific humidity (same)
    float* __restrict__ SL_HFX,        // [JM,IM] OUT kept for the next SFCLAY call
    float* __restrict__ SL_QFX,        // [JM,IM] OUT
    const int im, const int jm, const int lm)
{
    COLUMN_IJ(im, jm)
    const int b1 = (i0 >= 1 && i0 <= im - 2 && j0 >= 1 && j0 <= jm - 2);
    const size_t c2 = IDX2(i0, j0, im);
    SL_HFX[c2] = TWBS[c2];
    SL_QFX[c2] = QWBS[c2];
    if (!b1) return;                   // SFCDIAGS computes the B1 tile; turbl_edge_fill fills the rim
    const size_t k1 = IDX3(i0, j0, lm - 1, im, jm);
    const float qlow = Q[k1];
    float r2 = Q2X[c2];
    float q2 = (r2 > 0.0f) ? r2 / (1.0f + r2) : qlow;   // ArWen/WRF commented remedy: Q2 = qv(1) when the
                                                         // flux inversion leaves the physical range
    const float th2 = TH2X[c2];
    TSHLTR[c2] = th2;
    QSHLTR[c2] = q2;
    PSHLTR[c2] = PHINT[IDX3(i0, j0, lm, im, jm)];
    const float za = 0.5f * DZ[k1];
    const float w = (za > 2.0f) ? fminf(fmaxf((10.0f - 2.0f) / (za - 2.0f), 0.0f), 1.0f) : 1.0f;
    TH10[c2] = th2 + w * (TH[k1] - th2);
    Q10[c2] = q2 + w * (qlow - q2);
}

// (0) before SFCLAY: the roughness length the MM5 scheme and YSU see. Over land WRF's Noah route hands SFCLAY the
//     vegetation-table roughness (ZNT = Z0 of the land-use class), not NMMB's Z0, which carries NAM's orographic
//     roughness (up to 2.75 m on the 12 km grid). That NMMB Z0 in the MM5 log law gave a 10 m wind slow bias that
//     grew with Z0 (+6 h vs 2,174 ASOS: -1.3 kt at Z0 0.1-0.3 m, -2.8 kt at Z0 > 1.5 m). Water keeps the scheme's
//     own Charnock roughness, carried step to step in Z0 as before.
extern "C" __global__ void sfclayrev_znt(
    const int* __restrict__ IVGTYP,    // [JM,IM] vegetation class, 1-based
    const float* __restrict__ XLAND,   // [JM,IM] 1 land (sea ice included), 2 water
    const float* __restrict__ Z0TBL,   // [NLUS] Noah background roughness by class (phys-noah-redprm table)
    const float* __restrict__ Z0,      // [JM,IM] NMMB Z0 (water: the Charnock value of the previous call)
    float* __restrict__ ZNT,           // [JM,IM] OUT roughness for SFCLAY (INOUT there) and YSU
    const int nlus, const int im, const int jm)
{
    COLUMN_IJ(im, jm)
    const size_t c2 = IDX2(i0, j0, im);
    if (XLAND[c2] < 1.5f) {
        const int vt = min(max(IVGTYP[c2], 1), nlus) - 1;
        ZNT[c2] = Z0TBL[vt];
    } else {
        ZNT[c2] = Z0[c2];
    }
}

// (0b) after SFCLAY: water points keep the scheme's Charnock roughness in NMMB Z0 (land Z0 is NAM's and untouched).
extern "C" __global__ void sfclayrev_znt_back(
    const float* __restrict__ XLAND, const float* __restrict__ ZNT, float* __restrict__ Z0,
    const int im, const int jm)
{
    COLUMN_IJ(im, jm)
    const size_t c2 = IDX2(i0, j0, im);
    if (XLAND[c2] >= 1.5f) Z0[c2] = ZNT[c2];
}
