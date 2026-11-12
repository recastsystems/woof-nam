// io_history_fields.cu -- wrfout-shaped history fields from the NMMB state (unit io-history-fields,
// mode "new": no Fortran counterpart; the mapping follows the NAM sources named below).
//
// Ported to CUDA for WOOF-NAM from nam.v4.2.11 (83df4e13c1) sorc/nam_nems_nmmb.fd/src/atmos/nmm:
//   module_SOLVER_GRID_COMP.F90:11144-11186  UPDATE_WATER, CASE ('fer','fer_hires'), spec_adv=F branch:
//                                            QC=(1-F_RAIN)*(1-F_ICE)*CWM, QR=F_RAIN*(1-F_ICE)*CWM,
//                                            QS=F_ICE*CWM, all zero when CWM <= EPSQ. QI is not set.
//   Z is MIDLAYER height (module_DYNAMICS_ROUTINES.F90:3075, z(i,j,l)=zlo+dz/2; dyn-cdzdt-nonhydro writes it
//   every step). Z[LM] is not used: the surface interface is FIS/g. See io_history_fields.py header.
//   Constants: NMMB_G (module_CONSTANTS g), CAPPA = R_D/CP, EPSQ (nmmb_common.cuh). P00 = 1.0e5 Pa is the
//   WRF theta reference (not an NMMB constant; used only for the output theta).
//   Pressure at mid layers: PMID(l) = PSGML1(l) + SGML2(l)*PD (grid.py psgml1, same as PGFORCE), PINT as state.
//
// Layout: state arrays are [l, j, i] (interfaces LM+1), PD/FIS/ACPREC_TOT/CUPREC_TOT/UE10/VE10 are [j, i].
// Mixing ratios: NMMB Q and CW are specific (kg/kg); WRF outputs are mixing ratios, so each moisture
// output is x/(1-Q). Vapour and condensate species use the same factor.
#include "nmmb_common.cuh"

#ifndef NMMB_P00
#define NMMB_P00 100000.0f
#endif

// Mid-layer fields: pressure, theta (written as theta-300, WRF's T), vapour and condensate mixing ratios.
// QICE is zero under fer_hires (ice is all QS, see UPDATE_WATER above); kept so the output set is WRF's.
extern "C" __global__ void hist_moist_theta(
    int IM, int JM, int LM,
    const float* __restrict__ T, const float* __restrict__ Q, const float* __restrict__ CW,
    const float* __restrict__ F_ICE, const float* __restrict__ F_RAIN,
    const float* __restrict__ PD,
    const float* __restrict__ PSGML1, const float* __restrict__ SGML2,
    float* __restrict__ P_OUT, float* __restrict__ T_OUT,
    float* __restrict__ QVAPOR, float* __restrict__ QCLOUD, float* __restrict__ QRAIN,
    float* __restrict__ QICE, float* __restrict__ QSNOW)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t n = IDX3(i0, j0, l0, IM, JM);
    const size_t n2 = IDX2(i0, j0, IM);

    const float pmid = PSGML1[l0] + SGML2[l0] * PD[n2];
    const float t = T[n];
    const float q = Q[n];
    const float cw = CW[n];
    const float fi = F_ICE[n];
    const float fr = F_RAIN[n];
    const float inv = 1.0f / (1.0f - q);

    const float theta = t * powf(NMMB_P00 / pmid, NMMB_CAPPA);

    float qc = 0.0f, qr = 0.0f, qs = 0.0f;
    if (cw > NMMB_EPSQ) {
        const float liqw = (1.0f - fi) * cw;
        qc = (1.0f - fr) * liqw;
        qr = fr * liqw;
        qs = fi * cw;
    }

    P_OUT[n]   = pmid;
    T_OUT[n]   = theta - 300.0f;
    QVAPOR[n]  = q * inv;
    QCLOUD[n]  = qc * inv;
    QRAIN[n]   = qr * inv;
    QICE[n]    = 0.0f;
    QSNOW[n]   = qs * inv;
}

// hist_moist_theta with one output per launch (io_history_fields.StreamField): the same expressions, stored only
// for WHICH (0 P, 1 T, 2 QVAPOR, 3 QCLOUD, 4 QRAIN, 5 QICE, 6 QSNOW), so the writer can stream one 3-D buffer.
extern "C" __global__ void hist_moist_one(
    int IM, int JM, int LM, int WHICH,
    const float* __restrict__ T, const float* __restrict__ Q, const float* __restrict__ CW,
    const float* __restrict__ F_ICE, const float* __restrict__ F_RAIN,
    const float* __restrict__ PD,
    const float* __restrict__ PSGML1, const float* __restrict__ SGML2,
    float* __restrict__ OUT)
{
    POINT_IJL
    if (i0 >= IM || j0 >= JM || l0 >= LM) return;
    const size_t n = IDX3(i0, j0, l0, IM, JM);
    const size_t n2 = IDX2(i0, j0, IM);

    const float pmid = PSGML1[l0] + SGML2[l0] * PD[n2];
    const float t = T[n];
    const float q = Q[n];
    const float cw = CW[n];
    const float fi = F_ICE[n];
    const float fr = F_RAIN[n];
    const float inv = 1.0f / (1.0f - q);

    const float theta = t * powf(NMMB_P00 / pmid, NMMB_CAPPA);

    float qc = 0.0f, qr = 0.0f, qs = 0.0f;
    if (cw > NMMB_EPSQ) {
        const float liqw = (1.0f - fi) * cw;
        qc = (1.0f - fr) * liqw;
        qr = fr * liqw;
        qs = fi * cw;
    }

    switch (WHICH) {
        case 0: OUT[n] = pmid; break;
        case 1: OUT[n] = theta - 300.0f; break;
        case 2: OUT[n] = q * inv; break;
        case 3: OUT[n] = qc * inv; break;
        case 4: OUT[n] = qr * inv; break;
        case 5: OUT[n] = 0.0f; break;
        default: OUT[n] = qs * inv; break;
    }
}

// Geopotential at the LM+1 interfaces, WRF PH on bottom_top_stag: PH = g * zint. The state Z is MIDLAYER
// height (dyn-cdzdt-nonhydro, Fortran z = zlo + dz/2; init-derive-state writes the same convention), so the
// interfaces are rebuilt from the surface: zint(LM) = FIS/g, zint(l) = 2*Z(l) - zint(l+1). One thread per column.
extern "C" __global__ void hist_ph(
    int IM, int JM, int LM,
    const float* __restrict__ Z, const float* __restrict__ FIS, float* __restrict__ PH)
{
    COLUMN_IJ(IM, JM)
    float zb = FIS[IDX2(i0, j0, IM)] * (1.0f / NMMB_G);
    PH[IDX3(i0, j0, LM, IM, JM)] = NMMB_G * zb;
    for (int l0 = LM - 1; l0 >= 0; --l0) {
        const float zt = 2.0f * Z[IDX3(i0, j0, l0, IM, JM)] - zb;
        PH[IDX3(i0, j0, l0, IM, JM)] = NMMB_G * zt;
        zb = zt;
    }
}

// Surface fields: PSFC = PINT(LM+1) (bottom interface), HGT = FIS/g. Precipitation from the running totals
// since start (io_history_fields.py "Accumulators"): RAINC = CUPREC_TOT (convective), RAINNC = ACPREC_TOT -
// CUPREC_TOT (grid-scale), both converted m -> mm. Not the NPREC bucket CUPREC.
extern "C" __global__ void hist_surface(
    int IM, int JM, int LM,
    const float* __restrict__ PINT, const float* __restrict__ FIS,
    const float* __restrict__ ACPREC_TOT, const float* __restrict__ CUPREC_TOT,
    float* __restrict__ PSFC, float* __restrict__ HGT,
    float* __restrict__ RAINNC, float* __restrict__ RAINC)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int j0 = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || j0 >= JM) return;
    const size_t n2 = IDX2(i0, j0, IM);

    PSFC[n2] = PINT[IDX3(i0, j0, LM, IM, JM)];
    HGT[n2] = FIS[n2] / NMMB_G;
    const float cu = CUPREC_TOT[n2] * 1000.0f;
    RAINC[n2] = cu;
    RAINNC[n2] = ACPREC_TOT[n2] * 1000.0f - cu;
}

// Radar reflectivity with the convective rain included: history REFL_10CM and its column maximum COMPOSITE.
//
// The microphysics reflectivity (dBZ) leaves out the precipitation of the convection scheme, which falls in the
// same columns. That rain is turned into an equivalent reflectivity factor with the WSR-88D default relation
//     Z = 300 R^1.4        (Z in mm6 m-3, R in mm h-1; Fulton et al. 1998, Wea. Forecasting 13, 377-395,
//                           section 3d(1), p. 382),
// the same constants as the model's own CUREFL (nmm/module_DIAGNOSE.F90: CU_A = 300, CU_B = 1.4), and added to the
// microphysics value in linear units, as the model's MAX_FIELDS_HR and CALMICT add CUREFL:
//     Z_total = 10^(dBZ_mp / 10) [0 where dBZ_mp is DBZmin] + Z_conv,   dBZ = 10 log10(Z_total) if Z_total > Zmin,
//     else DBZmin (DBZmin = -20, module_CONSTANTS; Zmin = 0.01, module_DIAGNOSE).
// R is the scheme's last precipitation over its own call interval: CPRATE (m per call) / (NCNVC DT) in mm h-1.
// Vertical extent: from the lowest layer up to the highest convective cloud top layer of the output interval
// (CNVTOP, layers counted from the bottom; 0 = no convective cloud, then the whole column takes the term).
// Where T < 0 C the precipitation is ice, and the term falls off with height h above the 0 C level z0:
//     Z_conv(h) = Z_conv * 10^(-(D0 + S h) / 10),   D0 = 3.5 dB, S = 4.0 dB km-1   (layers with T < 0 C)
// Reflectivity factors are radar-equivalent (water dielectric factor |K_w|^2 = 0.93). For dry ice particles of the
// same mass, sized by melted diameter, Z_e = (|K_i|^2 / |K_w|^2) (rho_w / rho_i)^2 Z = 0.224 Z (-6.5 dB; Smith 1984,
// J. Climate Appl. Meteor. 23, 1258-1260); the profile reaches that value 0.75 km above z0, where mixed-phase
// particles have frozen. Above it the peak reflectivity of convective cores decreases by a few dB per km with
// height (Heymsfield et al. 2010, J. Atmos. Sci. 67, 285-308, section 3b and Fig. 11: mean peak profiles fall from
// near their low-level value at the -10 C level, about 6 km, by roughly 2.5 to 3 dB km-1 up to 10 km, over land,
// ocean and in tropical cyclones). D0 and S are fitted to NCEP's NAM REFD records: the difference REFD(4000 m AGL) -
// REFD(hybrid level 1) against the height of 4000 m AGL above the 0 C isotherm height (both NCEP records), in
// columns where the hour's convective precipitation is at least 1 mm and 90% of the total, ten 00-24 h cycles of
// 2024-2025; least absolute error over 0.75 to 3.5 km above z0, with the term absent (drop of 25 dB or more)
// left out. z0 is the lowest 0 C crossing of the column, linear in height between layer midpoints; the lowest
// layer midpoint if that layer is already below 0 C.
// Columns with no convective rain (CPRATE <= 0) are copied unchanged.
//
// One thread per column. REFL, T, Z (layer midpoint height, m), OUT3 are [LM, JM, IM] (l0 = 0 at the top);
// CPRATE, CNVTOP, COMP are [JM, IM].
// WRITE3 = 0 writes COMP only (the 3-D field is then computed later into a shared buffer); OUT3 may alias REFL.
#define HR_CONV_A      300.0f     // Z = A R^B (Fulton et al. 1998; module_DIAGNOSE CU_A)
#define HR_CONV_B      1.4f       // (module_DIAGNOSE CU_B)
#define HR_ICE_D0      3.5f       // dB below the liquid value at the 0 C level (fit, see above)
#define HR_ICE_SLOPE   4.0e-3f    // dB per m above the 0 C level (fit, see above)
#define HR_DBZMIN      (-20.0f)   // module_CONSTANTS dbzmin
#define HR_ZMIN        0.01f      // module_DIAGNOSE Zmin

extern "C" __global__ void hist_refl_conv(
    int IM, int JM, int LM,
    const float* __restrict__ REFL, const float* __restrict__ T, const float* __restrict__ Z,
    const float* __restrict__ CPRATE, const float* __restrict__ CNVTOP,
    float rate_mmh, int WRITE3,
    float* OUT3, float* __restrict__ COMP)
{
    COLUMN_IJ(IM, JM)
    const size_t n2 = IDX2(i0, j0, IM);
    const float cp = CPRATE[n2];
    float zconv = 0.0f;
    if (cp > 0.0f) zconv = HR_CONV_A * powf(cp * rate_mmh, HR_CONV_B);
    int ktop = (int)rintf(CNVTOP[n2]);
    if (ktop < 1 || ktop > LM) ktop = LM;
    // 0 C level: the lowest crossing going up from the bottom layer (l0 = LM - 1)
    float z0 = 3.0e38f;
    if (zconv > 0.0f) {
        const size_t nb = IDX3(i0, j0, LM - 1, IM, JM);
        if (T[nb] < NMMB_TIW) {
            z0 = Z[nb];
        } else {
            for (int l0 = LM - 2; l0 >= 0; --l0) {
                const size_t nu = IDX3(i0, j0, l0, IM, JM), nl = IDX3(i0, j0, l0 + 1, IM, JM);
                if (T[nu] < NMMB_TIW) {
                    const float w = (T[nl] - NMMB_TIW) / (T[nl] - T[nu]);
                    z0 = Z[nl] + w * (Z[nu] - Z[nl]);
                    break;
                }
            }
        }
    }
    float cmax = HR_DBZMIN;
    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t n = IDX3(i0, j0, l0, IM, JM);
        float dbz = REFL[n];
        if (zconv > 0.0f && (LM - l0) <= ktop) {
            const float zc = (T[n] < NMMB_TIW)
                ? zconv * exp10f(-0.1f * (HR_ICE_D0 + HR_ICE_SLOPE * fmaxf(Z[n] - z0, 0.0f)))
                : zconv;
            const float z = ((dbz > HR_DBZMIN) ? exp10f(0.1f * dbz) : 0.0f) + zc;
            dbz = (z > HR_ZMIN) ? 10.0f * log10f(z) : HR_DBZMIN;
        }
        if (WRITE3) OUT3[n] = dbz;
        cmax = fmaxf(cmax, dbz);
    }
    COMP[n2] = cmax;
}
