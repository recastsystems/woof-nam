// phys_tempo_driver.cu -- WOOF-NAM unit phys-tempo-driver: GSMDRIVE glue (the NMMB 'thompson' route,
// nmm/module_MICROPHYSICS.F90:195-558, nam.v4.2.11 83df4e13c1) with the TEMPO v3.0.3 column (NCAR/TEMPO
// 9adf9ef, Apache-2.0, run through gpuwm-hex's hexcore.mpasphys.rrfs.tempo_v303 column kernel) in place of
// mp_gt_driver.
//
// NOTICE: GSMDRIVE glue is NOAA/NCEP NMMB source (US government work). TEMPO: Copyright NOAA, UCAR/NCAR,
// CU/CIRES, Apache License 2.0.
//
// Layout: NMMB arrays are [LM, JM, IM], L = 0 at the model top. The TEMPO column blob is (17, LM, NC), level
// k = 0 at the ground (MPAS order), carriers t p w dz qv qc qr qi qs qg ni nr nc nwfa nifa ng qb, column-fastest.
// A chunk is the rows [j0, j0 + nrows) of the domain: column c = (j - j0) * IM + i.
//
// Kernels (one launch each, no host sync, no device malloc):
//   tempo_pack      one thread per (i, j, k) of the chunk: the GSMDRIVE column fill (lines 222-263: DPL,
//                   PHMID, RR, PI, DZ = DPL R_G / RR) and USE_QV (lines 275-298: QV = Q / (1 - Q)), the species
//                   from their TRACERS slots; absent carriers (nc nwfa nifa ng qb, presence 0) are zero.
//   tempo_unpack    one thread per (i, j, k), B1 interior only (its_b1..ite_b1, jts_b1..jte_b1, lines 488-531):
//                   Q = QV / (1 + QV), the species back, CWM = QC + QR + QI + QS + QG, TNEW, TRAIN, T (USE_RADAR
//                   branch kept), REFL_10CM from TEMPO's reflectivity_10cm profile.
//   tempo_surface   one thread per (i, j), B1 interior: RAINNCV = rain + ice + snow + graupel (mm per call, as
//                   mp_gt_driver's RAINNCV, "all precip"), PREC / ACPREC / ACPREC_TOT += RAINNCV 1e-3 (lines
//                   543-546), SR = TEMPO frozen fraction, the thompson accumulators acpcp_sn / gr / ra (lines
//                   440-452), AVRAIN += 1 (lines 204-206, every column of the tile).
#include "nmmb_common.cuh"

#define PTD_R_G (1.0f / NMMB_G)
#define PTD_NF 17

extern "C" __global__ void tempo_pack(
    const int IM, const int JM, const int LM, const int j0, const int nrows,
    const float* __restrict__ pd, const float* __restrict__ t, const float* __restrict__ q,
    const float* __restrict__ w,
    const float* __restrict__ dsg2, const float* __restrict__ sgml2,
    const float* __restrict__ pdsg1, const float* __restrict__ psgml1,
    const float* __restrict__ qc, const float* __restrict__ qr, const float* __restrict__ qi,
    const float* __restrict__ qs, const float* __restrict__ qg, const float* __restrict__ ni,
    const float* __restrict__ nr,
    float* __restrict__ f)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (i0 >= IM || jj >= nrows || l0 >= LM) return;
    const int jg = j0 + jj;
    const size_t nc = (size_t)nrows * IM;
    const size_t c = (size_t)jj * IM + i0;
    const int k = LM - 1 - l0;                       // TEMPO level (0 = ground)
    const size_t h = IDX2(i0, jg, IM);
    const size_t p = IDX3(i0, jg, l0, IM, JM);
    const float pdsl = pd[h];
    const float dpl = dsg2[l0] * pdsl + pdsg1[l0];
    const float phmid = sgml2[l0] * pdsl + psgml1[l0];
    const float tl = t[p];
    const float ql = fmaxf(q[p], NMMB_EPSQ);
    const float rr = phmid / (NMMB_R_D * tl * (NMMB_P608 * ql + 1.0f));
    const float qq = q[p];
#define PTD_F(n) f[((size_t)(n) * LM + k) * nc + c]
    PTD_F(0) = tl;
    PTD_F(1) = phmid;
    PTD_F(2) = w[p];
    PTD_F(3) = dpl * PTD_R_G / rr;
    PTD_F(4) = qq / (1.0f - qq);
    PTD_F(5) = qc[p];
    PTD_F(6) = qr[p];
    PTD_F(7) = qi[p];
    PTD_F(8) = qs[p];
    PTD_F(9) = qg[p];
    PTD_F(10) = ni[p];
    PTD_F(11) = nr[p];
    PTD_F(12) = 0.0f; PTD_F(13) = 0.0f; PTD_F(14) = 0.0f; PTD_F(15) = 0.0f; PTD_F(16) = 0.0f;
#undef PTD_F
}

extern "C" __global__ void tempo_unpack(
    const int IM, const int JM, const int LM, const int j0, const int nrows,
    const float* __restrict__ f, const float* __restrict__ diag,
    const float* __restrict__ dfi_tten, const float rdtphs, const int use_radar,
    float* __restrict__ t, float* __restrict__ q, float* __restrict__ cwm, float* __restrict__ train,
    float* __restrict__ qc, float* __restrict__ qr, float* __restrict__ qi,
    float* __restrict__ qs, float* __restrict__ qg, float* __restrict__ ni, float* __restrict__ nr,
    float* __restrict__ refl, float* __restrict__ re_c, float* __restrict__ re_i, float* __restrict__ re_s)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (i0 >= IM || jj >= nrows || l0 >= LM) return;
    const int jg = j0 + jj;
    if (i0 < 1 || i0 > IM - 2 || jg < 1 || jg > JM - 2) {
        // Boundary ring (outside the GSMDRIVE B1 tile): the species are reset to zero every microphysics call.
        // Breakage it prevents: with spec_adv on, condensate advected into the ring had no sink (microphysics
        // skips it; UPDATE_WATER folds it back into CWM there), piled up to 19 g/kg on the outflow edge and rained
        // 19 mm in 2 h out of the first interior column (12 km smoke, 2026-11-08). Zero is the boundary value of a
        // GFS start (NPS do_clouds false: boundary CWM = 0); NMMB has no species boundary arrays.
        const size_t pb = IDX3(i0, jg, l0, IM, JM);
        qc[pb] = 0.0f; qi[pb] = 0.0f; qr[pb] = 0.0f; qs[pb] = 0.0f; qg[pb] = 0.0f; ni[pb] = 0.0f; nr[pb] = 0.0f;
        return;
    }
    const size_t nc = (size_t)nrows * IM;
    const size_t c = (size_t)jj * IM + i0;
    const int k = LM - 1 - l0;
    const size_t p = IDX3(i0, jg, l0, IM, JM);
#define PTD_F(n) f[((size_t)(n) * LM + k) * nc + c]
    const float qv = PTD_F(4);
    const float c1 = PTD_F(5), r1 = PTD_F(6), i1 = PTD_F(7), s1 = PTD_F(8), g1 = PTD_F(9);
    q[p] = qv / (1.0f + qv);
    qc[p] = c1; qr[p] = r1; qi[p] = i1; qs[p] = s1; qg[p] = g1;
    ni[p] = PTD_F(10); nr[p] = PTD_F(11);
    cwm[p] = c1 + r1 + i1 + s1 + g1;
    const float tnew = PTD_F(0);
#undef PTD_F
    train[p] = train[p] + (tnew - t[p]) * rdtphs;
    if (use_radar) {
        float mp_tten = (tnew - t[p]) * rdtphs;
        const float dfi = dfi_tten[p];
        if (dfi > mp_tten && dfi < 0.01f && mp_tten < 0.0018f) mp_tten = dfi;
        t[p] = t[p] + mp_tten / rdtphs;
    } else {
        t[p] = tnew;
    }
    refl[p] = diag[((size_t)8 * LM + k) * nc + c];   // reflectivity_10cm (dBZ), TEMPO diag field 8
    re_c[p] = diag[((size_t)9 * LM + k) * nc + c];   // effective radii (m), TEMPO fields 9-11, for progcld8
    re_i[p] = diag[((size_t)10 * LM + k) * nc + c];
    re_s[p] = diag[((size_t)11 * LM + k) * nc + c];
}

extern "C" __global__ void tempo_surface(
    const int IM, const int JM, const int LM, const int j0, const int nrows,
    const float* __restrict__ diag,
    float* __restrict__ avrain, float* __restrict__ rainncv, float* __restrict__ sr,
    float* __restrict__ prec, float* __restrict__ acprec, float* __restrict__ acprec_tot,
    float* __restrict__ acpcp_sn, float* __restrict__ acpcp_gr, float* __restrict__ acpcp_ra)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || jj >= nrows) return;
    const int jg = j0 + jj;
    const size_t nc = (size_t)nrows * IM;
    const size_t c = (size_t)jj * IM + i0;
    const size_t h = IDX2(i0, jg, IM);
    avrain[h] = avrain[h] + 1.0f;
    if (i0 < 1 || i0 > IM - 2 || jg < 1 || jg > JM - 2) return;
    // diag field f, column scalar at level 0: diag[(f * LM + 0) * nc + c]
    const float r = diag[(size_t)0 * LM * nc + c], ice = diag[(size_t)1 * LM * nc + c];
    const float sn = diag[(size_t)2 * LM * nc + c], gr = diag[(size_t)3 * LM * nc + c];
    const float tot = ((r + ice) + sn) + gr;
    const float snow = ice + sn;
    rainncv[h] = tot;
    sr[h] = diag[(size_t)5 * LM * nc + c];
    const float pcpcol = tot * 1.0e-3f;
    prec[h] = prec[h] + pcpcol;
    acprec[h] = acprec[h] + pcpcol;
    acprec_tot[h] = acprec_tot[h] + pcpcol;
    acpcp_sn[h] = acpcp_sn[h] + snow;
    acpcp_gr[h] = acpcp_gr[h] + gr;
    acpcp_ra[h] = acpcp_ra[h] + fmaxf(0.0f, tot - snow - gr);
}

// ---------------------------------------------------------------------------------------------------------------
// Warp path (NAMGPU_TEMPO_KERNEL=warp, the default): hexcore's tempo_warp_steps reads MPAS carriers through a
// pointer table, theta in slot 0 and the Exner function in slot 17 (it forms T = theta * exner itself), and
// writes the surface precipitation into s[] and the reflectivity / radii profiles into profiles[]. Same column
// physics as tempo_columns, one warp per column (lanes over levels), resident warps loop over the columns.
//   tempo_pack_w     as tempo_pack, slot 0 = TL / PI with PI = (PHMID 1e-5)^CAPPA (GSMDRIVE PI_PHY, TH_PHY)
//   tempo_unpack_w   as tempo_unpack, TNEW = theta * PI; reflectivity and radii from profiles
//   tempo_surface_w  as tempo_surface, from s[1] (all precip), s[3] (ice + snow), s[5] (graupel), s[6] (SR)
extern "C" __global__ void tempo_pack_w(
    const int IM, const int JM, const int LM, const int j0, const int nrows,
    const float* __restrict__ pd, const float* __restrict__ t, const float* __restrict__ q,
    const float* __restrict__ w,
    const float* __restrict__ dsg2, const float* __restrict__ sgml2,
    const float* __restrict__ pdsg1, const float* __restrict__ psgml1,
    const float* __restrict__ qc, const float* __restrict__ qr, const float* __restrict__ qi,
    const float* __restrict__ qs, const float* __restrict__ qg, const float* __restrict__ ni,
    const float* __restrict__ nr,
    float* __restrict__ f, float* __restrict__ pii)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (i0 >= IM || jj >= nrows || l0 >= LM) return;
    const int jg = j0 + jj;
    const size_t nc = (size_t)nrows * IM;
    const size_t c = (size_t)jj * IM + i0;
    const int k = LM - 1 - l0;
    const size_t h = IDX2(i0, jg, IM);
    const size_t p = IDX3(i0, jg, l0, IM, JM);
    const float pdsl = pd[h];
    const float dpl = dsg2[l0] * pdsl + pdsg1[l0];
    const float phmid = sgml2[l0] * pdsl + psgml1[l0];
    const float tl = t[p];
    const float ql = fmaxf(q[p], NMMB_EPSQ);
    const float rr = phmid / (NMMB_R_D * tl * (NMMB_P608 * ql + 1.0f));
    const float pi = powf(phmid * 1.0e-5f, NMMB_CAPPA);
    const float qq = q[p];
#define PTD_F(n) f[((size_t)(n) * LM + k) * nc + c]
    PTD_F(0) = tl / pi;
    PTD_F(1) = phmid;
    PTD_F(2) = w[p];
    PTD_F(3) = dpl * PTD_R_G / rr;
    PTD_F(4) = qq / (1.0f - qq);
    PTD_F(5) = qc[p]; PTD_F(6) = qr[p]; PTD_F(7) = qi[p]; PTD_F(8) = qs[p]; PTD_F(9) = qg[p];
    PTD_F(10) = ni[p]; PTD_F(11) = nr[p];
#undef PTD_F
    pii[(size_t)k * nc + c] = pi;
}

extern "C" __global__ void tempo_unpack_w(
    const int IM, const int JM, const int LM, const int j0, const int nrows,
    const float* __restrict__ f, const float* __restrict__ pii, const float* __restrict__ prof,
    const float* __restrict__ dfi_tten, const float rdtphs, const int use_radar,
    float* __restrict__ t, float* __restrict__ q, float* __restrict__ cwm, float* __restrict__ train,
    float* __restrict__ qc, float* __restrict__ qr, float* __restrict__ qi,
    float* __restrict__ qs, float* __restrict__ qg, float* __restrict__ ni, float* __restrict__ nr,
    float* __restrict__ refl, float* __restrict__ re_c, float* __restrict__ re_i, float* __restrict__ re_s)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    const int l0 = blockIdx.z;
    if (i0 >= IM || jj >= nrows || l0 >= LM) return;
    const int jg = j0 + jj;
    if (i0 < 1 || i0 > IM - 2 || jg < 1 || jg > JM - 2) {     // boundary ring: see tempo_unpack
        const size_t pb = IDX3(i0, jg, l0, IM, JM);
        qc[pb] = 0.0f; qi[pb] = 0.0f; qr[pb] = 0.0f; qs[pb] = 0.0f; qg[pb] = 0.0f; ni[pb] = 0.0f; nr[pb] = 0.0f;
        return;
    }
    const size_t nc = (size_t)nrows * IM;
    const size_t c = (size_t)jj * IM + i0;
    const int k = LM - 1 - l0;
    const size_t p = IDX3(i0, jg, l0, IM, JM);
    const size_t x = (size_t)k * nc + c;
#define PTD_F(n) f[((size_t)(n) * LM + k) * nc + c]
    const float qv = PTD_F(4);
    const float c1 = PTD_F(5), r1 = PTD_F(6), i1 = PTD_F(7), s1 = PTD_F(8), g1 = PTD_F(9);
    q[p] = qv / (1.0f + qv);
    qc[p] = c1; qr[p] = r1; qi[p] = i1; qs[p] = s1; qg[p] = g1;
    ni[p] = PTD_F(10); nr[p] = PTD_F(11);
    cwm[p] = c1 + r1 + i1 + s1 + g1;
    const float tnew = PTD_F(0) * pii[x];
#undef PTD_F
    train[p] = train[p] + (tnew - t[p]) * rdtphs;
    if (use_radar) {
        float mp_tten = (tnew - t[p]) * rdtphs;
        const float dfi = dfi_tten[p];
        if (dfi > mp_tten && dfi < 0.01f && mp_tten < 0.0018f) mp_tten = dfi;
        t[p] = t[p] + mp_tten / rdtphs;
    } else {
        t[p] = tnew;
    }
    const size_t pl = (size_t)LM * nc;
    refl[p] = prof[x];
    re_c[p] = prof[pl + x]; re_i[p] = prof[2 * pl + x]; re_s[p] = prof[3 * pl + x];
}

extern "C" __global__ void tempo_surface_w(
    const int IM, const int JM, const int j0, const int nrows,
    const float* __restrict__ sfc,
    float* __restrict__ avrain, float* __restrict__ rainncv, float* __restrict__ sr,
    float* __restrict__ prec, float* __restrict__ acprec, float* __restrict__ acprec_tot,
    float* __restrict__ acpcp_sn, float* __restrict__ acpcp_gr, float* __restrict__ acpcp_ra)
{
    const int i0 = blockIdx.x * blockDim.x + threadIdx.x;
    const int jj = blockIdx.y * blockDim.y + threadIdx.y;
    if (i0 >= IM || jj >= nrows) return;
    const int jg = j0 + jj;
    const size_t nc = (size_t)nrows * IM;
    const size_t c = (size_t)jj * IM + i0;
    const size_t h = IDX2(i0, jg, IM);
    avrain[h] = avrain[h] + 1.0f;
    if (i0 < 1 || i0 > IM - 2 || jg < 1 || jg > JM - 2) return;
    const float tot = sfc[1 * nc + c], snow = sfc[3 * nc + c], gr = sfc[5 * nc + c];
    rainncv[h] = tot;
    sr[h] = sfc[6 * nc + c];
    const float pcpcol = tot * 1.0e-3f;
    prec[h] = prec[h] + pcpcol;
    acprec[h] = acprec[h] + pcpcol;
    acprec_tot[h] = acprec_tot[h] + pcpcol;
    acpcp_sn[h] = acpcp_sn[h] + snow;
    acpcp_gr[h] = acpcp_gr[h] + gr;
    acpcp_ra[h] = acpcp_ra[h] + fmaxf(0.0f, tot - snow - gr);
}
