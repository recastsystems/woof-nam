// nam-gpu unit phys-ntiedtke-cu: the NMMB <-> WRF column seam for New Tiedtke (cu_physics 16).
//
// The scheme itself is ArWen's GPU New Tiedtke, copied unmodified into
// namgpu/third_party/arwen/core/{ntiedtke.py, kernels/ntiedtke_assembled.cu} (Apache-2.0, WRF
// module_cu_ntiedtke.F lineage, public domain). This file only maps the NMMB B-grid column (L = 1 at
// the model top, specific humidity, total condensate CW + F_ICE/F_RAIN) onto the WRF column the scheme
// reads (k = 0 at the ground, mixing ratio, qc/qi) and maps the tendencies back into the arrays the
// CUCNVC post kernel (phys_bmj_driver_post) applies, exactly as it applies BMJ's.
//
// The domain (or band) is walked in chunks of the scheme's tile width W: chunk columns col0 .. col0+n-1
// (col = j0*IM + i0). One thread per workspace slot t in [0, W) (pack) or [0, n) (unpack). WRF arrays are
// the pipeline's own workspace buffers, level-major [k, t] with stride W (k = 0 lowest); interface arrays
// have LM+1 rows. A short last chunk pads slots n..W-1 with the chunk's first column, as ArWen's adapter
// does (the one cross-column operation, llo3's OR, cannot move under a repeated member).
// NMMB arrays are [l0, j0, i0] (IDX3), l0 = 0 at the top: k = LM-1-l0.
//
// Kernels
//   nt_pack     NMMB state -> WRF scheme inputs (t3d qv3d qc3d qi3d u3d v3d pcps p8w dz8w rho3d exner
//               w qvften thften xland hfx qfx dx)
//   nt_unpack   scheme outputs -> RTHCUTEN, RQCUTEN, DUDT_PHY, DVDT_PHY, RAINCV, CUTOP, CUBOT, and the
//               detrained condensate into CW/F_ICE/F_RAIN (and QC/QR/QS kept consistent with them).
//               shallow_off != 0 drops ktype 2 (shallow) columns: the EDMF pairing rule.
//   nt_snap     T and Q right after the microphysics (GSMDRIVE), the reference for the next call's
//               large-scale forcing (see the wrapper docstring).
//
// Numerics: float32, FMA contraction on (namgpu kernels/__init__.py OPTIONS). No host sync.

#include "nmmb_common.cuh"

// Species from the NMMB total condensate, the way FER_HIRES splits it (CW is the only advected
// condensate with spec_adv false, so QC/QR/QS can be stale between microphysics calls; CW, F_ICE and
// F_RAIN are not).
__device__ __forceinline__ void nt_split(float cw, float fice, float frain,
                                         float* qcw, float* qr, float* qi)
{
    const float c = fmaxf(cw, 0.0f);
    const float ice = c * fminf(fmaxf(fice, 0.0f), 1.0f);
    const float liq = c - ice;
    const float rain = liq * fminf(fmaxf(frain, 0.0f), 1.0f);
    *qi = ice;
    *qr = rain;
    *qcw = liq - rain;
}

extern "C" __global__ void nt_pack(
    const int IM, const int JM, const int LM, const int col0, const int n, const int W,
    const int forcing_ok, const float rdtcnvc,
    const float* __restrict__ T,        // [LM,JM,IM]
    const float* __restrict__ Q,
    const float* __restrict__ CW,
    const float* __restrict__ F_ICE,
    const float* __restrict__ F_RAIN,
    const float* __restrict__ U_PHY,
    const float* __restrict__ V_PHY,
    const float* __restrict__ PHMID,
    const float* __restrict__ PHINT,    // [LM+1,JM,IM]
    const float* __restrict__ DZ,
    const float* __restrict__ RR,
    const float* __restrict__ EXNER,
    const float* __restrict__ W_TOT,    // [>=LM,JM,IM], first LM levels are mid layers
    const float* __restrict__ TSNAP,    // [LM,JM,IM] T after the last microphysics call
    const float* __restrict__ QSNAP,    // [LM,JM,IM] Q after the last microphysics call
    const float* __restrict__ XLAND,    // [JM,IM]  1 land, 2 water (prep: SM+1)
    const float* __restrict__ TWBS,     // [JM,IM]  NMMB sensible flux after TURBL (W m-2, + down)
    const float* __restrict__ QWBS,     // [JM,IM]  NMMB latent flux after TURBL (W m-2, + down)
    const float* __restrict__ DXH,      // [JM]
    const float dyh,
    float* __restrict__ t3d, float* __restrict__ qv3d, float* __restrict__ qc3d,
    float* __restrict__ qi3d, float* __restrict__ u3d, float* __restrict__ v3d,
    float* __restrict__ pcps, float* __restrict__ p8w, float* __restrict__ dz8w,
    float* __restrict__ rho3d, float* __restrict__ exner, float* __restrict__ w,
    float* __restrict__ qvften, float* __restrict__ thften,
    float* __restrict__ xland, float* __restrict__ hfx, float* __restrict__ qfx,
    float* __restrict__ dx)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= W) return;
    const int col = col0 + (t < n ? t : 0);
    const int i0 = col % IM;
    const int j0 = col / IM;
    const size_t p2 = IDX2(i0, j0, IM);

    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t pk = IDX3(i0, j0, l0, IM, JM);
        const size_t wk = (size_t)(LM - 1 - l0) * W + t;
        float qcw, qr, qi;
        nt_split(CW[pk], F_ICE[pk], F_RAIN[pk], &qcw, &qr, &qi);
        const float q = fminf(fmaxf(Q[pk], NMMB_EPSQ), 0.5f);
        const float qv = q / (1.0f - q);
        t3d[wk] = T[pk];
        qv3d[wk] = qv;
        qc3d[wk] = qcw;
        qi3d[wk] = qi;
        u3d[wk] = U_PHY[pk];
        v3d[wk] = V_PHY[pk];
        pcps[wk] = PHMID[pk];
        dz8w[wk] = DZ[pk];
        rho3d[wk] = RR[pk];
        exner[wk] = EXNER[pk];
        if (forcing_ok) {
            // Temperature forcing (the WRF fold hands New Tiedtke (RTHFTEN+RTHRATEN+RTHBLTEN)*pi, a T
            // tendency) and the vapour-mixing-ratio forcing, both over the last NCNVC steps with the
            // convective and microphysical increments taken out (TSNAP/QSNAP are after GSMDRIVE).
            const float qs = fminf(fmaxf(QSNAP[pk], NMMB_EPSQ), 0.5f);
            thften[wk] = (T[pk] - TSNAP[pk]) * rdtcnvc;
            qvften[wk] = (qv - qs / (1.0f - qs)) * rdtcnvc;
        } else {
            thften[wk] = 0.0f;
            qvften[wk] = 0.0f;
        }
    }
    // Interfaces, WRF kf = 0 at the ground: p8w[kf] = PHINT(l0 = LM - kf). w: 0 at the ground and the
    // top, the mean of the two adjacent mid-layer W_TOT in between (the scheme forms its omega from
    // 0.5*(w[k]+w[k+1]), so the mid-layer value comes back up to the averaging).
    for (int kf = 0; kf <= LM; ++kf) {
        const size_t wf = (size_t)kf * W + t;
        p8w[wf] = PHINT[IDX3(i0, j0, LM - kf, IM, JM)];
        float wv = 0.0f;
        if (kf > 0 && kf < LM) {
            const int l_below = LM - kf;       // WRF layer kf-1
            const int l_above = LM - 1 - kf;   // WRF layer kf
            wv = 0.5f * (W_TOT[IDX3(i0, j0, l_below, IM, JM)] + W_TOT[IDX3(i0, j0, l_above, IM, JM)]);
        }
        w[wf] = wv;
    }
    xland[t] = XLAND[p2];
    // WRF HFX (W m-2) and QFX (kg m-2 s-1) are positive upward; TURBL leaves TWBS and QWBS positive
    // downward in W m-2 (phys_turbl_driver.cu:615-616, xlv = NMMB_ELWV).
    hfx[t] = -TWBS[p2];
    qfx[t] = -QWBS[p2] / NMMB_ELWV;
    dx[t] = sqrtf(DXH[j0] * dyh);
}

extern "C" __global__ void nt_unpack(
    const int IM, const int JM, const int LM, const int col0, const int n, const int W,
    const float dtcnvc, const int shallow_off,
    const float* __restrict__ rthcuten, const float* __restrict__ rqvcuten,
    const float* __restrict__ rqccuten, const float* __restrict__ rqicuten,
    const float* __restrict__ rucuten, const float* __restrict__ rvcuten,
    const float* __restrict__ raincv_w,          // [W] mm per model step (rn/stepcu)
    const int* __restrict__ ktype, const int* __restrict__ ldcum,
    const int* __restrict__ kctop, const int* __restrict__ kcbot,
    const float* __restrict__ Q,
    float* __restrict__ RTHCUTEN, float* __restrict__ RQCUTEN,
    float* __restrict__ DUDT_PHY, float* __restrict__ DVDT_PHY,
    float* __restrict__ RAINCV, float* __restrict__ CUTOP, float* __restrict__ CUBOT,
    float* __restrict__ CW, float* __restrict__ F_ICE, float* __restrict__ F_RAIN,
    float* __restrict__ QC, float* __restrict__ QR, float* __restrict__ QS,
    int* __restrict__ ACTIVE,                   // [JM,IM] 0 none, 1 deep, 2 shallow, 3 mid, -2 shallow dropped
    const int write_species)                    // 1: FER_HIRES (QC/QR/QS are Ferrier diagnostics, kept consistent);
                                                // 0: thompson layout (QC/QR/QS ARE the TRACERS species; the CW change
                                                //    is handed to QC/QI by phys-bmj-driver instead)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int col = col0 + t;
    const int i0 = col % IM;
    const int j0 = col / IM;
    const size_t p2 = IDX2(i0, j0, IM);
    const bool interior = (i0 >= 1 && i0 <= IM - 2 && j0 >= 1 && j0 <= JM - 2);   // its_b1..ite_b1

    const int kt = ktype[t];
    const bool cum = ldcum[t] != 0;
    const bool drop = shallow_off && kt == 2;
    const bool apply = interior && cum && !drop;
    ACTIVE[p2] = !cum ? 0 : (drop ? -2 : kt);

    // The prep kernel already zeroed RTHCUTEN/RQCUTEN/DUDT_PHY/DVDT_PHY/RAINCV and set CUTOP/CUBOT
    // to 999; a column that does not convect (or is dropped) keeps exactly that.
    if (!apply) return;

    RAINCV[p2] = raincv_w[t];
    CUTOP[p2] = (float)(LM + 1 - kctop[t]);   // kctop/kcbot are 1-based from the top (IFS order)
    CUBOT[p2] = (float)(LM + 1 - kcbot[t]);

    for (int l0 = 0; l0 < LM; ++l0) {
        const size_t pk = IDX3(i0, j0, l0, IM, JM);
        const size_t wk = (size_t)(LM - 1 - l0) * W + t;
        RTHCUTEN[pk] = rthcuten[wk];
        // d(specific humidity)/dt from d(mixing ratio)/dt: dq = dr/(1+r)^2.
        const float q = fminf(fmaxf(Q[pk], NMMB_EPSQ), 0.5f);
        const float r1 = 1.0f + q / (1.0f - q);
        RQCUTEN[pk] = rqvcuten[wk] / (r1 * r1);
        DUDT_PHY[pk] = rucuten[wk];
        DVDT_PHY[pk] = rvcuten[wk];

        // Detrained condensate, the NMMB SAS template (module_CONVECTION.F90:566-593): species
        // updated, then CWM, F_ICE, F_RAIN recoupled. Done here on the species split from CW.
        const float dqc = rqccuten[wk] * dtcnvc;
        const float dqi = rqicuten[wk] * dtcnvc;
        if (dqc != 0.0f || dqi != 0.0f) {
            float qcw, qr, qi;
            nt_split(CW[pk], F_ICE[pk], F_RAIN[pk], &qcw, &qr, &qi);
            qcw = fmaxf(qcw + dqc, 0.0f);
            qi = fmaxf(qi + dqi, 0.0f);
            const float cwm = qcw + qr + qi;
            CW[pk] = cwm;
            F_ICE[pk] = (cwm > NMMB_EPSQ) ? qi / cwm : 0.0f;
            F_RAIN[pk] = (qr > NMMB_EPSQ) ? qr / (qcw + qr) : 0.0f;
            // Breakage it prevents (2026-11-09, scale-aware-cu lane): under TEMPO these names are bound to the
            // TRACERS slots, so the Ferrier split of CW (which already holds QC+QR+QI+QS+QG) overwrote QC/QR and
            // put all ice in QS while QI/QG stayed: condensate created in every detraining column.
            if (write_species) {
                QC[pk] = qcw;
                QR[pk] = qr;
                QS[pk] = qi;
            }
        }
    }
}

extern "C" __global__ void nt_snap(
    const int n, const float* __restrict__ T, const float* __restrict__ Q,
    float* __restrict__ TSNAP, float* __restrict__ QSNAP)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    TSNAP[p] = T[p];
    QSNAP[p] = Q[p];
}
