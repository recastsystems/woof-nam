// phys_ugwp_oro.cu -- NOAA-GSL UGWP orographic drag suite (bl_ugwp_gwdo) for nam-gpu: large-scale
// gravity-wave drag (Kim and Doyle 2005), flow blocking (Lott and Miller 1997), small-scale GWD
// (Steeneveld et al. 2008, Tsiringakis et al. 2017) and turbulent orographic form drag (Beljaars et al.
// 2004), one thread per NMMB H column, reading the NAM TURBL working arrays directly (no packing copy).
//
// Copyright 2026, NOAA, UCAR/NCAR CU/CIRES (NOAA GSL Unified Gravity Wave Physics, UGWP).
// Licensed under the Apache License, Version 2.0 (LICENSE-UGWP-Apache-2.0.txt beside NOTICE).
// Source: https://github.com/NOAA-GSL/UGWP commit c1c893edcf171af5639af60e3a3a528816f6cc2b (tag MPAS_20241223),
// bl_ugwp.F:15-1059 (bl_ugwp_run); upstream LICENSE added at 4108d491470c. The column body is the gpuwm-hex
// CUDA port of that routine (gpuwm-hex src/hexcore/mpasphys/ugwp_oro_v842.py, Apache-2.0), with the
// bitwise FP32 sinf/powf/expf helpers replaced by the CUDA ones (nam-gpu runs fast numerics, no bitwise goal).
//
// NMMB mapping (this file's prologue, the only nam-gpu code here):
//   column level k (1-based, lowest layer first) = NMMB mass level fl0 = LM - k (0-based, top-down);
//   uproj/vproj = U_PHY/V_PHY (H-point, rotated-grid winds); the scheme rotates them to Earth with
//     u_e = u*cosa - v*sina, v_e = u*sina + v*cosa, so cosa = CROT, sina = -SROT (phys-gwd-init; the same
//     rotation module_GWD.F90 applies), and rotates its tendencies back the same way;
//   t1 = T, q1 = Q, prsl = PHMID (Pa), prslk = EXNER, prsi(k) = PHINT[LM+1-k] (Pa),
//   zl = 0.5 (Z[fl0] + Z[fl0+1]), dz = Z[fl0] - Z[fl0+1] (TURBL_Z interface heights, m MSL);
//   kpbl = LM + 1 - LPBL (MYJ LPBL is top-down), clamped to [1, LM];
//   hpbl = PBLH, br1 = RIB (JSFC bulk Richardson number), xland = XLAND (1 land, 2 water);
//   dx = DXM[j] (row spacing sqrt(dxh * dyh), m); the twenty orography statistics are UGWP_ORO
//   (twenty [JM][IM] fields UGWP_<NAME>) in the RRFS ugwp_oro_data order (var2dls, conls, oa1-4ls, ol1-4ls, var2dss, conss,
//   oa1-4ss, ol1-4ss), built on this grid by tools/ugwp-oro (rw_hex_ugwp_oro, Rust, every core).
// Out: DUDT_GWD, DVDT_GWD [LM][JM][IM] (rotated-grid tendencies, m s-2, joined to DUDT/DVDT by
//   turbl_dudt_add as NAM's GWD is), UGWP_DUSFC/UGWP_DVSFC (Earth-relative surface stress, N m-2).
// The per-scheme diagnostic stresses (dtaux3d_*) are not kept (diags = 0): they go to a register sink.

// Keep divisors in constant memory so every target retains division.
#ifndef HEX_DIVISOR_3F800000_DEFINED
#define HEX_DIVISOR_3F800000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_3f800000 = 1.0f;
#else
static const float hex_divisor_3f800000 = 1.0f;
#endif
#endif
#ifndef HEX_DIVISOR_44BB8000_DEFINED
#define HEX_DIVISOR_44BB8000_DEFINED
#if defined(__CUDACC__) || defined(__CUDACC_RTC__)
__device__ __constant__ float hex_divisor_44bb8000 = 1500.0f;
#else
static const float hex_divisor_44bb8000 = 1500.0f;
#endif
#endif

// MPAS physics_noaa/UGWP/bl_ugwp.F:15-1059, complete bl_ugwp_run.
#ifndef UGWP_LEVELS
#define UGWP_LEVELS 60
#endif

extern "C" __global__ void phys_ugwp_oro(
    float* __restrict__ DUDT_GWD, float* __restrict__ DVDT_GWD,
    float* __restrict__ DUSFC, float* __restrict__ DVSFC, int* __restrict__ status,
    const float* __restrict__ U, const float* __restrict__ V, const float* __restrict__ T,
    const float* __restrict__ Q, const float* __restrict__ PMID, const float* __restrict__ EXNR,
    const float* __restrict__ PINT, const float* __restrict__ Z, const int* __restrict__ LPBL,
    const float* __restrict__ PBLH, const float* __restrict__ RIB, const float* __restrict__ XLAND,
    const float* __restrict__ CROT, const float* __restrict__ SROT, const float* __restrict__ DXM,
    const float* __restrict__ O0, const float* __restrict__ O1, const float* __restrict__ O2,
    const float* __restrict__ O3, const float* __restrict__ O4, const float* __restrict__ O5,
    const float* __restrict__ O6, const float* __restrict__ O7, const float* __restrict__ O8,
    const float* __restrict__ O9, const float* __restrict__ O10, const float* __restrict__ O11,
    const float* __restrict__ O12, const float* __restrict__ O13, const float* __restrict__ O14,
    const float* __restrict__ O15, const float* __restrict__ O16, const float* __restrict__ O17,
    const float* __restrict__ O18, const float* __restrict__ O19,
    float g, float cp, float rd, float rv, float fv, float pi_c, float dt,
    int im, int jm, int nz, int jlo, int jhi)
{
    const int c = blockDim.x * blockIdx.x + threadIdx.x;
    const int n2 = im * jm;
    if (c >= n2) return;
    const int j0 = c / im;
    const int kte = nz, kpblmax = nz, kpblmin = 2;
    const int diags = 0;
    // Output column zeroed first: rows outside [jlo, jhi] (the B-grid edge rows NAM's GWD also skips by
    // its terrain/KPBL tests) and any column the KPBL guard stops keep zero tendencies.
    for (int k = 0; k < nz; ++k) { DUDT_GWD[(size_t)k * n2 + c] = 0.0f; DVDT_GWD[(size_t)k * n2 + c] = 0.0f; }
    DUSFC[c] = 0.0f; DVSFC[c] = 0.0f;
    if (j0 < jlo || j0 > jhi) return;
    float sink = 0.0f;
    const float* const oro[20] = {O0, O1, O2, O3, O4, O5, O6, O7, O8, O9, O10, O11, O12, O13, O14,
                                  O15, O16, O17, O18, O19};
    #define FL(k) ((size_t)(nz - (k)) * n2 + c)
    #define ZI(l) Z[(size_t)(l) * n2 + c]
    #define IN(m,k) ((m)==0 ? U[FL(k)] : (m)==1 ? V[FL(k)] : (m)==2 ? T[FL(k)] : (m)==3 ? Q[FL(k)] : \
                     (m)==4 ? PMID[FL(k)] : (m)==5 ? EXNR[FL(k)] : \
                     (m)==6 ? 0.5f * (ZI(nz - (k)) + ZI(nz - (k) + 1)) : \
                     (m)==7 ? (ZI(nz - (k)) - ZI(nz - (k) + 1)) : 0.0f)
    #define P(k) PINT[(size_t)(nz + 1 - (k)) * n2 + c]
    #define S(m) ((m)==0 ? -SROT[c] : (m)==1 ? CROT[c] : (m)==2 ? PBLH[c] : (m)==3 ? RIB[c] : \
                  (m)==4 ? XLAND[c] : (m)==5 ? DXM[j0] : oro[(m) - 6][c])
    #define O(m,k) (*((m)==0 ? &DUDT_GWD[FL(k)] : (m)==1 ? &DVDT_GWD[FL(k)] : &sink))
    #define SO(m) (*((m)==0 ? &DUSFC[c] : (m)==1 ? &DVSFC[c] : &sink))
    const float sa = S(0), ca = S(1), hpbl = S(2), br = S(3);
    const float xland = S(4), dx = S(5), var = S(6), oc1 = S(7), varss = S(16);
    const int kpbl = min(max(nz + 1 - LPBL[c], 1), nz);
    const float inv_g = 1.0f / g, fdir = 8.0f / (2.0f*pi_c);
    float u[UGWP_LEVELS+2], v[UGWP_LEVELS+2], vtj[UGWP_LEVELS+2];
    float vtk[UGWP_LEVELS+2], rho[UGWP_LEVELS+2], del[UGWP_LEVELS+2];
    float za[UGWP_LEVELS+2], bnv2[UGWP_LEVELS+2], ri[UGWP_LEVELS+2];
    float velco[UGWP_LEVELS+2], taup[UGWP_LEVELS+2], taufb[UGWP_LEVELS+2];
    float taud_ls[UGWP_LEVELS+2], taud_bl[UGWP_LEVELS+2];
    float dudt[UGWP_LEVELS+2], dvdt[UGWP_LEVELS+2];
    float dusfc = 0.0f, dvsfc = 0.0f;
    float dus[4] = {0.0f,0.0f,0.0f,0.0f}, dvs[4] = {0.0f,0.0f,0.0f,0.0f};
    // bl_ugwp.F:275-354: constants, taper, initialization.
    float ls_taper;
    if (dx >= 13000.0f) ls_taper = 1.0f;
    else if (dx <= 3000.0f) ls_taper = 0.0f;
    else ls_taper = 0.5f * (sinf(pi_c * (dx-0.5f*(13000.0f+3000.0f))
                                    / (13000.0f-3000.0f)) + 1.0f);
    const float ss_taper = 1.0f;
    float dxy4[4] = {dx, dx, __fsqrt_rn(powf(dx,2.0f)+powf(dx,2.0f)),0.0f};
    dxy4[3] = dxy4[2];
    for (int k=1; k<=kte+1; ++k) { taup[k]=0.0f; taufb[k]=0.0f; }
    const float zs = IN(6,1) - 0.5f * IN(7,1);
    // bl_ugwp.F:356-380: carriers and earth-relative winds.
    for (int k=1; k<=kte; ++k) {
        vtj[k] = IN(2,k) * (1.0f + fv*IN(3,k));
        vtk[k] = vtj[k] / IN(5,k);
        rho[k] = 1.0f / rd * IN(4,k) / vtj[k];
        del[k] = P(k)-P(k+1);
        u[k] = IN(0,k)*ca - IN(1,k)*sa;
        v[k] = IN(0,k)*sa + IN(1,k)*ca;
        za[k] = IN(6,k)-zs;
        bnv2[k]=0.0f; ri[k]=0.0f; velco[k]=0.0f;
        taud_ls[k]=0.0f; taud_bl[k]=0.0f; dudt[k]=0.0f; dvdt[k]=0.0f;
        for(int m=2;m<12;++m) O(m,k)=0.0f;
    }
    // bl_ugwp.F:382-481: reference layer and direction.
    int klowtop=0;
    const float zlowtop=2.0f*var;
    if(zlowtop>0.0f) for(int k=2;k<=kte;++k) {
        if(IN(6,k)-IN(6,1)>=zlowtop) { klowtop=k+1; break; }
    }
    const int kbl = max(min(max(kpbl,klowtop),kpblmax),kpblmin);
    const int komax = klowtop-1;
    const float delks = 1.0f/(P(1)-P(kbl));
    const float delks1 = 1.0f/(IN(4,1)-IN(4,kbl));
    float ubar=0.0f,vbar=0.0f,rhobar=0.0f;
    for(int k=1;k<kbl;++k) {
        const float rcsks=del[k]*delks, rdelks=del[k]*delks;
        ubar=ubar+rcsks*u[k]; vbar=vbar+rcsks*v[k]; rhobar=rhobar+rdelks*rho[k];
    }
    // Same shared atan2 policy as terrain_drag.cu: double atan2 -> RN float.
    const float wdir=__double2float_rn(atan2((double)ubar,(double)vbar))+pi_c;
    const int idir=((int)roundf(fdir*wdir))%8;
    const int nwdir[8]={6,7,5,8,2,3,1,4};
    const int nwd=nwdir[idir], m=(nwd-1)%4;
    const float oa=(float)(1-2*((nwd-1)/4))*S(8+m), ol=S(12+m);
    const int cross[4]={1,0,3,2};
    const float olp=S(12+cross[m]);
    const float od=fmaxf(fminf(olp/fmaxf(ol,1.e-5f),10.0f),0.1f);
    const float dxy=dxy4[m];
    const bool ls_on=ls_taper>1.e-02f;
    bool ldrag=false;
    float ulow=0.0f, rulow=0.0f, xn=0.0f,yn=0.0f,coefm=0.0f,taub=0.0f;
    // bl_ugwp.F:487-603: mesoscale base stress, stability and propagation.
    if(ls_on) {
        for(int k=1;k<kte;++k) {
            const float ti=2.0f/(IN(2,k)+IN(2,k+1));
            const float rdz=1.0f/(IN(6,k+1)-IN(6,k));
            const float tem1=u[k]-u[k+1],tem2=v[k]-v[k+1];
            const float dw2=tem1*tem1+tem2*tem2;
            const float shr2=fmaxf(dw2,1.0f)*rdz*rdz;
            const float bvf2=g*(g/cp+rdz*(vtj[k+1]-vtj[k]))*ti;
            ri[k]=fmaxf(bvf2/shr2,-100.0f);
            bnv2[k]=2.0f*g*rdz*(vtk[k+1]-vtk[k])/(vtk[k+1]+vtk[k]);
            bnv2[k]=fmaxf(bnv2[k],1.e-5f);
        }
        ulow=fmaxf(__fsqrt_rn(ubar*ubar+vbar*vbar),1.0f);
        rulow=1.0f/ulow;
        for(int k=1;k<kte;++k) {
            velco[k]=0.5f*((u[k]+u[k+1])*ubar+(v[k]+v[k+1])*vbar);
            velco[k]=velco[k]*rulow;
            if(velco[k]<1.0f && velco[k]>0.0f) velco[k]=1.0f;
        }
        ldrag=velco[1]<=0.0f;
        for(int k=2;k<kbl;++k) ldrag=ldrag || velco[k]<=0.0f;
        const float wtkbj=(IN(4,1)-IN(4,2))*delks1;
        bnv2[1]=wtkbj*bnv2[1]; ri[1]=wtkbj*ri[1];
        for(int k=2;k<kbl;++k) {
            const float rdelks=(IN(4,k)-IN(4,k+1))*delks1;
            bnv2[1]=bnv2[1]+bnv2[k]*rdelks; ri[1]=ri[1]+ri[k]*rdelks;
        }
        ldrag=ldrag || bnv2[1]<=0.0f;
        ldrag=ldrag || ulow==1.0f;
        ldrag=ldrag || var<=0.0f;
        const float ksmax=4.0f*pi_c/dx;
        if(bnv2[1]>0.0f) ldrag=ldrag || __fsqrt_rn(bnv2[1])*rulow<ksmax;
        for(int k=2;k<kbl;++k) ri[k]=ri[1];
        if(!ldrag) {
            const float bnv=__fsqrt_rn(bnv2[1]);
            float fr=bnv*rulow*var*od; fr=fminf(fr,10.0f);
            xn=ubar*rulow; yn=vbar*rulow;
            float efact=powf(oa+2.0f,0.8f*fr/hex_divisor_3f800000);
            efact=fminf(fmaxf(efact,0.0f),10.0f);
            coefm=powf(1.0f+ol,oa+1.0f);
            const float xlinv=coefm*8.84e-6f;
            const float tem=fr*fr*oc1;
            const float gfobnv=1.0f*tem/((tem+0.5f)*bnv);
            taub=xlinv*rhobar*ulow*ulow*ulow*gfobnv*efact;
        }
    }
    // bl_ugwp.F:609-731: small-scale stable boundary-layer waves.
    float hpbl2=hpbl+10.0f;
    int kpbl2=kpbl;
    for(int k=2;k<=max(kpbl,2);++k) {
        // UGWP_INVALID_KPBL: bl_ugwp.F:634-646 reads za(k), but may
        // exit before reaching an excessive KPBL. Check the actual read.
        if(k>nz) { atomicExch(status,1); return; }
        if(za[k]>300.0f) {
            kpbl2=k;
            hpbl2=(k==kpbl)?hpbl+10.0f:za[k]+10.0f;
            break;
        }
    }
    float tauwavex0=0.0f,tauwavey0=0.0f;
    bool wave=false;
    // bl_ugwp.F:614-615,645-709: for kpbl<=0 the native write loop
    // is empty and initialized SS tendencies remain zero, even if the
    // preceding thvx(kpbl2) read is outside storage. Preserve those
    // observable outputs without refusing a native-completing column.
    if(xland-1.5f<=0.0f && 2.0f*varss<=hpbl && br>0.0f && kpbl>0) {
        // UGWP_INVALID_KPBL: bl_ugwp.F:645-685 dereferences kpbl2 only
        // in this branch. Native water/inactive columns may carry zero.
        if(kpbl2<1 || kpbl2>nz) { atomicExch(status,1); return; }
        const float thv1=(IN(2,1)/IN(5,1))*(1.0f+fv*IN(3,1));
        const float thvk=(IN(2,kpbl2)/IN(5,kpbl2))*(1.0f+fv*IN(3,kpbl2));
        if(br>0.0f && thvk-thv1>0.0f) {
            const float xlinv=0.001f*pi_c;
            const float govrth=g/(0.5f*(thvk+thv1));
            const float xnbv=__fsqrt_rn(govrth*(thvk-thv1)/hpbl2);
            if(u[kpbl2]==0.0f || fabsf(xnbv/u[kpbl2])>xlinv) {
                const float tv=2.0f*varss;
                const float vtemp2=0.5f*xnbv*xlinv*(tv*tv)*rho[1];
                tauwavex0=vtemp2*u[1]/(1.0f+vtemp2*dt);
                tauwavex0=tauwavex0*ss_taper;
            }
            if(v[kpbl2]==0.0f || fabsf(xnbv/v[kpbl2])>xlinv) {
                const float tv=2.0f*varss;
                const float vtemp2=0.5f*xnbv*xlinv*(tv*tv)*rho[1];
                tauwavey0=vtemp2*v[1]/(1.0f+vtemp2*dt);
                tauwavey0=tauwavey0*ss_taper;
            }
            wave=true;
        }
    }
    // UGWP_INVALID_KPBL: bl_ugwp.F:701-709 writes through kpbl when
    // waves are active. Inactive columns do not perform those writes.
    if(wave && kpbl>nz) { atomicExch(status,1); return; }
    for(int k=1;k<=kte;++k) {
        float ut=0.0f,vt=0.0f;
        if(wave && k<=kpbl) {
            ut=-1.0f*tauwavex0*2.0f*fmaxf(1.0f-za[k]/hpbl2,0.0f)/hpbl2;
            vt=-1.0f*tauwavey0*2.0f*fmaxf(1.0f-za[k]/hpbl2,0.0f)/hpbl2;
        }
        dudt[k]=dudt[k]+ut; dvdt[k]=dvdt[k]+vt;
        O(2,k)=O(2,k)+ut; O(3,k)=O(3,k)+vt;
        dusfc=dusfc+ut*del[k]; dvsfc=dvsfc+vt*del[k];
        if(diags) { O(8,k)=ut; O(9,k)=vt; dus[2]=dus[2]+ut*del[k]; dvs[2]=dvs[2]+vt*del[k]; }
    }
    // bl_ugwp.F:737-807: semi-implicit Beljaars topographic form drag.
    float var_temp=fminf(varss,500.0f)+fmaxf(0.0f,0.2f*(varss-500.0f));
    const float a1=0.00026615161f*(var_temp*var_temp), a2=a1*0.005363f;
    bool form=xland-1.5f<=0.0f;
    for(int k=1;k<=kte;++k) {
        float ut=0.0f,vt=0.0f;
        if(form) {
            const float wsp=__fsqrt_rn(u[k]*u[k]+v[k]*v[k]);
            var_temp=12.0f*6.325e-3f*expf(-powf(za[k]/hex_divisor_44bb8000,1.5f))*a2
                     *powf(za[k],-1.2f)*ss_taper;
            ut=-var_temp*wsp*u[k]/(1.0f+var_temp*dt*wsp);
            vt=-var_temp*wsp*v[k]/(1.0f+var_temp*dt*wsp);
            if(za[k]/hex_divisor_44bb8000>4.0f) form=false;
        }
        dudt[k]=dudt[k]+ut; dvdt[k]=dvdt[k]+vt;
        O(2,k)=O(2,k)+ut; O(3,k)=O(3,k)+vt;
        dusfc=dusfc+ut*del[k]; dvsfc=dvsfc+vt*del[k];
        if(diags) { O(10,k)=ut; O(11,k)=vt; dus[3]=dus[3]+ut*del[k]; dvs[3]=dvs[3]+vt*del[k]; }
    }
    // bl_ugwp.F:814-878: vertical mesoscale stress and Lindzen saturation.
    if(ls_on) {
        for(int k=1;k<=kbl;++k) taup[k]=taub;
        bool icrilv=false;
        float brvf=0.0f;
        for(int k=2;k<kte;++k) {
            const int kp1=k+1;
            if(k>=kbl) {
                icrilv=icrilv || ri[k]<0.25f || velco[k]<=0.0f;
                brvf=__fsqrt_rn(fmaxf(bnv2[k],1.e-5f));
            }
            if(k>=kbl && !ldrag) {
                if(!icrilv && taup[k]>0.0f) {
                    const float temv=1.0f/velco[k];
                    const float tem1=coefm/dxy*(rho[kp1]+rho[k])*brvf*velco[k]*0.5f;
                    float hd=__fsqrt_rn(taup[k]/tem1);
                    const float fro=brvf*hd*temv;
                    const float tem2=__fsqrt_rn(ri[k]), tem=1.0f+tem2*fro;
                    const float rim=ri[k]*(1.0f-fro)/(tem*tem);
                    if(rim<=0.25f) {
                        if(oa<=0.0f || kp1>=kpblmin) {
                            const float temc=2.0f+1.0f/tem2;
                            hd=velco[k]*(2.0f*__fsqrt_rn(temc)-temc)/brvf;
                            taup[kp1]=tem1*hd*hd;
                        }
                    } else taup[kp1]=taup[k];
                }
            }
        }
        // bl_ugwp.F:884-940: flow-blocking layer and stress.
        if(!ldrag) {
            int kblk=0;
            float pe=0.0f,zblk=0.0f;
            for(int k=kte;k>=2;--k) {
                if(kblk==0 && k<=komax) {
                    pe=pe+bnv2[k]*(IN(6,komax)-IN(6,k))*del[k]/g/rho[k];
                    const float ke=0.5f*(powf(u[k],2.0f)+powf(v[k],2.0f));
                    if(pe>=ke) { kblk=min(k,kbl); zblk=za[kblk]; }
                }
            }
            if(kblk!=0) {
                const float cd=fmaxf(2.0f-1.0f/od,0.0f);
                taufb[1]=0.5f*1.92e-4f*rhobar*coefm*cd*olp*zblk*(ulow*ulow);
                const float tautem=taufb[1]/(float)(kblk-1);
                for(int k=2;k<=kblk;++k) taufb[k]=taufb[k-1]-tautem;
            }
        }
        // bl_ugwp.F:943-1025: tendencies, surface sigma and mesospheric limiters.
        taup[kte+1]=taup[kte];
        for(int k=1;k<=kte;++k) {
            taud_ls[k]=1.0f*(taup[k+1]-taup[k])*g/del[k];
            taud_bl[k]=1.0f*(taufb[k+1]-taufb[k])*g/del[k];
        }
        float dtfac=1.0f;
        for(int k=1;k<kpblmax;++k) {
            if(P(k)>=0.5f*P(1)) {
                const float s=taud_ls[k]+taud_bl[k];
                if(s!=0.0f) dtfac=fminf(dtfac,fabsf(velco[k]/(dt*s)));
            } else break;
        }
        for(int k=1;k<=kte;++k) {
            float dtfac_meso=1.0f;
            if(IN(4,k)<=70.0f) {
                if(taud_ls[k]!=0.0f)
                    dtfac_meso=fminf(dtfac_meso,0.5f*fabsf(velco[k]/(dt*taud_ls[k])));
            }
            taud_ls[k]=taud_ls[k]*dtfac*dtfac_meso*ls_taper;
            taud_bl[k]=taud_bl[k]*dtfac*dtfac_meso*ls_taper;
            const float ux=taud_ls[k]*xn,vy=taud_ls[k]*yn;
            const float ub=taud_bl[k]*xn,vb=taud_bl[k]*yn;
            dudt[k]=ux+ub+dudt[k]; dvdt[k]=vy+vb+dvdt[k];
            O(2,k)=O(2,k)+ux+ub; O(3,k)=O(3,k)+vy+vb;
            dusfc=dusfc+(ux+ub)*del[k]; dvsfc=dvsfc+(vy+vb)*del[k];
            if(diags) {
                O(4,k)=ux; O(5,k)=vy; O(6,k)=ub; O(7,k)=vb;
                dus[0]=dus[0]+ux*del[k]; dvs[0]=dvs[0]+vy*del[k];
                dus[1]=dus[1]+ub*del[k]; dvs[1]=dvs[1]+vb*del[k];
            }
        }
    }
    // bl_ugwp.F:1019-1059: integrate earth-relative stresses, rotate tendencies.
    SO(0)=-inv_g*dusfc; SO(1)=-inv_g*dvsfc;
    if(diags) for(int m=0;m<4;++m) { SO(2+2*m)=-inv_g*dus[m]; SO(3+2*m)=-inv_g*dvs[m]; }
    for(int k=1;k<=kte;++k) {
        O(0,k)=IN(8,k)+dudt[k]*ca+dvdt[k]*sa;
        O(1,k)=IN(9,k)-dudt[k]*sa+dvdt[k]*ca;
    }
    #undef IN
    #undef S
    #undef P
    #undef O
    #undef SO
    #undef FL
    #undef ZI
    (void)sink;
}
