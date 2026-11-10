# arwen_noahmp (vendored)

Noah-MP land surface model, WRF v4.6.1 lineage (`phys/noahmp`, WRF public-domain release),
as ported to CUDA/CuPy in ArWen (gpuwm `gpuwm/core/noahmp*.py`, `gpuwm/core/kernels/noahmp*.cu`),
Apache-2.0 (LICENSE here). Tables: WRF v4.6.1 MPTABLE.TBL / SOILPARM.TBL / GENPARM.TBL
(see data/noahmp/PROVENANCE.md). Not the UCAR-licensed Noah-MP v5.0 (MPAS) code.

Copied unchanged except for: import paths (`gpuwm.core` -> `namgpu.third_party.arwen_noahmp.core`);
`core/dudhia.py`, which keeps only `wrf_solar_geometry`; and `core/noahmp_column_slab.py`
`parameter_fields`, which keeps the per-identity device parameter tables between calls instead of
rebuilding them on the host every call (same values and gather, state digest identical over 5 calls on
the full 12 km NAM domain; ~2x faster per call). That last change belongs back in ArWen too. The NMMB adapter is
`namgpu/physics/noahmp/phys_noahmp_driver.py`.
