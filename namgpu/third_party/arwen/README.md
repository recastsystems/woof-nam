# ArWen files copied into woof-nam

Copied 2026-11-04 from the ArWen source tree (Python package "gpuwm", 2.9.11 line),
Apache-2.0 (LICENSE here). Copyright 2026 ArWen contributors. Copies, changed in comment wording only; the woof-nam
units of mode "copy-arwen" adapt them in namgpu/physics/{myj,jsfc,noah}/ and keep this notice.

| file | ArWen path |
|---|---|
| core/myjpbl.py, core/kernels/myjpbl.cu | gpuwm/core/myjpbl.py, gpuwm/core/kernels/myjpbl.cu (MYJ PBL, WRF v4.6.1 lineage) |
| core/myjsfc.py, core/myjsfc_tables.py, core/kernels/myjsfc.cu | Eta/MYJ surface layer |
| core/noah.py, core/kernels/noah.cu, core/kernels/noah_init.cu | Noah LSM |
| verify/myj_ref.py | float32 CPU transcription of WRF MYJ/Eta (reading aid only) |

Known differences to close (FEASIBILITY.md): the NAM MYJ copy is ~26% line-identical to WRF
v4.7.1, the NAM JSFC ~80%, and NAM Noah carries the NCEP v4 changes (frozen-soil
transpiration, reduced roughness for five vegetation types) that WRF's Noah lacks. ArWen
kernels are bottom-up columns; NMMB is top-down (l=1 top).

## New Tiedtke (lane cu-tiedtke, 2026-11-08)

| file | ArWen path | sha256 of the ArWen file |
|---|---|---|
| core/ntiedtke.py | gpuwm/core/ntiedtke.py (WRF v4.6.1 module_cu_ntiedtke.F lineage, cu_physics 16) | f77792c3c8abd2cf... |
| core/kernels/ntiedtke.cu | gpuwm/core/kernels/ntiedtke.cu | 06daa934a71a0279... |
| core/kernels/ntiedtke_assembled.cu | gpuwm.core.kernels.module_source("ntiedtke"): CUDA_DEFINES preamble + common.cuh + FP32 maths header + ntiedtke.cu, the exact NVRTC input in ArWen | a6e092d7142982a1... |

The sha256 column is of the ArWen files; all three copies here also differ from them in comment wording.
ntiedtke.py differs from the ArWen copy in three imports only (no gpuwm dependency): DTYPE is float32 here,
NtStages.launch compiles ntiedtke_assembled.cu with the ArWen options ("-std=c++17",), and nt_tile_columns
uses GF_BLOCK = 64, GF_TILE_BLOCKS_PER_SM = 4 (gpuwm/core/gf.py:129,139). The NewTiedtke class at the end
is the ArWen WRF adapter and is not used here; namgpu/physics/ntiedtke/ is the NMMB seam.
