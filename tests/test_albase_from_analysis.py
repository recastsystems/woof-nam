"""The snow-free base albedo comes from the archived NAM analysis' own albedo, not from a vegetation-class fill.

Breakage this prevents: the analysis file has no ALBASE record, so every cell's base albedo was
init-static-missing's table fill. From the first land step the albedo of a run started on the NAM's own analysis
(2024-11-05 12Z) sat 3.2 points above the archived NAM's on 76,057 snow-free land cells (mean abs 4.0, pattern
correlation 0.60), 5-15 points over the Mojave and the Colorado Plateau, where the 2 m temperature ran 1-3 F cool
at 21Z; the NAM's own albedo on those cells does not change from +0 h to +9 h.
"""
from __future__ import annotations

import inspect

import numpy as np

from namgpu.init import gfs_start, sources


def test_albase_is_the_analysis_albedo_on_snow_free_land_and_missing_elsewhere():
    # columns: desert, grass, snowy, trace of snow water, water, invalid albedo
    albdo = np.array([[27.0, 18.5, 52.0, 21.0, 6.0, 0.0]], dtype=np.float32)
    snowc = np.array([[0.0, 0.0, 0.64, 0.0, 0.0, 0.0]], dtype=np.float32)
    weasd = np.array([[0.0, 0.0, 21.0, 0.3, 0.0, 0.0]], dtype=np.float32)
    land = np.array([[1.0, 1.0, 1.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    a = sources.albase_from_analysis(albdo, snowc, weasd, land)
    assert a.dtype == np.float32
    np.testing.assert_allclose(a[0, :2], [0.27, 0.185], rtol=1e-6)     # snow-free land: the analysis albedo
    assert (a[0, 2:] == 0).all()                                      # snow, water, invalid: left to the fill
    # snow cover in percent (the record's own unit) gives the same cells
    np.testing.assert_array_equal(sources.albase_from_analysis(albdo, snowc * 100.0, weasd, land), a)


def test_the_starts_stage_it_for_init_static_missing():
    # the analysis' snow-free albedo is the fallback the NCEP native surface file's own fields go over
    # (sources.land_from_surface_file; tests/test_native_land_file.py)
    src = inspect.getsource(sources.stage_native)
    assert "albase = albase_from_analysis(" in src and "land_from_surface_file(rec, albase" in src
    assert 'ctx.tables["static_extra"] = extra' in src
    src = inspect.getsource(gfs_start.load_static)
    assert "srcmod.albase_from_analysis(" in src and "STATIC_SNOW" in src
    src = inspect.getsource(gfs_start.build_static)
    assert "srcmod.albase_from_analysis(" in src and "land_from_surface_file(rec, albase" in src
    assert "**{k: cp.asnumpy(v) for k, v in extra.items()}" in src
