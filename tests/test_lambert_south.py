"""The Lambert product grids south of the equator (products/sets.py OutGrid): forward and inverse agree, so a nest's
box there is placed on the nest. Breakage it prevents: a WOOF-NAM box at 10.5S 164W (1.5 km, 343 x 400) had its hiresf
product grid placed at 9.98E to 16.8E, 91.6 degrees east of the box (inverse took theta without the cone's sign, so the
longitude was off by 180/n degrees), and products.nest_window refused every hour's map frame: "the nest does not reach
the hiresf output grid: no product point lies on it" (2026-11-18)."""
import numpy as np

from namgpu.products import sets as S


def box(lat0, lon0, half=2.7, n=60):
    return np.meshgrid(np.linspace(lat0 - half, lat0 + half, n), np.linspace(lon0 - half, lon0 + half, n), indexing="ij")


def test_a_southern_nest_grid_lies_on_the_nest():
    lat, lon = box(-10.5, -164.0)
    og = S.from_nest_grid(lat, lon, 1500.0)
    la, lo = og.latlon()
    assert abs(float(la.min()) - lat.min()) < 0.3 and abs(float(la.max()) - lat.max()) < 0.3
    lo = ((lo + 180.0) % 360.0) - 180.0
    assert abs(float(lo.min()) - lon.min()) < 0.3 and abs(float(lo.max()) - lon.max()) < 0.3


def test_forward_then_inverse_returns_the_point_in_both_hemispheres():
    for lat0, lon0 in ((-10.5, -164.0), (-43.6, 170.8), (36.0, -98.0), (53.5, -4.0)):
        lat, lon = box(lat0, lon0, n=9)
        og = S.from_nest_grid(lat, lon, 3000.0)
        x, y = og.forward(lat, lon)
        la, lo = og.inverse(x, y)
        assert np.allclose(la, lat, atol=1e-6)
        assert np.allclose(((lo - lon + 180.0) % 360.0) - 180.0, 0.0, atol=1e-6)
