"""The Lambert grid's inverse in both hemispheres (namgpu/products/sets.py OutGrid): a point projected and brought back
lands where it was. Breakage: southern-hemisphere boxes came back 115-145 degrees of longitude away, so their catalog map
frame was refused (18 Nov 2026)."""
import numpy as np
import pytest

from namgpu.products.sets import OutGrid, from_nest_grid


@pytest.mark.parametrize("lat0, lon0", [(-10.75, 196.75), (-43.6, 170.8), (-60.0, 300.0), (10.75, 196.75),
                                        (38.5, 262.5)])
def test_lambert_round_trip(lat0, lon0):
    m = OutGrid("lambert", 2, 2, lat0, lon0, 12000.0, 12000.0, lov=lon0, latin1=lat0, latin2=lat0)
    lat = np.array([lat0, lat0 + 2.0, lat0 - 3.0])
    lon = np.array([lon0, lon0 + 3.0, lon0 - 4.0])
    x, y = m.forward(lat, lon)
    la, lo = m.inverse(x, y)
    assert np.abs(la - lat).max() < 1e-6
    assert np.abs(((lo - lon + 180.0) % 360.0) - 180.0).max() < 1e-6


def test_a_southern_box_lies_inside_its_own_nest_grid():
    lat, lon = np.meshgrid(np.linspace(-21.0, -0.5, 40), np.linspace(-174.0, -152.5, 50), indexing="ij")
    og = from_nest_grid(lat, lon, 12000.0)
    la, lo = og.latlon()
    lo = ((lo + 180.0) % 360.0) - 180.0
    assert -22.0 < la.min() and la.max() < 0.5
    assert -175.0 < lo.min() and lo.max() < -151.0
