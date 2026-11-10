"""WRF v4.6.1 calc_coszen solar geometry (from ArWen gpuwm/core/dudhia.py, Apache-2.0; WRF public domain)."""
from __future__ import annotations
from datetime import datetime
import numpy as np

def _array_namespace(*values):
    """Return CuPy only when at least one input is a CUDA array."""
    if any(hasattr(value, "__cuda_array_interface__") for value in values):
        import cupy as cp
        return cp
    return np


def wrf_solar_geometry(valid_time: datetime, latitude_deg, longitude_deg,
                       *, hour_offset_seconds: float = 0.0):
    """Return WRF v4.6.1 ``(coszen, solcon)`` for one radiation call.

    This is the standard ``radconst``/``calc_coszen`` path in
    ``module_radiation_driver.F:3469-3541``.  As in WRF, only the hour
    angle receives the half-radiation-interval offset; declination and the
    equation of time use the unshifted call-time Julian day.
    """
    if not isinstance(valid_time, datetime):
        raise TypeError("valid_time must be a datetime")
    if not np.isfinite(hour_offset_seconds):
        raise ValueError("hour_offset_seconds must be finite")
    xp = _array_namespace(latitude_deg, longitude_deg)
    lat = xp.asarray(latitude_deg)
    lon = xp.asarray(longitude_deg)
    if lat.shape != lon.shape:
        raise ValueError("radiation latitude/longitude shapes must match")

    hour = (valid_time.hour + valid_time.minute / 60.0
            + valid_time.second / 3600.0
            + valid_time.microsecond / 3.6e9)
    julian = valid_time.timetuple().tm_yday - 1.0 + hour / 24.0
    degrad = np.pi / 180.0
    dpd = 360.0 / 365.0
    solar_longitude = dpd * (
        julian - 80.0 if julian >= 80.0 else julian + 285.0)
    declination = np.arcsin(
        np.sin(23.5 * degrad) * np.sin(solar_longitude * degrad))
    da = 2.0 * np.pi * (julian - 1.0) / 365.0
    equation = 229.18 * (
        0.000075 + 0.001868 * np.cos(da) - 0.032077 * np.sin(da)
        - 0.014615 * np.cos(2.0 * da) - 0.04089 * np.sin(2.0 * da))
    solar_minutes = (60.0 * (hour + hour_offset_seconds / 3600.0)
                     + equation + 4.0 * lon)
    hour_angle = xp.deg2rad(solar_minutes / 4.0 - 180.0)
    latitude = xp.deg2rad(lat)
    cosine_zenith = (
        xp.sin(latitude) * np.sin(declination)
        + xp.cos(latitude) * np.cos(declination) * xp.cos(hour_angle))
    cosine_zenith = xp.clip(cosine_zenith, -1.0, 1.0)

    # module_radiation_driver.F:3504-3509 (Paltridge & Platt orbit).
    orbit = 2.0 * np.pi * julian / 365.0
    eccentricity = (
        1.000110 + 0.034221 * np.cos(orbit) + 0.001280 * np.sin(orbit)
        + 0.000719 * np.cos(2.0 * orbit)
        + 0.000077 * np.sin(2.0 * orbit))
    return cosine_zenith, 1370.0 * eccentricity


