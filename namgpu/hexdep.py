"""What in woof-nam runs gpuwm-hex code, and one plain line when gpuwm-hex is not there.

woof-nam runs on gpuwm alone (`pip install recast-woof-nam`): the NAM default physics, every grid, nest and the suite,
the GRIB2 products, maps and scores, and the options built on gpuwm's schemes (Thompson mp8, aerosol-aware Thompson
mp28, P3, RUC, Noah-MP, RRTMG, RRTMGP, YSU, MYNN-EDMF, Grell-Freitas, New Tiedtke, UGWP drag).

Two physics pieces come from gpuwm-hex 0.5.0 or newer (`pip install "recast-woof-nam[modern]"`):
  TEMPO microphysics (hex's column kernels, its tables and the NOAA lookup files it fetches): NAMGPU_MP=tempo, and
    every physics suite whose row puts TEMPO in the microphysics slot (hrrrv5, rapv5, woof-nam-next-12km,
    woof-nam-next-7p6km; rapv5 runs mp28 but keeps TEMPO's tracer table and tables);
  MYNN v6.1 PBL and surface layer: NAMGPU_PBL=mynn-v61.
A run that asks for one of them without gpuwm-hex stops before any work, with one line naming what it asked for and
how to install it. Before this, it fetched and built its start and then stopped at the physics start with a traceback.
"""
from __future__ import annotations

import importlib.metadata as md
import importlib.util
import itertools
import os
import sys

MIN = (0, 5, 0)
NEEDS = "gpuwm-hex 0.5.0 or newer"
INSTALL = 'pip install "recast-woof-nam[modern]"'
TEMPO = "tempo"           # the column microphysics config.MP_COLUMNS knows (namgpu/config.py)


def _version_tuple(v: str) -> tuple[int, int, int]:
    nums = [int("".join(itertools.takewhile(str.isdigit, p)) or 0) for p in v.split(".")[:3]]
    return tuple(nums + [0] * (3 - len(nums)))  # type: ignore[return-value]


def wanted(suite: str | None = None) -> list[str]:
    """What this run asks for that runs gpuwm-hex code: its physics suite (the argument, else NAMGPU_PHYSICS_SUITE,
    else the table's default) and the NAMGPU_MP / NAMGPU_PBL knobs, read as config.load will read them (a knob the
    caller set wins over the suite row's env)."""
    from namgpu.physics import suites
    t = suites.table()
    name = str(suite or os.environ.get(suites.ENV) or t["default"]).strip()
    row = t["suites"].get(name) or {}
    env = {str(k): str(v) for k, v in (row.get("env") or {}).items()}
    mp = os.environ.get("NAMGPU_MP", env.get("NAMGPU_MP", "")).strip().lower()
    pbl = os.environ.get("NAMGPU_PBL", env.get("NAMGPU_PBL", "")).strip().lower()
    out = []
    if str((row.get("nmmb") or {}).get("microphysics", "")).strip().lower() == TEMPO:
        out.append(f"--physics-suite {name} (TEMPO microphysics)")
    elif mp == TEMPO:
        out.append("NAMGPU_MP=tempo (TEMPO microphysics)" if "NAMGPU_MP" in os.environ
                   else f"--physics-suite {name} (TEMPO microphysics)")
    if pbl == "mynn-v61":
        out.append("NAMGPU_PBL=mynn-v61 (MYNN v6.1)" if "NAMGPU_PBL" in os.environ
                   else f"--physics-suite {name} (MYNN v6.1)")
    return out


def missing() -> str | None:
    """None when gpuwm-hex's hexcore can be imported at 0.5.0 or newer; otherwise what is wrong, in a few words."""
    extra = os.environ.get("NAMGPU_HEXCORE_PATH")      # a hex source tree's src, as the MYNN v6.1 route reads it
    if extra and extra not in sys.path:
        sys.path.insert(0, extra)
    try:
        spec = importlib.util.find_spec("hexcore")
    except (ImportError, ValueError):
        spec = None
    if spec is None:
        return "gpuwm-hex is not installed"
    try:
        v = md.version("gpuwm-hex")
    except md.PackageNotFoundError:
        return None             # a hex source tree on the path (NAMGPU_HEXCORE_PATH or a .pth file): used as it is
    if _version_tuple(v) < MIN:
        return f"the installed gpuwm-hex is {v}"
    return None


def installed() -> str | None:
    """The installed gpuwm-hex version, 'source tree' for a hexcore on the path, or None."""
    if missing() is not None:
        return None
    try:
        return md.version("gpuwm-hex")
    except md.PackageNotFoundError:
        return "source tree"


def require(suite: str | None = None, prog: str = "woof-nam") -> None:
    """Stop with one line when the run asks for a gpuwm-hex piece and gpuwm-hex is missing or too old."""
    want = wanted(suite)
    if not want:
        return
    why = missing()
    if why is None:
        return
    raise SystemExit(f"{prog}: {' and '.join(want)} needs {NEEDS}, and {why}. Install it with {INSTALL}. "
                     "Without it, the NAM default and every option that does not use TEMPO or MYNN v6.1 run on "
                     "gpuwm alone.")
