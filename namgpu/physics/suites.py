"""Physics suites: which physics a NAM run uses, as table work (parm/physics_suites.json).

The NMMB dycore is the same for every suite. A suite row names (1) the NMMB configfile keys it overrides
(microphysics tracer table, convection, species advection, NMMB GWD), (2) the PHYSICS step plan it runs, and for
an ArWen-physics suite (3) the ArWen selection the column seam runs (gpuwm.core.mpas_column_suite, the same seam
gpuwm-hex uses for ArWen physics on MPAS columns). No code path keys on a suite name: a new suite is a new row.

Selection: NAMGPU_PHYSICS_SUITE (set by `woof-nam run --physics-suite NAME`, inherited by ranks and children) or
the configfile key `physics_suite:`. Unset means the table default (`nam`), and the default changes nothing:
config.load() returns the configfile as read and the driver runs the base PHYSICS plan, byte for byte.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache

from ..config import PARM

TABLE = PARM / "physics_suites.json"
ENV = "NAMGPU_PHYSICS_SUITE"


@lru_cache(maxsize=1)
def table() -> dict:
    t = json.loads(TABLE.read_text())
    if t.get("default") not in t.get("suites", {}):
        raise ValueError(f"{TABLE}: default {t.get('default')!r} has no row")
    return t


def names() -> list[str]:
    return sorted(table()["suites"])


def selected(raw: dict | None = None) -> str:
    name = os.environ.get(ENV) or (raw or {}).get("physics_suite") or table()["default"]
    name = str(name).strip()
    if name not in table()["suites"]:
        raise SystemExit(f"physics suite {name!r} has no row in {TABLE.name}; suites: {names()}")
    return name


def row(name: str | None = None) -> dict:
    return table()["suites"][name or selected()]


def prepare(raw: dict) -> dict:
    """Before config.load builds the NamConfig: the selected row's `env` (lane knobs a run reads from the
    environment, set only where the caller left them unset, so an explicit NAMGPU_* still wins and ranks/children
    inherit them) and its `configfile` keys (as if written in the configfile). Rows without either change nothing."""
    r = row(selected(raw))
    for k, v in (r.get("env") or {}).items():
        os.environ.setdefault(str(k), str(v))
    if r.get("configfile"):
        raw = dict(raw, **r["configfile"])
    return raw


def apply(cfg) -> str:
    """Override the configfile keys the selected suite owns (config.load calls this). Returns the suite name."""
    name = selected(cfg.raw)
    r = row(name)
    cfg.raw["physics_suite"] = name
    for key, val in (r.get("nmmb") or {}).items():
        cfg.raw[key] = val
        if hasattr(cfg, key):
            setattr(cfg, key, val)
    return name


def physics_plan(cfg, base: list) -> list:
    """The PHYSICS step plan for the run's suite: the base plan for `plan: base`, else the row's entries."""
    r = row(selected(cfg.raw))
    if r.get("plan", "base") == "base":
        return base
    out = []
    for e in r["plan"]:
        out.append(tuple(e[:2]) + ((dict(e[2]),) if len(e) > 2 else ()))
    return out
