"""woof-nam-next presets: table rows only. Breakage they prevent: a preset that names a grid that does not exist,
a preset whose knobs the config ignores, or a preset row that changes the frozen nam default."""
import json
import os

from namgpu.config import PARM
from namgpu.physics import suites


def test_rows_name_real_grids_and_keep_the_default():
    t = json.loads((PARM / "physics_suites.json").read_text())
    doms = json.loads((PARM / "domains.json").read_text())
    doms = doms.get("domains", doms)
    assert t["default"] == "nam" and "env" not in t["suites"]["nam"]
    for name, grid, cu in (("woof-nam-next-12km", "conus12", None), ("woof-nam-next-7p6km", "na7p6", "ntiedtke")):
        r = t["suites"][name]
        assert r["domain"] in doms, (name, r["domain"])
        assert r["domain"] == grid
        assert r["env"]["NAMGPU_LSM"] == "ruc" and r["env"]["NAMGPU_RUC_URBAN"] == "noah"
        assert r["env"].get("NAMGPU_CONVECTION") == cu


def test_preset_env_reaches_the_config_keys(monkeypatch):
    env = {k: v for k, v in os.environ.items() if not k.startswith("NAMGPU_")}   # prepare() writes os.environ
    env[suites.ENV] = "woof-nam-next-7p6km"
    monkeypatch.setattr(os, "environ", env)
    suites.table.cache_clear()
    raw = suites.prepare({})
    assert os.environ["NAMGPU_CONVECTION"] == "ntiedtke" and os.environ["NAMGPU_MP"] == "tempo"
    assert suites.selected(raw) == "woof-nam-next-7p6km"
    suites.table.cache_clear()
