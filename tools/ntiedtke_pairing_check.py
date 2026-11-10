import os, sys, types
from types import SimpleNamespace as NS
from namgpu.physics.ntiedtke import phys_ntiedtke_cu as m
def cfg(turb="myj", **raw): return NS(turbulence=turb, raw=raw)
rows = []
os.environ.pop("NAMGPU_PBL", None)
rows.append(("myj auto", m.shallow_off(cfg())))
rows.append(("ysu auto", m.shallow_off(cfg("ysu"))))
rows.append(("mynn cfg edmf default", m.shallow_off(cfg("mynn"))))
rows.append(("mynn cfg bl_mynn_edmf=0", m.shallow_off(cfg("mynn", bl_mynn_edmf=0))))
rows.append(("myj + force on", m.shallow_off(cfg(ntiedtke_shallow="on"))))
rows.append(("myj + force off", m.shallow_off(cfg(ntiedtke_shallow="off"))))
# the MYNN lane's module present (as once merged): selected() via NAMGPU_PBL, MYNN_OPTIONS dict
fake = types.ModuleType("namgpu.physics.mynn.phys_mynn_edmf")
fake.MYNN_OPTIONS = {"bl_mynn_edmf": 1}
fake.selected = lambda: os.environ.get("NAMGPU_PBL", "myj") in ("mynn", "mynn-edmf")
pkg = types.ModuleType("namgpu.physics.mynn"); pkg.phys_mynn_edmf = fake
sys.modules["namgpu.physics.mynn"] = pkg; sys.modules["namgpu.physics.mynn.phys_mynn_edmf"] = fake
os.environ["NAMGPU_PBL"] = "mynn-edmf"
rows.append(("lane module, NAMGPU_PBL=mynn-edmf, edmf 1", m.shallow_off(cfg())))
fake.MYNN_OPTIONS = {"bl_mynn_edmf": 0}
rows.append(("lane module, NAMGPU_PBL=mynn-edmf, edmf 0", m.shallow_off(cfg())))
fake.MYNN_OPTIONS = {"bl_mynn_edmf": 1}
rows.append(("lane module + force on (warns)", m.shallow_off(cfg(ntiedtke_shallow="on"))))
os.environ["NAMGPU_PBL"] = "myj"
rows.append(("lane module, NAMGPU_PBL=myj", m.shallow_off(cfg())))
for r, v in rows: print(f"{r:45s} shallow {'OFF' if v else 'on'}")
