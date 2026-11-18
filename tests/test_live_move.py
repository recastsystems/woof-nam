"""namgpu/nest/live_move.py host parts (no card): the reach cut and the seam band.

The whole move is proven on a real run (woof-nam nest --concurrent --move-nest): every carried array checked bit for
bit at every move, and a forced 0-cell move byte-identical to no move (receipts on the branch's page)."""
import pickle

import numpy as np

from namgpu.init import slab_init
from namgpu.nest import live_move as lm


def _envelope(tmp_path, jme=13, ime=17, lm_=4):
    rng = np.random.default_rng(0)
    owners = {0: rng.random((lm_, jme, ime), dtype=np.float32),       # 3-D state [LM, JM, IM]
              1: rng.random((2, lm_, 5, ime), dtype=np.float32),      # S/N strip [2, NL, w, IM]
              2: rng.random((2, lm_, jme, 5), dtype=np.float32),      # W/E strip [2, NL, JM, w]
              3: rng.random(jme).astype(np.float32),                  # per-row table
              4: rng.random(7).astype(np.float32)}                    # a global table
    for o, a in owners.items():
        np.save(tmp_path / f"a{o}.npy", a)
    meta = {o: {"owner": o, "dev": False, "dtype": "float32", "shape": list(a.shape), "off": 0, "strides": [],
                "c": True} for o, a in owners.items()}
    d = {"format": slab_init.FORMAT, "skel": {"st": {}, "bc": {}, "tables": {}, "attrs": {}}, "vgrid": {},
         "meta": meta, "jm": jme, "im": ime, "lm": lm_}
    with open(tmp_path / "skeleton.pkl", "wb") as f:
        pickle.dump(d, f)
    return owners


def test_reach_cut_takes_rows_and_columns_only(tmp_path):
    owners = _envelope(tmp_path)
    env = lm.Envelope(tmp_path, -4, -2)
    r, c = slice(2, 9), slice(4, 15)
    assert env.axes(owners[0].shape) == (1, 2)
    assert env.axes(owners[1].shape) == (None, 3)
    assert env.axes(owners[2].shape) == (2, None)
    assert np.array_equal(env.cut_owner(0, r, c), owners[0][:, r, c])
    assert np.array_equal(env.cut_owner(1, r, c), owners[1][..., c])
    assert np.array_equal(env.cut_owner(2, r, c), owners[2][:, :, r, :])
    assert np.array_equal(env.cut_owner(3, r, c), owners[3][r])
    g = env.cut_owner(4, r, c)
    assert np.array_equal(g, owners[4])
    g[0] = -1.0                       # a writable copy, never the read-only map (smoke1, 2026-11-18)
    assert owners[4][0] != -1.0


def test_seam_band_is_kept_cells_next_to_the_strip():
    strip = np.zeros((10, 12), bool)
    strip[:, 9:] = True                                 # a 3-column east strip
    band = lm._band(strip, 2)
    assert not (band & strip).any()
    assert band[:, 7:9].all() and not band[:, :7].any()
    assert not lm._band(strip, 0).any()
