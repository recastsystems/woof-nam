"""woof-nam suite: nests that share a card start only when their memory estimates fit the card together.

Release check 2026-11-10 (suite-1h-d): five nests started at once on one 95 GiB card, every door read 94.42 GiB free and
sized itself to the whole card (226.6 GiB of estimates), and CONUS, Hawaii and fire weather ran out of memory."""
import threading
from types import SimpleNamespace

from namgpu.nest import domain
from namgpu.suite import cli


def _est(total, path="fit", rows=32):
    return {"total_gib": total, "memory_path": path, "rrtm_rows": rows}


# the doors' estimates from suite-1h-d (the first that fits, in the door's order), CONUS and Alaska on the fit path
EST = {"conusnest": [76.14], "alaskanest": [44.98, 40.0], "hawaiinest": [43.69, 10.0], "priconest": [32.89, 9.0],
       "firewxnest": [28.91, 8.0]}


def _suite(monkeypatch, free=94.42, total=95.0):
    s = cli.Suite.__new__(cli.Suite)
    s.lock = threading.Lock()
    s.cv = threading.Condition()
    s.claims, s.claim_procs, s.waiting, s.granted, s.cands = {}, {}, [], {}, {}
    s.t = {"parent": {"domain": "12km"}}
    s.a = SimpleNamespace(parent_run=None)
    s.rec = {"domains": {}}
    s.plan = {"domains": {"12km": {"kind": "parent", "cards": ["2"]},
                          **{n: {"kind": "nest", "cards": ["2"]} for n in EST}}}
    s.said = []
    monkeypatch.setattr(cli.Suite, "say", lambda self, m: self.said.append(m))
    monkeypatch.setattr(cli.Suite, "_candidates", lambda self, d: [{"2": _est(t)} for t in EST[d]])
    monkeypatch.setattr(domain, "free_gib", lambda c=None: free)
    monkeypatch.setattr(domain, "card_total_gib", lambda c=None: total)
    monkeypatch.setattr(cli, "_used_gib_by_pid", lambda c: {})
    return s


def test_nests_on_one_card_are_granted_together_only_while_they_fit(monkeypatch):
    s = _suite(monkeypatch)
    s.waiting = list(EST)
    s._grant()
    # CONUS first (plan order) with its estimate + margin; then only what still fits beside it
    assert s.granted["conusnest"] == 76.9
    assert "alaskanest" in s.waiting
    assert s.granted["hawaiinest"] == 10.76                      # 94.42 - 76.9 = 17.52 left: Hawaii on its small path
    assert "priconest" in s.waiting and "firewxnest" in s.waiting  # 6.76 left: neither fits
    assert sum(s.claims["2"].values()) <= 94.42
    assert any("conusnest: starts with 76.9 GiB" in m for m in s.said)


def test_a_running_door_that_has_taken_its_memory_is_not_counted_twice(monkeypatch):
    s = _suite(monkeypatch, free=94.42 - 60.0)               # CONUS has taken 60 GiB of its 76.9
    s.claims = {"2": {"conusnest": 76.9}}
    s.claim_procs = {"conusnest": SimpleNamespace(p=SimpleNamespace(pid=4242))}
    monkeypatch.setattr(cli, "_tree_pids", lambda pid: {pid, pid + 1})
    monkeypatch.setattr(cli, "_used_gib_by_pid", lambda c: {4243: 60.0})
    assert abs(s._avail("2") - (34.42 - 16.9)) < 1e-9


def test_a_nest_larger_than_the_card_goes_to_its_door_to_be_refused_by_name(monkeypatch):
    s = _suite(monkeypatch, free=20.0, total=24.0)
    s.waiting = ["conusnest"]
    s._grant()
    assert s.granted == {"conusnest": None} and s.waiting == []


def test_a_nest_that_fits_an_empty_card_waits_while_others_hold_it(monkeypatch):
    s = _suite(monkeypatch, free=20.0, total=95.0)
    s.claims = {"2": {"alaskanest": 45.73}}
    s.waiting = ["conusnest"]
    s._grant()
    assert s.granted == {} and s.waiting == ["conusnest"]


def test_the_door_reads_no_more_than_its_share(monkeypatch):
    monkeypatch.undo()
    monkeypatch.setattr(domain.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout="96686\n"))
    monkeypatch.delenv("NAMGPU_CARD_SHARE_GIB", raising=False)
    assert abs(domain.free_gib("0") - 96686 / 1024) < 1e-9
    monkeypatch.setenv("NAMGPU_CARD_SHARE_GIB", "10.76")
    assert domain.free_gib("0") == 10.76


def test_the_init_pool_stays_inside_the_share(monkeypatch):
    from namgpu.nest import cli as door
    pick = {"memory_path": "fit", "pool_limit_bytes": int(73.2 * 2**30), "pool_limit_gib": 73.2, "lmem_gib": 2.69,
            "fixed_gib": 0.5}
    monkeypatch.setenv("NAMGPU_CARD_SHARE_GIB", "76.9")
    env, rec = {}, {}
    door._pool_cap(env, rec, pick, 76.9)
    assert rec["init_pool_limit_gib"] + 2.69 + 0.5 <= 76.9
    monkeypatch.delenv("NAMGPU_CARD_SHARE_GIB")
    env, rec = {}, {}
    door._pool_cap(env, rec, pick, 94.42)                     # no suite share: the door's rule as before
    assert rec["init_pool_limit_gib"] == round((94.42 - 1.0), 2)
