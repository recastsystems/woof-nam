"""Memory path selection (CPU only): the 32 GB fit is the default below 64 GiB, selectable on any card."""
import pytest

from namgpu import state


@pytest.mark.parametrize("want,card,path,how", [
    ("", 31.4, "fit", "auto"),          # RTX 5090
    ("auto", 47.4, "fit", "auto"),      # 48 GB cards
    ("", 63.9, "fit", "auto"),
    ("", 64.0, "full", "auto"),
    ("auto", 94.9, "full", "auto"),     # 96 GB cards
    ("fit", 94.9, "fit", "set"),
    ("FULL", 31.4, "full", "set"),
])
def test_resolve(want, card, path, how):
    r = state.resolve_memory_path(want, card)
    assert (r["path"], r["how"]) == (path, how)
    assert r["reason"]


def test_bad_setting():
    with pytest.raises(ValueError):
        state.resolve_memory_path("small", 31.4)


def test_scratch_sets_are_not_state_fields():
    # released arrays must be step locals, never fields of the NAM state tables
    specs = set(state.all_specs())
    released = set(state.STEP_SCRATCH) | set(state.HISTORY_SCRATCH) | set(state.PHYS_SCRATCH)
    for v in state.UNIT_SCRATCH.values():
        released |= set(v)
    assert not (released & specs), sorted(released & specs)


def test_full_run_need_is_the_measured_figure():
    from namgpu.nest import domain
    assert 75.0 < domain.full_run_gib({"im": 954, "jm": 835}) < 77.0          # 12 km parent: fits a 96 GB card
    assert domain.full_run_gib({"im": 1506, "jm": 1318}) > 180.0              # 7.6 km North America: does not
    assert domain.full_run_gib({"im": 1506, "jm": 1318}, 4) < 50.0


@pytest.mark.parametrize("im,jm,free,want,expect", [
    (1506, 1318, 94.5, "", "fit"),      # the 7.6 km grid on a free 96 GB card: OutOfMemoryError at 99 GB before this
    (954, 835, 94.5, "", None),         # the 12 km parent on a free 96 GB card keeps the full path
    (954, 835, 60.0, "", "fit"),        # the same card with a co-tenant: the full path would not fit now
    (1506, 1318, 94.5, "full", "full"),  # a path the caller set is never changed
])
def test_run_door_takes_fit_when_the_full_path_cannot_fit(monkeypatch, im, jm, free, want, expect):
    import types
    from namgpu import config, domains, run_forecast
    from namgpu.nest import domain
    for name in ("NAMGPU_PREFLIGHT_DONE", "NAMGPU_PREFLIGHT", "NAMGPU_RRTM_ROWS", "NAMGPU_SCRATCH_ARENA",
                 "NAMGPU_MEMORY_PATH_WHY"):
        monkeypatch.setenv(name, "")          # empty reads as unset; whatever the door sets is undone after the test
    monkeypatch.setenv("NAMGPU_MEMORY_PATH", want)
    monkeypatch.setattr(config, "load", lambda *a, **k: types.SimpleNamespace(im=im, jm=jm))
    monkeypatch.setattr(domains, "current", lambda: {})
    monkeypatch.setattr(domain, "_visible_index", lambda: "0")
    monkeypatch.setattr(domain, "free_gib", lambda c=None: free)
    monkeypatch.setattr(domain, "card_total_gib", lambda c=None: 95.1)
    monkeypatch.setattr(domain, "card_name", lambda c=None: "test card")
    monkeypatch.setattr(domain, "sm_threads", lambda c=None: 188 * 2048)
    assert run_forecast.memory_check(types.SimpleNamespace(cards=1, card_list="")) == 0
    import os
    got = os.environ.get("NAMGPU_MEMORY_PATH") or None
    assert got == expect
    if expect == "fit" and not want:
        r = state.resolve_memory_path("fit", 95.1)
        assert r["path"] == "fit" and r["how"] == "auto" and "full path needs about" in r["reason"]
