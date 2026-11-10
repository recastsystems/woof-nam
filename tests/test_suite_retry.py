"""woof-nam suite: a nest on the parent's card that ran out of memory before its first forecast hour runs once more
after the parent (suite proof 2026-11-06: the Alaska nest stopped at init-gfs-bocos beside the running 12 km parent)."""
from namgpu.suite import cli


def _suite():
    return cli.Suite.__new__(cli.Suite)


def test_an_init_out_of_memory_is_seen_in_the_last_attempt_only(tmp_path):
    log = tmp_path / "alaskanest-nest.log"
    log.write_text("# t0 woof-nam nest\ninit-case-driver\nRuntimeError: init-case-driver: stage init-gfs-bocos failed: "
                   "OutOfMemoryError: Out of memory allocating 356,414,976 bytes\n")
    assert _suite()._init_out_of_memory(log)
    log.write_text("# t0\nOutOfMemoryError\n# t1 second attempt\nfh   1.00  step 576\n")
    assert not _suite()._init_out_of_memory(log)


def test_an_out_of_memory_after_the_forecast_started_is_not_retried(tmp_path):
    log = tmp_path / "x.log"
    log.write_text("# t0\nfh   1.00  step 576\ncupy.cuda.memory.OutOfMemoryError: Out of memory\n")
    assert not _suite()._init_out_of_memory(log)


def test_a_retry_waits_for_every_domain_on_its_card(monkeypatch):
    # release check 2026-11-09: on one card the CONUS nest's second attempt started beside the running Alaska nest and
    # ran out of memory again; it now waits for every card mate
    import threading
    from types import SimpleNamespace
    s = _suite()
    s.lock = threading.Lock()
    s.t = {"parent": {"domain": "12km"}}
    s.a = SimpleNamespace(parent_run=None)
    s.plan = {"domains": {"12km": {"cards": ["0"]}, "conusnest": {"cards": ["0"]}, "alaskanest": {"cards": ["0"]},
                          "hawaiinest": {"cards": ["1"]}}}
    s.rec = {"domains": {"alaskanest": {"state": "forecast"}, "12km": {"state": "complete"}}}
    monkeypatch.setattr(cli.Suite, "parent_hours_written", lambda self: (6, True))
    assert s._card_mates("conusnest") == ["12km", "alaskanest"]
    assert s._busy_mates("conusnest") == ["alaskanest"]
    s.rec["domains"]["alaskanest"]["state"] = "complete"
    assert s._busy_mates("conusnest") == []
    assert s._card_mates("hawaiinest") == []
    s.a = SimpleNamespace(parent_run="given")       # a parent run from elsewhere is not on this suite's cards
    assert s._card_mates("conusnest") == ["alaskanest"]
