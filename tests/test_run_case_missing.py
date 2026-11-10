"""`woof-nam run --case X` with no case folder X stops in one line that names the fetch, before any card or init
work (it stopped in the init with a 35-line traceback). CPU only, no card."""
import pytest

from namgpu import run_forecast


def test_missing_case_folder_stops_in_one_line(tmp_path, monkeypatch):
    monkeypatch.setenv("NAMGPU_ROOT", str(tmp_path))
    monkeypatch.delenv("NAMGPU_RANK", raising=False)
    monkeypatch.setattr(run_forecast, "memory_check", lambda a: pytest.fail("reached the memory check"))
    with pytest.raises(SystemExit) as e:
        run_forecast.main(["--case", "2026-01-23_12z", "--source", "gfs", "--hours", "1", "--out", str(tmp_path / "r")])
    msg = str(e.value)
    assert msg.startswith(f"woof-nam run: no case folder {tmp_path / 'cases' / '2026-01-23_12z'}; fetch it first:"), msg
    assert "woof-nam fetch --cycle 2026012312 --hours 1 --case-dir " in msg and msg.endswith(" --source gfs"), msg
    assert "\n" not in msg
    assert not (tmp_path / "r").exists()


def test_an_existing_case_folder_goes_on(tmp_path, monkeypatch):
    monkeypatch.setenv("NAMGPU_ROOT", str(tmp_path))
    monkeypatch.delenv("NAMGPU_RANK", raising=False)
    (tmp_path / "cases" / "2026-01-23_12z").mkdir(parents=True)

    class Reached(Exception):
        pass

    def stop(a):
        raise Reached
    monkeypatch.setattr(run_forecast, "memory_check", stop)
    with pytest.raises(Reached):
        run_forecast.main(["--case", "2026-01-23_12z", "--source", "gfs", "--hours", "1", "--out", str(tmp_path / "r")])

def test_part_hours_ask_for_whole_fetch_hours(tmp_path, monkeypatch):
    monkeypatch.setenv("NAMGPU_ROOT", str(tmp_path))
    monkeypatch.delenv("NAMGPU_RANK", raising=False)
    monkeypatch.setattr(run_forecast, "memory_check", lambda a: pytest.fail("reached the memory check"))
    with pytest.raises(SystemExit) as e:
        run_forecast.main(["--case", "2026-01-23_12z", "--hours", "2.5", "--out", str(tmp_path / "r")])
    assert " --hours 3 --case-dir " in str(e.value) and not str(e.value).endswith("--source native"), str(e.value)
