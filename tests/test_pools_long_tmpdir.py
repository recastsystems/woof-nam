"""The products field pool, the column pool and the history writer pool start under a long TMPDIR.

Breakage it prevents: multiprocessing puts the forkserver's listening socket at $TMPDIR/pymp-XXXXXXXX/listener-XXXXXXXX,
and an AF_UNIX socket path holds at most 107 bytes on Linux. With a TMPDIR of 115 characters, on Python 3.12, the
products and column pools failed at their first task with "OSError: AF_UNIX path too long" (2026-11-12) and a run
with --products lost its GRIB2 hours. CPU only. Each pool runs one task in its own fresh interpreter: the forkserver
is one per process, so a pool started after another would reuse the first one's socket and prove nothing."""
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

CHILD = textwrap.dedent("""
    import os, sys, tempfile
    from concurrent.futures import ProcessPoolExecutor
    sys.path.insert(0, sys.argv[1])
    from multiprocessing import util
    from namgpu.io import columns, io_history, products

    def main(which):
        assert len(tempfile.gettempdir()) > 100, tempfile.gettempdir()
        if which == "history":
            try:
                assert io_history._pool().submit(os.getpid).result(timeout=120) != os.getpid()
            finally:
                io_history.shutdown()
        else:
            ctx = products._field_context() if which == "products" else columns._pool_context()
            with ProcessPoolExecutor(1, mp_context=ctx) as ex:
                assert ex.submit(os.getpid).result(timeout=120) != os.getpid()
        # the user's TMPDIR stays the temporary folder for everything else
        assert len(tempfile.gettempdir()) > 100, tempfile.gettempdir()
        print("ok", which, util.get_temp_dir(), flush=True)

    if __name__ == "__main__":
        main(sys.argv[2])
""")


@pytest.mark.skipif(os.name == "nt", reason="AF_UNIX forkserver sockets are a POSIX matter")
@pytest.mark.parametrize("which", ["products", "columns", "history"])
def test_pool_starts_under_a_long_tmpdir(tmp_path, which):
    long_tmp = tmp_path / ("t" * 100)
    long_tmp.mkdir()
    script = tmp_path / "pools_child.py"
    script.write_text(CHILD)
    env = dict(os.environ, TMPDIR=str(long_tmp))
    r = subprocess.run([sys.executable, str(script), str(ROOT), which], env=env, capture_output=True, text=True,
                       timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "AF_UNIX" not in r.stderr, r.stderr
    line = [x for x in r.stdout.splitlines() if x.startswith("ok ")]
    assert line, r.stdout + r.stderr
    used = Path(line[-1].split(" ", 2)[2])
    assert len(str(used)) + len("/listener-XXXXXXXX") <= 104, used        # room for the socket name
    assert not used.exists(), f"the socket folder {used} was left behind"


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-q", "-p", "no:cacheprovider", __file__]))
