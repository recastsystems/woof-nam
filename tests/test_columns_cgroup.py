"""woof-nam columns: the worker count follows the room under the cgroup's memory.max, not only the host's
MemAvailable. Breakage it prevents: in a login session capped at 80 GiB on a 123 GB host the store started workers
for the host's free memory and the kernel killed two at the cap (BrokenProcessPool; arctic-front 0.2.1 candidate)."""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from namgpu.io import columns  # noqa: E402

GB = 1024 ** 3


def _tree(t: Path, leaf_max: str, parent_max: str = "max"):
    root = t / "cg"
    leaf = root / "user.slice" / "session-1.scope"
    leaf.mkdir(parents=True)
    (t / "cgroup").write_text("0::/user.slice/session-1.scope\n")
    (leaf / "memory.max").write_text(leaf_max + "\n")
    (leaf / "memory.current").write_text(str(70 * GB) + "\n")
    (leaf / "memory.stat").write_text(f"anon {40 * GB}\nfile {30 * GB}\nshmem {20 * GB}\n")
    (root / "user.slice" / "memory.max").write_text(parent_max + "\n")
    (root / "user.slice" / "memory.current").write_text(str(70 * GB) + "\n")
    (root / "user.slice" / "memory.stat").write_text(f"file {30 * GB}\nshmem {20 * GB}\n")
    return str(t / "cgroup"), str(root)


def test_headroom_counts_shm_and_frees_cache():
    with tempfile.TemporaryDirectory() as d:
        pc, root = _tree(Path(d), str(80 * GB))
        # used = current 70 - reclaimable cache (file 30 - shmem 20 = 10) = 60; room = 80 - 60 = 20 GiB
        assert columns._cgroup_headroom(pc, root) == 20 * GB


def test_smallest_limit_up_the_tree_wins():
    with tempfile.TemporaryDirectory() as d:
        pc, root = _tree(Path(d), str(80 * GB), parent_max=str(65 * GB))
        assert columns._cgroup_headroom(pc, root) == 5 * GB


def test_no_limit_is_none():
    with tempfile.TemporaryDirectory() as d:
        pc, root = _tree(Path(d), "max")
        assert columns._cgroup_headroom(pc, root) is None


def test_workers_follow_the_smaller_room():
    with tempfile.TemporaryDirectory() as d:
        frames = []
        for i in range(37):
            f = Path(d) / f"wrfout_d01_{i:02d}"
            with open(f, "wb") as fh:
                fh.truncate(1_238_400_334)        # a 12 km history file
            frames.append(f)
        # no scratch folder: a worker counts one view of memory (VIEW_RATIO x the file, the header being unreadable);
        # a tmpfs scratch doubles it (tests/test_shm_leftovers.py)
        host = columns.auto_workers(frames, 24, None, mem_available=67 * GB, scratch_free=10 ** 13)
        capped = columns.auto_workers(frames, 24, None, mem_available=20 * GB, scratch_free=10 ** 13)
        assert host == 10 and capped == 3, (host, capped)


if __name__ == "__main__":
    test_headroom_counts_shm_and_frees_cache()
    test_smallest_limit_up_the_tree_wins()
    test_no_limit_is_none()
    test_workers_follow_the_smaller_room()
    print("test_columns_cgroup: 4 passed")
