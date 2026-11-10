"""One frozen NAM case, winter/CAD sampling, and option-off forecast identity.

Run outside a GPU lease. This controller builds through the supplied build helper,
then takes one pool lease for the two forecasts and releases it before scoring.
Inputs are existing files named and hashed in a JSON manifest. No downloading,
physics overrides, data deletion, or successful forecast replay is implicit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

BASE = "1ae1f668f2a432f607e4d2d143b789e15ca8a743"
CASE = "2026-01-23_12z"
CYCLE = "2026012312"
HOURS = 3
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(tree, *args):
    return subprocess.check_output(["git", "-C", str(tree), *args], text=True).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--build-helper", type=Path, required=True)
    ap.add_argument("--pool-helper", type=Path, required=True)
    ap.add_argument("--bridges", type=Path, required=True)
    ap.add_argument("--phase", choices=("all", "forecasts", "score"), default="all")
    a = ap.parse_args()
    for name in ("baseline", "inputs", "work", "build_helper", "pool_helper", "bridges"):
        setattr(a, name, getattr(a, name).resolve())
    evidence = a.work / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    receipt = evidence / "proof.json"
    if receipt.exists():
        if read(receipt).get("passed"):
            print(f"REUSE completed proving test: {receipt}")
            return 0
        raise RuntimeError(f"incomplete proof receipt already exists: {receipt}")
    if git(a.baseline, "rev-parse", "HEAD") != BASE:
        raise RuntimeError("baseline is not the frozen pre-scorer commit")
    for tree in (ROOT, a.baseline):
        if git(tree, "status", "--porcelain", "--untracked-files=no"):
            raise RuntimeError(f"tracked source changes in {tree}")
    # The scoring feature must not change the forecast implementation or tables.
    git(ROOT, "diff", "--exit-code", BASE, "HEAD", "--", "namgpu",
        ":!namgpu/io/namscore.py", ":!namgpu/io/namwinter.py", ":!namgpu/io/winter_bridge.rs")
    env = {k: v for k, v in os.environ.items() if not k.startswith("NAMGPU_")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT),
               NAMGPU_ROOT=str(a.work), NAMGPU_MEMORY_PATH="fit",
               CUPY_CACHE_DIR=str(a.work / "cache/cupy"),
               CUDA_CACHE_PATH=str(a.work / "cache/cuda"),
               XDG_CACHE_HOME=str(a.work / "cache"), TMPDIR=str(a.work / "scratch"),
               GPUWM_BRIDGES=str(a.bridges), OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", RAYON_NUM_THREADS="4",
               GPU_MUTEX_POOL="0")
    for rel in ("cache/cupy", "cache/cuda", "scratch", "runs", "static"):
        (a.work / rel).mkdir(parents=True, exist_ok=True)
    commands = evidence / "commands.jsonl"

    def run(label, command, local_env=env):
        start = time.time()
        print(f"{label}: {command}", flush=True)
        with (evidence / (label + ".log")).open("w") as stream:
            result = subprocess.run([str(x) for x in command], env=local_env,
                                    cwd=a.work, stdout=stream, stderr=subprocess.STDOUT)
        record = dict(label=label, command=[str(x) for x in command], cwd=str(a.work),
                      start_unix=start, seconds=time.time() - start, rc=result.returncode,
                      pythonpath=local_env["PYTHONPATH"])
        with commands.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if result.returncode:
            raise RuntimeError(f"{label} failed rc={result.returncode}; see {evidence / (label + '.log')}")

    if a.phase == "all":
        if os.environ.get("GPU_MUTEX_CARDS"):
            raise RuntimeError("run the controller outside a GPU lease")
        manifest = read(a.inputs)
        for item in manifest["files"]:
            source = Path(item["source"])
            if source.stat().st_size != item["bytes"] or sha(source) != item["sha256"]:
                raise RuntimeError(f"input checksum mismatch: {source}")
            relative = Path(item["target"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("input target must be a relative path inside work")
            target = a.work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise RuntimeError(f"different input already at {target}")
            elif target.exists():
                raise RuntimeError(f"input target already exists: {target}")
            else:
                target.symlink_to(source)
        shutil.copyfile(a.inputs, evidence / "inputs.json")
        for label, tree in (("baseline", a.baseline), ("candidate", ROOT)):
            run("warm-" + label, [a.build_helper, sys.executable, "-m", "namgpu.kernels", "warm", "--workers", "4"],
                dict(env, PYTHONPATH=str(tree)))
            warm = json.loads((evidence / ("warm-" + label + ".log")).read_text().splitlines()[-1])
            if warm["failed"]:
                raise RuntimeError(f"kernel build failures in {label}: {warm['failed']}")
        run("build-reader", [a.build_helper, "rustc", "--edition=2021", "-O",
                             ROOT / "namgpu/io/winter_bridge.rs", "-o", a.work / "rw_namwinter"])
        command = [sys.executable, Path(__file__).resolve(), "--phase", "forecasts"]
        for flag in ("baseline", "inputs", "work", "build-helper", "pool-helper", "bridges"):
            command += ["--" + flag, str(getattr(a, flag.replace("-", "_")))]
        run("pool", [a.pool_helper, "--cards", "1", "--vram-mib", "28000", "--est", "1200",
                     "bl-nam-wintercad-proof", "--", *command])

    if a.phase == "forecasts":
        if not os.environ.get("GPU_MUTEX_CARDS"):
            raise RuntimeError("forecasts require the pool helper lease")
        for label, tree in (("nam-control", a.baseline), ("candidate-option-off", ROOT)):
            out = a.work / "runs" / label
            archived = evidence / label / "forecast-receipt.json"
            if archived.exists():
                raise RuntimeError(f"completed forecast already archived; do not replay: {archived}")
            if out.exists() and any(out.glob("wrfout_d01_*")):
                raise RuntimeError(f"unarchived forecast frames already exist: {out}")
            local_env = dict(env, PYTHONPATH=str(tree))
            # The helper intentionally passes GPU runtime through without a build-slot wait.
            run(label, [a.build_helper, sys.executable, "-m", "namgpu.cli", "run",
                        "--case", CASE, "--source", "gfs", "--physics-suite", "nam", "--hours", str(HOURS),
                        "--no-maps", "--no-columns", "--history-fields", "full", "--full-every", "0",
                        "--out", str(out)], local_env)
            forecast = read(out / "forecast-receipt.json")
            if forecast.get("error") or forecast.get("nonfinite_end") or forecast.get("left_out"):
                raise RuntimeError(f"forecast diagnostics failed: {label}")
            if forecast["hours"] != HOURS or forecast["grid"] != [954, 835, 60]:
                raise RuntimeError(f"wrong forecast configuration: {label}")
            if len(list(out.glob("wrfout_d01_*"))) != HOURS + 1:
                raise RuntimeError(f"wrong hourly frame count: {label}")
            archived.parent.mkdir(exist_ok=True)
            for path in out.glob("*.json"):
                shutil.copyfile(path, archived.parent / path.name)
            save(archived.parent / "source.json", dict(commit=git(tree, "rev-parse", "HEAD"), tree=str(tree)))
        return 0

    env["NAMGPU_WINTER_BRIDGE"] = str(a.work / "rw_namwinter")
    control, candidate = (a.work / "runs" / x for x in ("nam-control", "candidate-option-off"))
    for path in (control, candidate):
        if len(list(path.glob("wrfout_d01_*"))) != HOURS + 1:
            raise RuntimeError(f"missing forecast frames: {path}")
    run("identity", [sys.executable, "-m", "namgpu.cli", "compare", control, candidate,
                     "--workers", "2", "--json", evidence / "option-off-identity.json"])
    identity = read(evidence / "option-off-identity.json")
    if not identity["bitwise"] or identity["words_diff"] != 0 or identity["hours_a"] != HOURS + 1:
        raise RuntimeError("option-off forecast identity failed")
    before = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in control.glob("wrfout_d01_*")}
    save(evidence / "control-history.json", before)
    run("winter-score", [sys.executable, "-m", "namgpu.cli", "score", "--winter", "--cycle", CYCLE,
                         "--hours", str(HOURS), "--run", f"nam-control={control}",
                         "--asos", a.work / "asos.csv", "--workers", "2", "--out", evidence / "winter"])
    score = read(evidence / "winter/winter_scores.json")["runs"]["nam-control"]
    if not score["coverage"]["erosion_stations"]:
        raise RuntimeError("no complete CAD panel")
    if any(score["hourly"][str(h)][v]["n"] <= 0 for h in range(HOURS + 1) for v in ("t2_f", "td2_f")):
        raise RuntimeError("empty hourly CAD temperature or dewpoint score")
    after = {p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in control.glob("wrfout_d01_*")}
    if after != before:
        raise RuntimeError("scoring changed a forecast history file")
    save(receipt, dict(schema="woof-nam.wintercad-proof.v1", passed=True, cycle=CYCLE, hours=HOURS,
                       baseline_commit=BASE, candidate_commit=git(ROOT, "rev-parse", "HEAD"),
                       identity=identity, scorer_input_unchanged=True, score_coverage=score["coverage"],
                       score_gaps=score["gaps"], input_manifest_sha256=sha(a.inputs),
                       script_sha256=sha(__file__), reader_sha256=sha(a.work / "rw_namwinter")))
    print(f"PASS: {receipt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
