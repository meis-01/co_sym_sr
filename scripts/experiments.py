"""Deterministic manifests and safe, validated execution of CFD experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cardiac_flow.model import load  # noqa: E402


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_info(repo: Path) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def build_manifest(study: str | None, config: Path | None, manifest_path: Path,
                   output_root: Path, repo: Path) -> dict:
    if (study is None) == (config is None):
        raise ValueError("provide exactly one of --study or --config")
    if study:
        with tempfile.TemporaryDirectory(prefix="cfd-manifest-") as temporary:
            subprocess.run([sys.executable, str(REPO / "scripts" / "study.py"), study,
                            "--root", temporary], cwd=repo, check=True)
            generated = Path(temporary) / study
            cases = [(p.stem, json.loads(p.read_text(encoding="utf-8")))
                     for p in sorted(generated.glob("*.json"))]
        kind = study
    else:
        config = config.resolve()
        cases = [(config.stem, json.loads(config.read_text(encoding="utf-8")))]
        kind = "single"

    config_dir = manifest_path.parent / f"{manifest_path.stem}.configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (case_id, parameters) in enumerate(cases):
        config_file = config_dir / f"{case_id}.json"
        config_file.write_text(json.dumps(parameters, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        checked = load(config_file)
        config_bytes = config_file.read_bytes()
        entries.append({
            "index": index,
            "id": case_id,
            "config": os.path.relpath(config_file, manifest_path.parent),
            "config_sha256": sha256(config_bytes),
            "output": str((output_root / case_id).resolve()),
            "parameters": checked,
        })
    result = {
        "schema": "cardiac-flow-experiment-manifest-1",
        "study": kind,
        "git": git_info(repo),
        "repo": ".",
        "output_root": str(output_root.resolve()),
        "cases": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def load_manifest(path: Path) -> tuple[dict, Path]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "cardiac-flow-experiment-manifest-1":
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')}")
    return manifest, REPO


def case_status(output: Path) -> str:
    fields = output / "fields.h5"
    validation_path = output / "validation.json"
    if fields.is_file() and validation_path.is_file():
        try:
            if json.loads(validation_path.read_text(encoding="utf-8")).get("status") == "passed":
                return "completed"
        except (OSError, ValueError):
            pass
    if (output / "failure.json").is_file():
        return "failed"
    return "incomplete" if output.exists() and any(output.iterdir()) else "not_started"


def execution_environment() -> dict:
    try:
        uv_version = subprocess.check_output(["uv", "--version"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        uv_version = None
    versions = {}
    for package in ("numpy", "scipy", "h5py", "matplotlib", "meshio"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    native = {}
    try:
        import dolfin
        native["dolfin"] = dolfin.__version__
    except ImportError:
        native["dolfin"] = None
    try:
        import gmsh
        native["gmsh"] = getattr(gmsh, "__version__", None)
    except ImportError:
        native["gmsh"] = None
    try:
        from petsc4py import PETSc
        native["petsc"] = PETSc.Sys.getVersion()
    except ImportError:
        native["petsc"] = None
    image = os.environ.get("APPTAINER_IMAGE")
    image_stat = Path(image).stat() if image and Path(image).exists() else None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "execution_mode": os.environ.get("CLUSTER_EXECUTION_MODE", "local"),
        "modules": os.environ.get("CLUSTER_MODULES_LOADED", ""),
        "uv": uv_version,
        "python_packages": versions,
        "native_libraries": native,
        "apptainer_image": image,
        "apptainer_image_bytes": image_stat.st_size if image_stat else None,
        "apptainer_image_mtime_ns": image_stat.st_mtime_ns if image_stat else None,
    }


def run_command(repo: Path, output: Path, *args: str) -> list[str]:
    if os.environ.get("CFD_IN_CONTAINER") == "1":
        return [sys.executable, "-m", "cardiac_flow", *args]
    mode = os.environ.get("CLUSTER_EXECUTION_MODE", "modules")
    if mode == "apptainer":
        image = os.environ.get("APPTAINER_IMAGE")
        container_python = os.environ.get("APPTAINER_PYTHON", "/opt/venv/bin/python")
        if not image:
            raise RuntimeError("APPTAINER_IMAGE must name a compatible CFD image")
        binds = [f"{repo}:{repo}", f"{output.parent}:{output.parent}"]
        bind_arg = ",".join(dict.fromkeys(binds))
        return ["apptainer", "exec", "--bind", bind_arg, "--pwd", str(repo), image,
                container_python, "-m", "cardiac_flow", *args]
    return [sys.executable, "-m", "cardiac_flow", *args]


def run_case(manifest_path: Path, index: int, dry_run: bool = False) -> int:
    manifest, repo = load_manifest(manifest_path.resolve())
    try:
        case = manifest["cases"][index]
    except IndexError as exc:
        raise ValueError(f"case index {index} outside 0..{len(manifest['cases']) - 1}") from exc
    if case["index"] != index:
        raise ValueError("manifest case indices must be contiguous and zero-based")
    config = (manifest_path.resolve().parent / case["config"]).resolve()
    if sha256(config.read_bytes()) != case["config_sha256"]:
        raise ValueError(f"configuration changed since manifest creation: {config}")
    output = Path(case["output"])
    status = case_status(output)
    if dry_run:
        print(json.dumps({"index": index, "id": case["id"], "config": str(config),
                          "output": str(output), "status": status,
                          "parameters": case["parameters"]}, indent=2))
        return 0
    if status == "completed":
        previous = output / "execution.json"
        if previous.is_file():
            metadata = json.loads(previous.read_text(encoding="utf-8"))
            if metadata.get("config_sha256") != case["config_sha256"]:
                raise RuntimeError(f"refusing to replace successful output from another configuration: {output}")
        print(f"[{case['id']}] already validated; skipping {output}", flush=True)
        return 0

    lock_dir = Path(manifest["output_root"]) / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_file = (lock_dir / f"{case['id']}.lock").open("a+")
    try:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"case {case['id']} is already running") from exc
        # Recheck after acquiring the lock in case another array submission finished it.
        status = case_status(output)
        if status == "completed":
            previous = output / "execution.json"
            if previous.is_file():
                metadata = json.loads(previous.read_text(encoding="utf-8"))
                if metadata.get("config_sha256") != case["config_sha256"]:
                    raise RuntimeError(f"refusing to replace successful output from another configuration: {output}")
            print(f"[{case['id']}] already validated; skipping {output}", flush=True)
            return 0
        if output.exists() and any(output.iterdir()):
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            archive = Path(manifest["output_root"]) / ".attempts" / case["id"] / stamp
            archive.parent.mkdir(parents=True, exist_ok=True)
            output.rename(archive)
            print(f"[{case['id']}] archived {status} attempt to {archive}", flush=True)

        output.mkdir(parents=True, exist_ok=True)
        metadata = {
            "status": "running", "case_id": case["id"], "array_index": index,
            "manifest_sha256": sha256(manifest_path.read_bytes()),
            "config_sha256": case["config_sha256"], "git": git_info(repo),
            "environment": execution_environment(), "parameters": case["parameters"],
        }
        metadata_path = output / "execution.json"
        running_path = lock_dir / f"{case['id']}.running.json"
        running_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try:
            subprocess.run(run_command(repo, output, "run", str(config), str(output)), cwd=repo, check=True)
            subprocess.run(run_command(repo, output, "validate", str(output / "fields.h5")),
                           cwd=repo, check=True)
            validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
            if validation.get("status") != "passed":
                raise RuntimeError(f"validator returned unexpected status: {validation}")
            metadata["status"] = "completed"
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["validation"] = validation
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            running_path.unlink(missing_ok=True)
            print(f"[{case['id']}] simulation and validation completed", flush=True)
            return 0
        except Exception as error:
            metadata["status"] = "failed"
            metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
            metadata["error"] = str(error)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            failure_path = output / "failure.json"
            if not failure_path.exists():
                failure_path.write_text(json.dumps({"status": "failed", "reason": str(error)},
                                                   indent=2) + "\n", encoding="utf-8")
            running_path.unlink(missing_ok=True)
            raise
    finally:
        lock_file.close()


def list_status(manifest_path: Path) -> int:
    manifest, repo = load_manifest(manifest_path.resolve())
    counts = {key: 0 for key in ("completed", "failed", "incomplete", "not_started")}
    for case in manifest["cases"]:
        status = case_status(Path(case["output"]))
        counts[status] += 1
        print(f"{case['index']:04d}\t{case['id']}\t{status}\t{case['output']}")
    print("\n" + json.dumps(counts, sort_keys=True))
    return 1 if counts["failed"] or counts["incomplete"] else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("manifest", help="create a deterministic study or single-case manifest")
    source = create.add_mutually_exclusive_group(required=True)
    source.add_argument("--study", choices=("sweep", "convergence", "rotation"))
    source.add_argument("--config", type=Path)
    create.add_argument("--manifest", required=True, type=Path)
    create.add_argument("--output-root", required=True, type=Path)
    create.add_argument("--repo", type=Path, default=REPO)
    case_parser = sub.add_parser("case", help="execute one manifest case")
    case_parser.add_argument("--manifest", required=True, type=Path)
    case_parser.add_argument("--index", required=True, type=int)
    case_parser.add_argument("--dry-run", action="store_true")
    status_parser = sub.add_parser("status", help="classify all manifest cases")
    status_parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "manifest":
        manifest = build_manifest(args.study, args.config, args.manifest.resolve(),
                                  args.output_root, args.repo.resolve())
        print(f"Wrote {len(manifest['cases'])} cases to {args.manifest}")
    elif args.command == "case":
        raise SystemExit(run_case(args.manifest, args.index, args.dry_run))
    else:
        raise SystemExit(list_status(args.manifest))


if __name__ == "__main__":
    main()
