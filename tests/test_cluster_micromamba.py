"""Mocked tests for the native Micromamba SLURM execution path."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


REPO = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")
pytestmark = pytest.mark.skipif(BASH is None or os.name == "nt",
                                reason="Micromamba shell tests require a POSIX bash")


def _fake_micromamba(tmp_path: Path) -> tuple[Path, Path]:
    executable = tmp_path / "micromamba"
    log = tmp_path / "micromamba.args"
    executable.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$@" >> "$MM_LOG"
printf 'PYTHONPATH=%s OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s\\n' \
  "${PYTHONPATH:-}" "${OMP_NUM_THREADS:-}" "${OPENBLAS_NUM_THREADS:-}" >> "${MM_ENV_LOG:-/dev/null}"
if [[ "${MM_FAIL:-0}" == 1 ]]; then echo fake-solver-failure >&2; exit 17; fi
if [[ "${MM_EXEC:-0}" != 1 ]]; then echo fake-micromamba-python; exit 0; fi
[[ "${1:-}" == run ]] || { echo 'missing run' >&2; exit 18; }
shift
while (($#)); do
  case "$1" in
    --root-prefix|--prefix) shift 2 ;;
    *) break ;;
  esac
done
shift
export IN_MICROMAMBA=1
exec "$FAKE_PYTHON" "$@"
"""
    )
    executable.chmod(0o755)
    return executable, log


def _config(tmp_path: Path, *, executable: Path, root: Path, env: Path,
            run: Path, log: Path, manifest: Path) -> Path:
    config = tmp_path / "cluster.env"
    config.write_text(
        f"""CLUSTER_EXECUTION_MODE=micromamba
PROJECT_ROOT={shlex_quote(str(REPO))}
CLUSTER_PROJECT_PARENT={shlex_quote(str(tmp_path))}
RUN_ROOT={shlex_quote(str(run))}
LOG_ROOT={shlex_quote(str(log))}
MANIFEST_ROOT={shlex_quote(str(manifest))}
SLURM_CPUS=1
SLURM_ARRAY_CONCURRENCY=2
MICROMAMBA_EXE={shlex_quote(str(executable))}
MICROMAMBA_ROOT_PREFIX={shlex_quote(str(root))}
MICROMAMBA_ENV_PREFIX={shlex_quote(str(env))}
MICROMAMBA_PYTHON=python
"""
    )
    return config


def _fake_environment_git(tmp_path: Path) -> Path:
    git = tmp_path / "git"
    git.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
[[ "${IN_MICROMAMBA:-0}" == 1 ]] || exit 127
if [[ "${1:-}" == --version ]]; then
  echo 'git version fake-cfd-env'
  exit 0
fi
[[ "${1:-}" == -C && "${3:-}" == rev-parse ]] || exit 2
echo fake-commit
"""
    )
    git.chmod(0o755)
    return git


def shlex_quote(value: str) -> str:
    # These tests run under bash while paths may contain spaces.
    import shlex
    return shlex.quote(value)


def _run(script: str, config: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    child_env = os.environ.copy()
    child_env.update(env)
    child_env["CLUSTER_CONFIG"] = str(config)
    return subprocess.run([BASH, "-c", script], cwd=REPO, env=child_env,
                          text=True, capture_output=True)


def test_micromamba_command_uses_configured_prefixes(tmp_path):
    exe, log = _fake_micromamba(tmp_path)
    root, env = tmp_path / "root", tmp_path / "cfd-env"
    root.mkdir()
    env.mkdir()
    config = _config(tmp_path, executable=exe, root=root, env=env,
                     run=tmp_path / "runs", log=tmp_path / "logs",
                     manifest=tmp_path / "manifests")
    result = _run(
        "source scripts/cluster-common.sh; cluster_python -c ignored",
        config, {"MM_LOG": str(log)},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "fake-micromamba-python"
    args = log.read_text().splitlines()
    assert args[:6] == ["run", "--root-prefix", str(root), "--prefix", str(env), "python"]
    assert "create" not in args and "install" not in args and "activate" not in args


def test_micromamba_failure_is_propagated_without_touching_env(tmp_path):
    exe, log = _fake_micromamba(tmp_path)
    root, env = tmp_path / "root", tmp_path / "cfd-env"
    root.mkdir()
    env.mkdir()
    marker = env / "keep.txt"
    marker.write_text("pre-provisioned")
    config = _config(tmp_path, executable=exe, root=root, env=env,
                     run=tmp_path / "runs", log=tmp_path / "logs",
                     manifest=tmp_path / "manifests")
    result = _run("source scripts/cluster-common.sh; cluster_prepare", config,
                  {"MM_LOG": str(log), "MM_FAIL": "1"})
    assert result.returncode == 17
    assert "fake-solver-failure" in result.stderr
    assert marker.read_text() == "pre-provisioned"
    assert not (env / "conda-meta").exists()


def test_prepare_sets_project_path_and_single_native_threads(tmp_path):
    exe, log = _fake_micromamba(tmp_path)
    root, env = tmp_path / "root", tmp_path / "cfd-env"
    root.mkdir()
    env.mkdir()
    env_log = tmp_path / "environment.log"
    config = _config(tmp_path, executable=exe, root=root, env=env,
                     run=tmp_path / "runs", log=tmp_path / "logs",
                     manifest=tmp_path / "manifests")
    result = _run("source scripts/cluster-common.sh; cluster_prepare", config,
                  {"MM_LOG": str(log), "MM_ENV_LOG": str(env_log)})
    assert result.returncode == 0, result.stderr
    assert any(line.startswith(f"PYTHONPATH={REPO}") and
               "OMP_NUM_THREADS=1" in line and
               "OPENBLAS_NUM_THREADS=1" in line
               for line in env_log.read_text().splitlines())


def test_git_probe_runs_inside_micromamba_when_host_git_fails(tmp_path):
    exe, log = _fake_micromamba(tmp_path)
    host_path = tmp_path / "host-path"
    host_path.mkdir()
    fake_git = _fake_environment_git(host_path)
    root, env = tmp_path / "root", tmp_path / "cfd-env"
    root.mkdir()
    env.mkdir()
    config = _config(tmp_path, executable=exe, root=root, env=env,
                     run=tmp_path / "runs", log=tmp_path / "logs",
                     manifest=tmp_path / "manifests")
    path = f"{host_path}:/usr/bin:/bin"
    common_env = {"MM_LOG": str(log), "MM_EXEC": "1", "FAKE_PYTHON": sys.executable,
                  "PATH": path}
    host_probe = _run("git -C . rev-parse --verify HEAD", config, common_env)
    assert host_probe.returncode != 0
    result = _run("source scripts/cluster-common.sh; cluster_prepare", config, common_env)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("mode", ["modules", "apptainer"])
def test_non_micromamba_defaults_remain_project_relative(tmp_path, mode):
    config = tmp_path / "cluster.env"
    config.write_text(f"CLUSTER_EXECUTION_MODE={mode}\nPROJECT_ROOT={shlex_quote(str(REPO))}\n")
    result = _run("source scripts/cluster-common.sh; printf '%s\\n' \"$RUN_ROOT\" \"$LOG_ROOT\"",
                  config, {})
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(REPO / "cluster-runs"),
                                          str(REPO / "cluster-logs")]


def test_sr_pilot_submission_path_is_dry_run_and_bounded(tmp_path):
    exe, log = _fake_micromamba(tmp_path)
    root, env = tmp_path / "root", tmp_path / "cfd-env"
    root.mkdir()
    env.mkdir()
    config = _config(tmp_path, executable=exe, root=root, env=env,
                     run=tmp_path / "runs", log=tmp_path / "logs",
                     manifest=tmp_path / "manifests")
    result = _run(
        "bash scripts/slurm/submit-array.sh sr_pilot",
        config,
        {"MM_LOG": str(log), "MM_EXEC": "1", "FAKE_PYTHON": sys.executable,
         "SLURM_DRY_RUN": "1"},
    )
    assert result.returncode == 0, result.stderr
    assert "--array=0-2%2" in result.stdout
    assert "--cpus-per-task=1" in result.stdout
    assert (tmp_path / "manifests" / "sr_pilot.json").exists()
