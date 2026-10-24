import json
from pathlib import Path

from scripts.experiments import REPO, build_manifest, case_status


def test_case_status_distinguishes_outcomes(tmp_path):
    output = tmp_path / "ok"
    output.mkdir()
    assert case_status(output) == "not_started"
    (output / "fields.h5").touch()
    assert case_status(output) == "incomplete"
    (output / "validation.json").write_text('{"status":"passed"}')
    assert case_status(output) == "completed"
    (output / "validation.json").write_text('{"status":"failed"}')
    assert case_status(output) == "incomplete"
    (output / "failure.json").write_text('{"status":"failed"}')
    assert case_status(output) == "failed"


def test_single_manifest_maps_config_and_output_deterministically(tmp_path):
    source = REPO / "configs" / "cluster-smoke.json"
    manifest_path = tmp_path / "manifests" / "smoke.json"
    output_root = tmp_path / "runs"
    manifest = build_manifest(None, source, manifest_path, output_root, REPO)

    assert manifest["schema"] == "cardiac-flow-experiment-manifest-1"
    assert len(manifest["cases"]) == 1
    case = manifest["cases"][0]
    assert case["index"] == 0
    assert case["id"] == "cluster-smoke"
    assert Path(case["output"]) == output_root / "cluster-smoke"
    config_copy = (manifest_path.parent / case["config"]).resolve()
    assert json.loads(config_copy.read_text()) == case["parameters"]
    assert len(case["config_sha256"]) == 64
    assert build_manifest(None, source, manifest_path, output_root, REPO) == manifest


def test_study_manifest_has_stable_array_indices(tmp_path):
    manifest = build_manifest("rotation", None, tmp_path / "rotation.json", tmp_path / "runs", REPO)
    assert [(case["index"], case["id"]) for case in manifest["cases"]] == [
        (0, "orientation_0"), (1, "orientation_90")]

