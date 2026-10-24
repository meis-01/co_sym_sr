"""Campaign, acceptance gates and solver-free SR export regression tests."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import h5py
import numpy as np
import pytest

from cardiac_flow.model import dimensionless
from cardiac_flow.sr_campaign import configurations, specification, group_id
from cardiac_flow.sr_accuracy import assess, convergence, _local_divergence
from cardiac_flow.sr_data import (sample_mesh, sample_field, export_dataset,
                                 iter_simulations, iter_batches, file_hash)
from cardiac_flow.storage import quadratic_grid

REPO = Path(__file__).resolve().parents[1]


def test_campaign_parameters_and_configuration_splits():
    cases = configurations('sr_sweep')
    assert len(cases) == 36
    assert cases == configurations('sr_sweep')
    splits = {name: set() for name in ('train', 'validation', 'test')}
    combinations = set()
    for name, c in cases.items():
        d = dimensionless(c)
        key = (round(d['Re']), round(d['alpha']), c['eccentricity'])
        assert np.isclose(d['Re'], c['_sr']['reynolds'])
        assert np.isclose(d['alpha'], c['_sr']['alpha'])
        assert name == group_id(*key)
        combinations.add(key)
        splits[c['_sr']['split']].add(key)
        assert (c['geometry'], c['expansion'], c['radius'], c['length']) == ('straight', 0, .012, .06)
        assert (c['rho'], c['mu'], c['orientation'], c['inlet']) == (1060, .0035, 0, 'parabolic')
        assert c['harmonics'] == [[.5, 0], [.15, -.5]]
        assert c['_sr']['numerics_status'] == 'candidate'
    assert len(combinations) == 36
    assert [len(splits[k]) for k in ('train', 'validation', 'test')] == [24, 6, 6]
    assert not (splits['train'] & splits['validation'] or splits['train'] & splits['test']
                or splits['validation'] & splits['test'])
    for split in splits.values():
        assert {p[0] for p in split} == {60, 150, 300}
        assert {p[1] for p in split} == {12, 16, 20}
        assert {p[2] for p in split} == {0, .1, .2, .3}


def test_pilot_and_accuracy_design():
    pilot, accuracy = configurations('sr_pilot'), configurations('sr_accuracy')
    assert len(pilot) == 3 and len(accuracy) == 32
    assert {c['eccentricity'] for c in pilot.values()} == {0, .1, .3}
    high = group_id(300, 20, .3)
    assert {accuracy[f'{high}__h{h}_t2']['mesh_size'] for h in range(3)} == {.0045, .003, .002}
    assert {accuracy[f'{high}__h2_t{t}']['steps_per_cycle'] for t in range(3)} == {100, 200, 400}
    assert accuracy[f'{high}__periodic_tight']['periodic_tolerance'] < accuracy[f'{high}__h2_t2']['periodic_tolerance']
    assert np.isclose(accuracy[f'{high}__rot0_h2']['orientation'], np.pi/4)
    assert all(c['_sr']['split'] == 'accuracy_only' for c in accuracy.values())


def test_gci_detects_convergence_and_rejects_bad_evidence():
    criteria = specification()['criteria']
    exact = np.arange(1., 33.).reshape(4, 8)
    values = [exact + .001*h**2 for h in (2.25, 1.5, 1)]
    result = convergence(values, 1.5, 2, .01, criteria)
    assert result['passed'] and np.isclose(result['observed_order'], 2)
    assert not convergence(values[::-1], 1.5, 2, .01, criteria)['passed']
    assert not convergence([exact + 1, exact, exact + 1], 2, 1, .01, criteria)['passed']
    assert convergence([exact, exact, exact], 2, 1, .01, criteria)['passed']


@pytest.fixture
def synthetic_source(tmp_path):
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tetra = np.array([[0, 1, 2, 3]])
    coords, tetra10 = quadratic_grid(vertices, tetra)
    c = next(iter(configurations('sr_sweep').values()))
    u = np.column_stack((coords[:, 0]**2+coords[:, 1]*coords[:, 2], coords[:, 1], coords[:, 2]**2))
    p = coords[:, 0]+2*coords[:, 1]-coords[:, 2]
    path = tmp_path/'source'/'fields.h5'
    path.parent.mkdir()
    with h5py.File(path, 'w') as h:
        h.attrs.update(schema='cardiac-flow-1.0', config_json=json.dumps(c), report_json='{}', status='periodic_converged')
        for name, values in {'mesh/coordinates': coords, 'mesh/tetra4': tetra, 'mesh/tetra10': tetra10,
                             'mesh/boundary_triangles': tetra[:, :3], 'mesh/boundary_tags': [1],
                             'phase': np.arange(25)/25, 'time': np.arange(25)*c['period']/25,
                             'velocity': np.repeat(u[None], 25, axis=0),
                             'pressure': np.repeat(p[None], 25, axis=0)}.items():
            h.create_dataset(name, data=values)
    from cardiac_flow.storage import read
    return path, read(path)


def test_sampling_is_reproducible_and_uses_native_polynomials(synthetic_source):
    _, data = synthetic_source
    a, b = sample_mesh(data, 200, 42), sample_mesh(data, 200, 42)
    assert np.array_equal(a['coordinates'], b['coordinates'])
    xyz = a['coordinates']
    expected = np.column_stack((xyz[:, 0]**2+xyz[:, 1]*xyz[:, 2], xyz[:, 1], xyz[:, 2]**2))
    assert np.allclose(sample_field(data, a), expected[None])
    assert np.allclose(sample_field(data, a, 'pressure'), (xyz[:, 0]+2*xyz[:, 1]-xyz[:, 2])[None])
    assert not np.array_equal(a['coordinates'], sample_mesh(data, 200, 43)['coordinates'])


def test_volume_sampling_and_exact_local_divergence(synthetic_source):
    _, data = synthetic_source
    x = data['coordinates']
    # div(x^2, -2xy, z) = 1, including its quadratic cross term.
    u = np.column_stack((x[:, 0]**2, -2*x[:, 0]*x[:, 1], x[:, 2]))
    d = dict(data, velocity=np.repeat(u[None], 25, axis=0), config=dict(data['config'], radius=1))
    local = _local_divergence(d, speed=1)
    assert np.allclose(local['cell_rms_scaled_max_per_phase'], 1)
    assert np.allclose(local['cell_rms_scaled_p95_per_phase'], 1)
    vertices = x[:4]
    coordinates = np.concatenate((vertices, 2*vertices+[3, 0, 0]))
    sampled = sample_mesh(dict(coordinates=coordinates, tetra4=np.array([[0, 1, 2, 3], [4, 5, 6, 7]])), 20000, 42)
    assert abs(np.mean(sampled['cell_index'] == 1)-8/9) < .01


def test_preliminary_export_preserves_sources_mesh_units_and_phases(synthetic_source, tmp_path):
    path, source = synthetic_source
    before = file_hash(path)
    output = tmp_path/'sr.h5'
    result = export_dataset(output, preliminary_paths=[path], samples=32, seed=7, pressure=True)
    assert result['preliminary'] and file_hash(path) == before
    with pytest.raises(ValueError, match='preliminary'):
        list(iter_simulations(output))
    d = next(iter_simulations(output, allow_preliminary=True))
    assert d['velocity'].shape == (25, 32, 3) and d['pressure'].shape == (25, 32)
    assert d['split'] == 'preliminary'
    speed = source['config']['mean_flow']/(np.pi*source['config']['radius']**2)
    assert np.allclose(d['dimensionless_velocity'], d['velocity']/speed)
    with h5py.File(output) as h:
        g = h['simulations/preliminary_000']
        assert not h.attrs['accuracy_certified']
        assert np.array_equal(g['mesh/tetra10'], source['tetra10'])
        assert g['velocity'].attrs['units'] == 'm/s'
    batches = list(iter_batches(output, 'preliminary', allow_preliminary=True))
    assert len(batches) == 25
    assert batches[0]['inputs'].shape == (32, 11)
    with pytest.raises(FileExistsError):
        export_dataset(output, preliminary_paths=[path])


def test_export_excludes_uncertified_or_stale_cases(synthetic_source, tmp_path):
    path, source = synthetic_source
    manifest = dict(study='sr_sweep', accuracy_qualification={'status': 'passed'},
                    accuracy_qualification_sha256='qualification', cases=[dict(id='case', output=str(path.parent),
                    config_sha256='config', parameters=source['config'])])
    m = tmp_path/'manifest.json'
    m.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='no accepted'):
        export_dataset(tmp_path/'rejected.h5', manifest_path=m)
    cert = dict(schema='cardiac-flow-sr-acceptance-1', status='accepted', source_sha256=file_hash(path),
                config_sha256='config', qualification_sha256='qualification', pressure_accepted=False)
    path.with_name('accuracy.json').write_text(json.dumps(cert))
    result = export_dataset(tmp_path/'accepted.h5', manifest_path=m, samples=10)
    assert result['simulations'] == 1 and not result['preliminary']
    with h5py.File(path, 'a') as h:
        h['velocity'][0, 0, 0] += 1
    with pytest.raises(ValueError, match='no accepted'):
        export_dataset(tmp_path/'stale.h5', manifest_path=m)


@pytest.mark.skipif(os.name == 'nt', reason='experiment locks/submission use Linux')
def test_draft_manifest_dry_run_and_accuracy_gate(tmp_path):
    from scripts.experiments import build_manifest, run_case
    path = tmp_path/'sr.json'
    result = build_manifest('sr_sweep', None, path, tmp_path/'runs', REPO)
    assert len(result['cases']) == 36
    assert result == build_manifest('sr_sweep', None, path, tmp_path/'runs', REPO)
    original = path.read_bytes()
    with pytest.raises(ValueError, match='immutable'):
        build_manifest('sr_sweep', None, path, tmp_path/'different_runs', REPO)
    assert path.read_bytes() == original
    assert run_case(path, 0, dry_run=True) == 0
    with pytest.raises(ValueError, match='qualification'):
        run_case(path, 0)
    # Copying a draft into a single-case manifest cannot bypass qualification.
    single = tmp_path/'single.json'
    config = path.parent/result['cases'][0]['config']
    build_manifest(None, config, single, tmp_path/'single', REPO)
    with pytest.raises(ValueError, match='qualification'):
        run_case(single, 0)
    accuracy = tmp_path/'accuracy.json'
    build_manifest('sr_accuracy', None, accuracy, tmp_path/'accuracy_runs', REPO)
    blocked = assess(accuracy, tmp_path/'qualification.json')
    assert blocked['status'] == 'blocked' and len(blocked['errors']) == 32
    assert 'selected_numerics' not in blocked


def test_qualification_gates_keep_velocity_pressure_and_conservation_separate(tmp_path, monkeypatch):
    """Synthetic convergence evidence exercises the gate, not CFD accuracy."""
    import cardiac_flow.sr_accuracy as accuracy
    entries = []
    for name, c in configurations('sr_accuracy').items():
        output = tmp_path/'runs'/name
        output.mkdir(parents=True)
        (output/'fields.h5').write_bytes(b'synthetic evidence for gate test only')
        (output/'execution.json').write_text('{"status":"completed"}')
        entries.append(dict(id=name, parameters=c, config_sha256='hash', output=str(output)))
    manifest = tmp_path/'accuracy.json'
    manifest.write_text(json.dumps(dict(study='sr_accuracy', cases=entries, git={'commit': 'test'})))
    monkeypatch.setattr(accuracy, '_verify_run', lambda entry, expected: Path(entry['output'])/'fields.h5')
    monkeypatch.setattr(accuracy, 'physical_metrics', lambda path: {'passed': True})

    def comparison(paths, *args):
        exact = np.ones((25, 32, 3))
        if len(paths) == 3:
            levels = (4, 2, 1) if '__h2_t0' in str(paths[0]) else (2.25**2, 1.5**2, 1)
            velocity = [exact+.001*h for h in levels]
            pressure = [exact[..., 0]+.5*h for h in levels]  # pressure budget fails
        else:
            velocity = [exact, exact+1e-6]
            pressure = [exact[..., 0], exact[..., 0]+1e-6]
        return {'velocity': velocity, 'pressure': pressure}, 1.

    monkeypatch.setattr(accuracy, '_comparison', comparison)
    qpath = tmp_path/'qualified.json'
    result = assess(manifest, qpath)
    assert result['status'] == 'passed' and not result['pressure_qualified']
    qualified = configurations('sr_sweep', qpath)
    assert len(qualified) == 36
    assert all(c['_sr']['numerics_status'] == 'qualified' for c in qualified.values())
    if os.name != 'nt':
        from scripts.experiments import build_manifest
        sweep_path = tmp_path/'sweep.json'
        build_manifest('sr_sweep', None, sweep_path, tmp_path/'sweep_runs', REPO)
        build_manifest('sr_sweep', None, sweep_path, tmp_path/'sweep_runs', REPO, qpath)
        original = sweep_path.read_bytes()
        with pytest.raises(ValueError, match='immutable'):
            build_manifest('sr_sweep', None, sweep_path, tmp_path/'sweep_runs', REPO)
        assert sweep_path.read_bytes() == original
    monkeypatch.setattr(accuracy, 'physical_metrics', lambda path: {'passed': False})
    failed = assess(manifest, tmp_path/'failed.json')
    assert failed['status'] == 'failed' and 'selected_numerics' not in failed


@pytest.mark.skipif(os.name == 'nt' or shutil.which('bash') is None, reason='requires Linux bash')
def test_slurm_submission_without_cluster_or_cfd(tmp_path):
    """Fake only Apptainer/native imports/sbatch; run the real manifest/submit code."""
    fakebin = tmp_path/'bin'
    fakebin.mkdir()
    apptainer = fakebin/'apptainer'
    apptainer.write_text(f'#!{sys.executable}\nimport os,sys\na=sys.argv[8:]\n'
        'if a and a[0]=="-c" and "import dolfin" in a[1]: sys.exit(0)\n'
        'os.execv(sys.executable,[sys.executable,*a])\n')
    sbatch = fakebin/'sbatch'
    sbatch.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
        'open(os.environ["TEST_SBATCH_LOG"],"w").write(json.dumps(sys.argv[1:]))\n')
    apptainer.chmod(0o755)
    sbatch.chmod(0o755)
    image = tmp_path/'fake.sif'
    image.touch()
    config = tmp_path/'cluster.env'
    config.write_text('\n'.join(f'{key}={shlex.quote(str(value))}' for key, value in dict(
        PROJECT_ROOT=REPO, RUN_ROOT=tmp_path/'runs', LOG_ROOT=tmp_path/'logs',
        CLUSTER_EXECUTION_MODE='apptainer', APPTAINER_IMAGE=image,
        APPTAINER_PYTHON=sys.executable, SLURM_ARRAY_CONCURRENCY=2).items()))
    log = tmp_path/'sbatch.json'
    env = dict(os.environ, CLUSTER_CONFIG=str(config), PATH=f'{fakebin}:{os.environ["PATH"]}',
               TEST_SBATCH_LOG=str(log), SLURM_CPUS='1', SLURM_DRY_RUN='1')
    script = REPO/'scripts/slurm/submit-array.sh'
    preview = subprocess.run(['bash', str(script), 'sr_sweep'], env=env, capture_output=True, text=True)
    assert preview.returncode == 0, preview.stderr
    assert '--cpus-per-task=1' in preview.stdout and '--array=0-35%2' in preview.stdout
    assert not log.exists()
    draft_config = tmp_path/'runs/manifests/sr_sweep.configs/re060_a12_e00.json'
    one_script = REPO/'scripts/slurm/submit-one.sh'
    for output in ('one', 'another'):
        single = subprocess.run(['bash', str(one_script), str(draft_config), str(tmp_path/output)],
                                env=env, capture_output=True, text=True)
        assert single.returncode == 0, single.stderr
    assert len(list((tmp_path/'runs/manifests').glob('single-*.json'))) == 2
    one_manifest = next((tmp_path/'runs/manifests').glob('single-*.json'))
    mapping = json.loads(one_manifest.read_text())
    assert Path(mapping['cases'][0]['output']).name in ('one', 'another')
    bad = subprocess.run(['bash', str(script), 'sr_pilot'], env=dict(env, SLURM_CPUS='9'), capture_output=True)
    assert bad.returncode != 0 and not log.exists()
    blocked = subprocess.run(['bash', str(script), 'sr_sweep'], env=dict(env, SLURM_DRY_RUN='0'), capture_output=True)
    assert blocked.returncode != 0 and not log.exists()
    submitted = subprocess.run(['bash', str(script), 'sr_pilot'], env=dict(env, SLURM_DRY_RUN='0'), capture_output=True)
    assert submitted.returncode == 0, submitted.stderr
    args = json.loads(log.read_text())
    assert '--ntasks=1' in args and '--cpus-per-task=1' in args and '--array=0-2%2' in args
    # A spool copy still finds the project code via the submitted environment.
    spool = tmp_path/'spool'/'slurm_script'
    spool.parent.mkdir()
    shutil.copy(REPO/'scripts/slurm/run-case.sbatch', spool)
    direct = subprocess.run(['bash', str(spool), str(tmp_path/'absent.json'), '0'],
        env=dict(env, CFD_PROJECT_ROOT=str(REPO)), capture_output=True, text=True)
    assert direct.returncode != 0 and 'FileNotFoundError' in direct.stderr
    assert 'cluster-common.sh: No such file' not in direct.stderr
