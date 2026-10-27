"""Exercise the dependency controller with fake SLURM and synthetic gates, never CFD."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import h5py
import pytest

from scripts import sr_workflow as w
from cardiac_flow.sr_campaign import (
    SPEC_PATH, assessment_signature, configurations, digest, production_numerics,
    solver_signature, verify_qualification,
)


def arguments(root, **kwargs):
    return argparse.Namespace(root=root, dry_run=False, cpus=1, memory='8G', time='12:00:00',
                              partition='cpu', account='science', concurrency=2, samples=2048,
                              seed=20261024, pressure=False, **kwargs)


def qualification():
    # Synthetic passed gate for orchestration tests only, not numerical evidence.
    return dict(schema='cardiac-flow-sr-qualification-1', status='passed',
                campaign_sha256=digest(SPEC_PATH), solver_signature=solver_signature(),
                assessment_signature=assessment_signature(), selected_numerics=production_numerics(),
                pressure_qualified=False, numerical_convergence={}, symmetry={})


@pytest.fixture
def scheduler(monkeypatch):
    commands = []
    original = subprocess.check_output

    def sbatch(cmd, **kwargs):
        if cmd[0] == 'git':
            return original(cmd, **kwargs)
        if cmd[0] == 'sacct':
            return ''
        assert cmd[0] == 'sbatch', 'tests must not access a real scheduler'
        commands.append(cmd)
        return f'{100 + len(commands)};fake-cluster\n'

    monkeypatch.setattr(w.subprocess, 'check_output', sbatch)
    return commands


def test_dry_run_entire_plan_and_paths(tmp_path, scheduler):
    args = arguments(tmp_path / 'preview')
    args.dry_run = True
    w.submit(args)
    assert scheduler == []
    root = args.root
    plan = w.read(root / 'plan.json')
    assert plan['counts'] == dict(accuracy=32, sweep=36)
    assert ['--dependency=' + s in plan['commands'][i] for i, s in enumerate(
        ('afterany:<accuracy>', 'afterok:<assess>', 'afterany:<sweep>', 'afterok:<certify>'), 1)] == [True]*4
    assert '--array=0-31%2' in plan['commands'][0]
    assert '--array=0-35%2' in plan['commands'][2]
    assert Path(plan['dataset']) == root / 'sr_dataset.h5'
    assert not (root / 'manifests/sr_sweep.json').exists()
    assert w.status(root)['overall'] == 'dry_run'
    with pytest.raises(ValueError, match='dry-run'):
        w.stage(root / 'workflow.json', 'assess')
    with pytest.raises(FileExistsError):
        w.submit(args)


def test_failed_qualification_preserves_real_report_and_submits_nothing_downstream(tmp_path, scheduler):
    root = tmp_path / 'failed'
    w.submit(arguments(root))
    assert len(scheduler) == 2
    assert '--dependency=afterany:101' in scheduler[1]
    with pytest.raises(subprocess.CalledProcessError):
        # Run the real solver-free assessment of 32 missing outputs.
        w.stage(root / 'workflow.json', 'assess')
    q = w.read(root / 'qualification.json')
    assert q['status'] == 'blocked' and len(q['errors']) == 32
    assert len(scheduler) == 2
    assert not (root / 'manifests/sr_sweep.json').exists()
    assert not (root / 'sr_dataset.h5').exists()
    assert (root / 'reports/assess.out').exists()
    assert w.read(root / 'stages/assess.json')['status'] == 'failed'


def test_success_chain_and_complete_export(tmp_path, scheduler, monkeypatch):
    root = tmp_path / 'success'
    w.submit(arguments(root))
    monkeypatch.setenv('SLURM_JOB_ID', '102')
    calls = []

    def synthetic_sr(state, name, *args):
        calls.append(name)
        if name == 'assess':
            w.write(root / 'qualification.json', qualification())
        elif name == 'export':
            target = args[args.index('--output')+1]
            with h5py.File(target, 'x') as h:
                h.attrs.update(status='ready', accuracy_certified=True, complete_campaign=True)
                groups = h.create_group('simulations')
                for i, split in enumerate(['train']*24 + ['validation']*6 + ['test']*6):
                    groups.create_group(str(i)).attrs['split'] = split

    monkeypatch.setattr(w, 'sr_command', synthetic_sr)
    w.stage(root / 'workflow.json', 'assess')
    assert len(scheduler) == 5
    assert '--dependency=afterok:102' in scheduler[2]
    assert '--dependency=afterany:103' in scheduler[3]
    assert '--dependency=afterok:104' in scheduler[4]
    for cmd in scheduler:
        assert all(x in cmd for x in ('--ntasks=1', '--cpus-per-task=1', '--mem=8G',
            '--time=12:00:00', '--partition=cpu', '--account=science'))
    sweep = w.read(root / 'manifests/sr_sweep.json')
    assert len(sweep['cases']) == 36
    assert all(c['parameters']['linear_solver'] == 'fieldsplit' and
               c['parameters']['grad_div_ratio'] == 10 for c in sweep['cases'])
    assert sweep['accuracy_qualification']['selected_numerics'] == production_numerics()
    w.stage(root / 'workflow.json', 'certify')
    w.stage(root / 'workflow.json', 'export')
    assert calls == ['assess', 'certify', 'certify', 'export']
    assert (root / 'sr_dataset.h5').exists()
    assert not (root / 'sr_dataset.inprogress.h5').exists()
    # Status is read-only; mock only sacct, never submit another job.
    monkeypatch.setattr(w.subprocess, 'check_output', lambda *a, **k: '101|COMPLETED|0:0|1:00|1G\n')
    assert w.status(root)['overall'] == 'complete'


@pytest.mark.parametrize('failure', ['certify', 'partial_export', 'export_error'])
def test_no_final_dataset_on_certification_or_export_failure(tmp_path, monkeypatch, failure):
    root = tmp_path / failure
    root.mkdir()
    state = dict(root=str(root), dataset=str(root / 'sr_dataset.h5'), samples=8, seed=1, pressure=False,
                 qualification=str(root / 'qualification.json'), source_sha256={}, dry_run=False)
    w.write(root / 'workflow.json', state)
    w.write(root / 'qualification.json', qualification())
    w.write(root / 'stages/certify.json', dict(status='passed'))

    def failing_sr(state, name, *args):
        if name == 'certify' and failure == 'certify':
            raise subprocess.CalledProcessError(1, ['synthetic certification'])
        if name == 'export':
            with h5py.File(root / 'sr_dataset.inprogress.h5', 'w') as h:
                h.attrs.update(status='ready', accuracy_certified=True, complete_campaign=False)
                h.create_group('simulations')
            if failure == 'export_error':
                raise subprocess.CalledProcessError(1, ['synthetic export'])

    monkeypatch.setattr(w, 'sr_command', failing_sr)
    with pytest.raises((subprocess.CalledProcessError, ValueError)):
        w.stage(root / 'workflow.json', 'export')
    assert not (root / 'sr_dataset.h5').exists()
    if failure != 'certify':
        with h5py.File(root / 'sr_dataset.inprogress.h5') as h:
            assert not h.attrs['accuracy_certified'] and h.attrs['status'] == 'failed'


def test_profile_consistency_and_mumps_default(tmp_path):
    from cardiac_flow.model import load
    assert load(w.REPO / 'configs/solver_baseline.json').get('linear_solver', 'mumps') == 'mumps'
    for study, count in (('sr_accuracy', 32), ('sr_sweep', 36)):
        cases = configurations(study)
        assert len(cases) == count
        assert all((c['linear_solver'], c['grad_div_ratio']) == ('fieldsplit', 10.) for c in cases.values())
    q = qualification()
    q['selected_numerics']['grad_div_ratio'] = 0
    with pytest.raises(ValueError, match='qualification'):
        verify_qualification(q)
    path = tmp_path / 'source'
    path.write_text('original')
    snapshot = dict(source_sha256={str(path): digest(path)})
    path.write_text('changed')
    with pytest.raises(ValueError, match='changed'):
        w.verify(snapshot)


@pytest.mark.parametrize('key,value', [('cpus', 9), ('cpus', 0), ('memory', '0'),
    ('time', '0:00:00'), ('time', 'UNLIMITED'), ('concurrency', 0)])
def test_resource_validation_before_submission(tmp_path, scheduler, key, value):
    args = arguments(tmp_path / 'invalid')
    setattr(args, key, value)
    with pytest.raises(ValueError):
        w.submit(args)
    assert not scheduler and not args.root.exists()


def test_partial_submission_preserves_ids_and_scheduler_failure_status(tmp_path, scheduler, monkeypatch):
    root = tmp_path / 'submission-failure'
    original = w.subprocess.check_output

    def reject_controller(cmd, **kwargs):
        if '--job-name=sr-assess' in cmd:
            raise subprocess.CalledProcessError(1, cmd)
        return original(cmd, **kwargs)

    monkeypatch.setattr(w.subprocess, 'check_output', reject_controller)
    with pytest.raises(subprocess.CalledProcessError):
        w.submit(arguments(root))
    assert w.read(root / 'jobs/accuracy.json')['job_id'] == '101'
    assert (root / 'submission-error.json').exists()
    monkeypatch.setattr(w.subprocess, 'check_output', lambda *a, **k: '101_0|TIMEOUT|0:0|12:00|8G\n')
    assert w.status(root)['overall'] == 'failed'


@pytest.mark.skipif(os.name == 'nt' or shutil.which('bash') is None, reason='Linux SLURM wrappers')
def test_host_submitter_and_spooled_controller_with_fake_apptainer(tmp_path):
    """Real shell/CLI/manifest/assessment paths, no native imports or real SLURM."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    apptainer = bin_dir / 'apptainer'
    apptainer.write_text(f'#!{sys.executable}\nimport os,sys\na=sys.argv[8:]\n'
        'if a and a[0]=="-c" and "import dolfin" in a[1]: sys.exit(0)\n'
        'os.environ["FAKE_IN_CONTAINER"]="1"\n'
        'os.execv(sys.executable,[sys.executable,*a])\n')
    sbatch = bin_dir / 'sbatch'
    sbatch.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
        'assert "FAKE_IN_CONTAINER" not in os.environ, "sbatch must execute on host"\n'
        'p=os.environ["TEST_SBATCH_LOG"]\n'
        'with open(p,"a") as f: f.write(json.dumps(sys.argv[1:])+"\\n")\n'
        'print(100+sum(1 for _ in open(p)))\n')
    sacct = bin_dir / 'sacct'
    sacct.write_text('#!/bin/sh\nprintf "101|COMPLETED|0:0|00:01|1G\\n"\n')
    for path in (apptainer, sbatch, sacct):
        path.chmod(0o755)
    image = tmp_path / 'fake.sif'
    image.touch()
    config = tmp_path / 'cluster.env'
    run_root = tmp_path / 'runs'
    config.write_text('\n'.join(f'{k}={shlex.quote(str(v))}' for k, v in dict(
        PROJECT_ROOT=w.REPO, RUN_ROOT=run_root, LOG_ROOT=tmp_path / 'logs',
        CLUSTER_EXECUTION_MODE='apptainer', APPTAINER_IMAGE=image,
        APPTAINER_PYTHON=sys.executable, CLUSTER_PYTHON=sys.executable,
        SLURM_PARTITION='cpu', SLURM_ARRAY_CONCURRENCY=2).items()))
    log = tmp_path / 'submissions.jsonl'
    env = dict(os.environ, CLUSTER_CONFIG=str(config), TEST_SBATCH_LOG=str(log),
               PATH=f'{bin_dir}:{os.environ["PATH"]}', SLURM_CPUS='1', SLURM_DRY_RUN='0')
    wrapper = w.REPO / 'scripts/slurm/submit-sr.sh'

    def run(*args, **overrides):
        return subprocess.run(['bash', str(wrapper), *map(str, args)], env=dict(env, **overrides),
                              text=True, capture_output=True)

    preview = run('submit', run_root / 'preview', '--dry-run')
    assert preview.returncode == 0, preview.stderr
    assert not log.exists()
    rejected = run('submit', tmp_path / 'outside')
    assert rejected.returncode != 0 and not log.exists()
    root = run_root / 'real'
    submitted = run('submit', root)
    assert submitted.returncode == 0, submitted.stderr
    commands = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(commands) == 2 and '--dependency=afterany:101' in commands[1]
    assert '--array=0-31%2' in commands[0]
    spool = tmp_path / 'spooled-script'
    shutil.copy(w.REPO / 'scripts/slurm/sr-stage.sbatch', spool)
    controller = subprocess.run(['bash', str(spool), str(root / 'workflow.json'), 'assess'],
        env=dict(env, CFD_PROJECT_ROOT=str(w.REPO), SLURM_JOB_ID='102'), text=True, capture_output=True)
    assert controller.returncode != 0
    assert w.read(root / 'qualification.json')['status'] == 'blocked'
    assert len(log.read_text().splitlines()) == 2
    checked = run('status', root)
    assert checked.returncode == 1, checked.stderr
    assert '"overall": "failed"' in checked.stdout
    # Array wrapper also resolves a spool copy and refuses a draft before CFD.
    shutil.copy(w.REPO / 'scripts/slurm/run-case.sbatch', spool)
    draft = run_root / 'preview'
    blocked_case = subprocess.run(['bash', str(spool), str(draft / 'manifests/sr_accuracy.json'),
        '', str(draft / 'workflow.json')], env=dict(env, CFD_PROJECT_ROOT=str(w.REPO),
        SLURM_ARRAY_TASK_ID='0'), text=True, capture_output=True)
    assert blocked_case.returncode != 0 and 'dry-run plans cannot execute CFD' in blocked_case.stderr
    assert not (draft / 'sr_accuracy').exists()
