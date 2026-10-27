"""One-session SLURM campaign orchestration; no solver is imported by this module."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qualified(state):
    if state.get('python_command'):
        subprocess.run([*state['python_command'], str(Path(__file__).resolve()), 'check-qualification',
                        '--qualification', state['qualification']], check=True)
    else:
        from cardiac_flow.sr_campaign import load_qualification
        load_qualification(state['qualification'])


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def case_status(output):
    # Match the existing executor's structural output states without importing
    # NumPy/HDF5 into the host-only SLURM controller.
    if (output / 'fields.h5').is_file() and (output / 'validation.json').is_file():
        try:
            if read(output / 'validation.json').get('status') == 'passed':
                return 'completed'
        except (OSError, ValueError):
            pass
    if (output / 'failure.json').is_file():
        return 'failed'
    return 'incomplete' if output.exists() and any(output.iterdir()) else 'not_started'


def source_hashes():
    paths = [p for folder in ('cardiac_flow', 'scripts', 'configs')
             for p in (REPO / folder).rglob('*')
             if p.is_file() and p.suffix in ('.py', '.sh', '.sbatch', '.json')]
    config = os.environ.get('CLUSTER_CONFIG')
    if config:
        paths.append(Path(config).resolve())
    return {str(p.resolve()): digest(p) for p in sorted(set(paths))}


def verify(state):
    for name, expected in state['source_sha256'].items():
        if not Path(name).is_file() or digest(name) != expected:
            raise ValueError(f'campaign source/configuration changed: {name}; use a fresh campaign root')


def resources(args):
    if not 1 <= args.cpus <= 8 or args.concurrency < 1:
        raise ValueError('CPUs must be 1..8 and concurrency must be positive')
    if not re.fullmatch(r'[1-9][0-9]*(?:[KMGTP]?)', args.memory):
        raise ValueError('memory must be a positive SLURM size, e.g. 8G (no unlimited allocation)')
    # Use an explicit finite duration, avoiding SLURM zero/unlimited semantics.
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)', args.time)
    if not match or not any(int(x or 0) for x in match.groups()):
        raise ValueError('time must be a positive [days-]hours:mm:ss duration')
    return dict(cpus=args.cpus, memory=args.memory, time=args.time,
                partition=args.partition, account=args.account, concurrency=args.concurrency)


def command(state, stage, dependency=None):
    root, r = Path(state['root']), state['resources']
    cmd = ['sbatch', '--parsable', '--ntasks=1', f'--cpus-per-task={r["cpus"]}',
           f'--mem={r["memory"]}', f'--time={r["time"]}', f'--chdir={REPO}', '--export=ALL',
           f'--job-name=sr-{stage}', f'--output={root}/logs/{stage}-%A_%a.out',
           f'--error={root}/logs/{stage}-%A_%a.err']
    for key in ('partition', 'account'):
        if r[key]:
            cmd.append(f'--{key}={r[key]}')
    if dependency:
        cmd.append(f'--dependency={dependency}')
        cmd.append('--kill-on-invalid-dep=yes')
    if stage in ('accuracy', 'sweep'):
        count = 32 if stage == 'accuracy' else 36
        cmd.extend([f'--array=0-{count-1}%{r["concurrency"]}',
                    str(REPO / 'scripts/slurm/run-case.sbatch'),
                    str(root / f'manifests/sr_{stage}.json'), '', str(root / 'workflow.json')])
    else:
        cmd.extend([str(REPO / 'scripts/slurm/sr-stage.sbatch'), str(root / 'workflow.json'), stage])
    return cmd


def submit_job(state, stage, dependency=None):
    cmd = command(state, stage, dependency)
    output = subprocess.check_output(cmd, text=True).strip()
    job = output.split(';')[0]  # --parsable can include a federation cluster suffix.
    if not re.fullmatch(r'[1-9][0-9]*', job):
        raise RuntimeError(f'sbatch returned an invalid job ID: {output!r}')
    write(Path(state['root']) / f'jobs/{stage}.json', dict(job_id=job, command=cmd))
    return job


def manifest(root, study, qualification=None, preview=False, python_command=None):
    path = root / 'manifests' / (f'{study}.preview.json' if preview else f'{study}.json')
    if python_command:
        cmd = [*python_command, str(REPO / 'scripts/experiments.py'), 'manifest', '--study', study,
               '--manifest', str(path), '--output-root', str(root / study)]
        if qualification:
            cmd.extend(['--qualification', str(qualification)])
        subprocess.run(cmd, check=True)
        result = read(path)
    else:
        from scripts.experiments import build_manifest
        result = build_manifest(study, None, path, root / study, REPO, qualification)
    expected = 32 if study == 'sr_accuracy' else 36
    if len(result['cases']) != expected:
        raise ValueError(f'{study} must have {expected} cases')
    outputs = [c['output'] for c in result['cases']]
    if len(set(outputs)) != expected or any(not Path(p).is_relative_to(root) for p in outputs):
        raise ValueError('case output paths must be unique and inside the campaign root')
    for c in result['cases']:
        if (c['parameters']['linear_solver'], c['parameters']['grad_div_ratio']) != ('fieldsplit', 10.):
            raise ValueError('campaign requires FieldSplit and grad_div_ratio=10')
    return path


def submit(args):
    r = resources(args)
    # Refuse all reuse, including dry-run roots: queued jobs must retain their inputs.
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / 'logs').mkdir()
    state = dict(schema='cardiac-flow-sr-workflow-1', root=str(root), resources=r,
                 dry_run=args.dry_run, source_sha256=source_hashes(),
                 dataset=str(root / 'sr_dataset.h5'), qualification=str(root / 'qualification.json'),
                 samples=args.samples, seed=args.seed, pressure=args.pressure)
    state['python_command'] = json.loads(args.python_command_json) if getattr(args, 'python_command_json', None) else None
    write(root / 'workflow.json', state)
    manifest(root, 'sr_accuracy', python_command=state['python_command'])
    # The sweep preview is a draft, never executed or presented as qualified evidence.
    manifest(root, 'sr_sweep', preview=True, python_command=state['python_command'])
    plan = [command(state, 'accuracy'), command(state, 'assess', 'afterany:<accuracy>'),
            command(state, 'sweep', 'afterok:<assess>'),
            command(state, 'certify', 'afterany:<sweep>'), command(state, 'export', 'afterok:<certify>')]
    write(root / 'plan.json', dict(counts=dict(accuracy=32, sweep=36), commands=plan,
                                  downstream_submission='only after assessment passes',
                                  dataset=state['dataset'], qualification=state['qualification']))
    if args.dry_run:
        print(json.dumps(read(root / 'plan.json'), indent=2))
        return
    try:
        accuracy = submit_job(state, 'accuracy')
        submit_job(state, 'assess', f'afterany:{accuracy}')
    except Exception as error:
        write(root / 'submission-error.json', dict(error=str(error)))
        raise
    print(f'Submitted campaign: {root}\nStatus: bash scripts/slurm/submit-sr.sh status {root}')


def sr_command(state, stage, *args):
    root = Path(state['root'])
    (root / 'reports').mkdir(exist_ok=True)
    # Keep complete diagnostics, including nonzero assess/certify results.
    with (root / f'reports/{stage}.out').open('w') as out, (root / f'reports/{stage}.err').open('w') as err:
        subprocess.run([*(state.get('python_command') or [sys.executable]),
                        str(REPO / 'scripts/sr.py'), stage, *map(str, args)],
                       check=True, stdout=out, stderr=err)


def stage(state_path, name):
    state = read(state_path)
    root = Path(state['root'])
    record = root / f'stages/{name}.json'
    if record.exists():
        raise ValueError(f'stage already attempted: {name}; inspect preserved diagnostics')
    write(record, dict(status='running', job_id=os.environ.get('SLURM_JOB_ID')))
    try:
        verify(state)
        if state['dry_run']:
            raise ValueError('a dry-run plan cannot execute stages')
        q = Path(state['qualification'])
        sweep = root / 'manifests/sr_sweep.json'
        if name == 'assess':
            sr_command(state, 'assess', '--manifest', root / 'manifests/sr_accuracy.json', '--output', q)
            qualified(state)
            manifest(root, 'sr_sweep', q, python_command=state.get('python_command'))
            # Wait for this controller to finish successfully before starting CFD.
            controller = os.environ['SLURM_JOB_ID']
            sweep_id = submit_job(state, 'sweep', f'afterok:{controller}')
            cert_id = submit_job(state, 'certify', f'afterany:{sweep_id}')
            submit_job(state, 'export', f'afterok:{cert_id}')
        elif name == 'certify':
            sr_command(state, 'certify', '--manifest', sweep, '--qualification', q)
        elif name == 'export':
            qualified(state)
            if read(root / 'stages/certify.json')['status'] != 'passed':
                raise ValueError('certification has not passed')
            # Recheck all certificates immediately before export; the generic exporter
            # permits partial datasets, whereas this workflow requires all 36 cases.
            sr_command(state, 'certify', '--manifest', sweep, '--qualification', q)
            if state.get('python_command'):
                subprocess.run([*state['python_command'], str(Path(__file__).resolve()),
                                'complete-export', '--state', str(state_path)], check=True)
            else:
                export_complete(state, sweep)
        else:
            raise ValueError(f'unknown stage: {name}')
        write(record, dict(status='passed'))
    except Exception as error:
        write(record, dict(status='failed', error=str(error)))
        raise


def export_complete(state, sweep):
    import h5py
    root = Path(state['root'])
    target = Path(state['dataset'])
    if target.exists():
        raise FileExistsError(target)
    # A failed/partial exporter must never leave a certified final artifact.
    temporary = root / 'sr_dataset.inprogress.h5'
    try:
        sr_command(state, 'export', '--manifest', sweep, '--output', temporary,
                   '--samples', state['samples'], '--seed', state['seed'],
                   *(['--pressure'] if state['pressure'] else []))
        with h5py.File(temporary, 'r') as h:
            splits = Counter(g.attrs['split'] for g in h['simulations'].values())
            if (h.attrs['status'] != 'ready' or not h.attrs['accuracy_certified']
                    or not h.attrs['complete_campaign'] or
                    splits != dict(train=24, validation=6, test=6)):
                raise ValueError('export is incomplete or has incorrect splits')
        temporary.rename(target)
    except Exception:
        if temporary.exists():
            with h5py.File(temporary, 'a') as h:
                h.attrs.update(accuracy_certified=False, complete_campaign=False, status='failed')
        raise


def status(root):
    state = read(root / 'workflow.json')
    jobs = {p.stem: read(p) for p in (root / 'jobs').glob('*.json')}
    stages = {p.stem: read(p) for p in (root / 'stages').glob('*.json')}
    result = dict(root=str(root), dry_run=state['dry_run'], jobs=jobs, stages=stages,
                  dataset=state['dataset'], dataset_exists=Path(state['dataset']).is_file())
    for study in ('sr_accuracy', 'sr_sweep'):
        path = root / f'manifests/{study}.json'
        result[study] = (dict(Counter(case_status(Path(c['output'])) for c in read(path)['cases']))
                         if path.exists() else 'not generated')
    q = Path(state['qualification'])
    result['qualification'] = read(q)['status'] if q.exists() else 'not assessed'
    scheduler = ''
    if jobs:
        try:
            scheduler = subprocess.check_output(['sacct', '-j', ','.join(j['job_id'] for j in jobs.values()),
                '--parsable2', '--noheader', '--format=JobID,State,ExitCode,Elapsed,MaxRSS'], text=True)
            result['scheduler'] = scheduler
        except (OSError, subprocess.CalledProcessError) as error:
            result['scheduler_unavailable'] = str(error)
    failures = ('FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'DEADLINE', 'BOOT_FAIL')
    if state['dry_run']:
        overall = 'dry_run'
    elif stages.get('export', {}).get('status') == 'passed' and result['dataset_exists']:
        overall = 'complete'
    elif (any(s['status'] == 'failed' for s in stages.values()) or
          result['qualification'] in ('failed', 'blocked') or (root / 'submission-error.json').exists() or
          any(f in scheduler for f in failures)):
        overall = 'failed'
    else:
        overall = 'pending_or_running' if jobs else 'not_submitted'
    result['overall'] = overall
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('submit')
    p.add_argument('root', type=Path, help='new, dedicated campaign directory')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--cpus', type=int, default=1)
    p.add_argument('--memory', default='8G')
    p.add_argument('--time', default='12:00:00')
    p.add_argument('--partition', default='')
    p.add_argument('--account', default='')
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--samples', type=int, default=2048)
    p.add_argument('--seed', type=int, default=20261024)
    p.add_argument('--pressure', action='store_true')
    p.add_argument('--python-command-json', help=argparse.SUPPRESS)
    p = sub.add_parser('status')
    p.add_argument('root', type=Path)
    for name in ('stage', 'verify', 'complete-export'):
        p = sub.add_parser(name)
        p.add_argument('--state', type=Path, required=True)
        if name == 'stage':
            p.add_argument('stage', choices=('assess', 'certify', 'export'))
    p = sub.add_parser('check-qualification')
    p.add_argument('--qualification', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'submit':
        if args.samples <= 0:
            parser.error('--samples must be positive')
        submit(args)
    elif args.command == 'status':
        raise SystemExit(int(status(args.root.resolve())['overall'] == 'failed'))
    elif args.command == 'verify':
        state = read(args.state)
        verify(state)
        if state['dry_run']:
            raise ValueError('dry-run plans cannot execute CFD')
    elif args.command == 'check-qualification':
        from cardiac_flow.sr_campaign import load_qualification
        load_qualification(args.qualification)
    elif args.command == 'complete-export':
        state = read(args.state)
        # Already inside the scientific runtime/container: use its Python directly.
        state['python_command'] = [sys.executable]
        export_complete(state, Path(state['root']) / 'manifests/sr_sweep.json')
    else:
        stage(args.state, args.stage)


if __name__ == '__main__':
    main()
