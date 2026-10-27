"""Evidence-based SR qualification; structural validation alone never qualifies.

No solver runtime is imported. Reports deliberately separate discretization,
physical conservation, and independently simulated rotation errors.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .analysis import validate
from .compare import evaluate
from .storage import EDGES, read
from .sr_data import sample_mesh, file_hash
from .sr_campaign import (SPEC_PATH, configurations, specification, group_id,
                          solver_signature, assessment_signature, load_qualification, production_numerics)


def write_report(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n', encoding='utf-8')
    return report


def relative_difference(a, b):
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(a), 1e-30))


def convergence(values, ratio, order_cap, limit, criteria):
    """Three-level, common-point L2 GCI estimate with a conservative order cap.

    This sampled field estimate is conditional on monotone, aligned differences;
    it is not a rigorous bound on every point or on unsampled configurations.
    """
    a, b, c = values
    first, second = a-b, b-c
    norm = max(float(np.linalg.norm(c)), 1e-30)
    d1, d2 = float(np.linalg.norm(first))/norm, float(np.linalg.norm(second))/norm
    result = dict(coarse_medium_relative_l2=d1, medium_fine_relative_l2=d2,
                  refinement_ratio=ratio, safety_factor=1.25, limit=limit)
    if max(d1, d2) < 1e-10:
        return dict(result, passed=True, fine_gci=0., observed_order=None,
                    effective_order=None, difference_alignment=None, note='differences below numerical floor')
    p = float(np.log(d1/d2)/np.log(ratio)) if d1 > 0 and d2 > 0 else None
    alignment = float(np.vdot(first, second).real / max(np.linalg.norm(first)*np.linalg.norm(second), 1e-30))
    monotone = p is not None and p >= criteria['minimum_observed_order'] and d2 < d1
    gci = 1.25*d2/(ratio**min(p, order_cap)-1) if monotone else None
    return dict(result, observed_order=p, effective_order=min(p, order_cap) if monotone else None,
                difference_alignment=alignment, fine_gci=gci,
                passed=bool(monotone and alignment >= criteria['minimum_error_alignment'] and gci <= limit))


def _local_divergence(data, speed):
    """Cell RMS/percentiles from exact degree-two quadrature of (div u)^2."""
    aa, bb = .5854101966249685, .1381966011250105
    quadrature = np.full((4, 4), bb)
    np.fill_diagonal(quadrature, aa)
    values, volumes = [], []
    for start in range(0, len(data['tetra4']), 512):
        cells = data['tetra4'][start:start+512]
        vertices = data['coordinates'][cells]
        volumes.append(np.abs(np.linalg.det(vertices[:, 1:]-vertices[:, :1]))/6)
        inv = np.linalg.inv(np.stack([vertices[:, j]-vertices[:, 0] for j in (1, 2, 3)], axis=-1))
        gradients = np.concatenate((-inv.sum(axis=1)[:, None], inv), axis=1)
        velocity = data['velocity'][:, data['tetra10'][start:start+512]]
        square = np.zeros(velocity.shape[:2])
        for l in quadrature:
            basis = np.concatenate(((4*l-1)[None, :, None]*gradients,
                np.stack([4*(l[a]*gradients[:, b]+l[b]*gradients[:, a]) for a, b in EDGES], axis=1)), axis=1)
            square += np.einsum('tcij,cij->tc', velocity, basis)**2/4
        values.append(np.sqrt(square)*data['config']['radius']/speed)
    local = np.concatenate(values, axis=1)
    weights = np.concatenate(volumes)
    percentiles = []
    for row in local:
        order = np.argsort(row)
        percentiles.append(float(np.interp(.95, np.cumsum(weights[order])/weights.sum(), row[order])))
    return dict(cell_rms_scaled_max_per_phase=local.max(axis=1).tolist(),
                cell_rms_scaled_p95_per_phase=percentiles)


def physical_metrics(path):
    validation = validate(path)  # Retains every existing validation threshold.
    data = read(path)
    c = data['config']
    with h5py.File(path, 'r') as source:
        report = json.loads(source.attrs['report_json'])
    speed = c['mean_flow']/(np.pi*c['radius']**2)
    xyz = data['coordinates'][data['tetra4']]
    volume = float(np.sum(np.abs(np.linalg.det(xyz[:, 1:]-xyz[:, :1])))/6)
    metrics = dict(local_divergence_scaled=validation['final_cycle_max_divergence_scaled'],
                   mass_imbalance_scaled=validation['exported_flux_error'],
                   prescribed_inlet_error_scaled=validation['requested_inlet_flow_error_scaled'],
                   periodic_velocity=validation['periodic_velocity'], periodic_pressure=validation['periodic_pressure'],
                   geometry_volume_error=abs(volume/(np.pi*c['radius']**2*c['length'])-1),
                   inlet_flow_per_phase=[r['inlet_flow'] for r in report['phases']],
                   outlet_flow_per_phase=[r['outlet_flow'] for r in report['phases']],
                   **_local_divergence(data, speed))
    s = specification()
    checks = {name: metrics[name] < s['criteria'][name] for name in (
        'local_divergence_scaled', 'mass_imbalance_scaled', 'prescribed_inlet_error_scaled', 'geometry_volume_error')}
    checks['local_divergence_p95_scaled'] = max(metrics['cell_rms_scaled_p95_per_phase']) < s['criteria']['local_divergence_p95_scaled']
    checks['periodic_velocity'] = metrics['periodic_velocity'] < s['periodic_tolerance']
    checks['periodic_pressure'] = metrics['periodic_pressure'] < s['periodic_tolerance']
    return dict(metrics=metrics, checks=checks, passed=all(checks.values()))


def _verify_run(entry, expected):
    path = Path(entry['output']) / 'fields.h5'
    execution = json.loads(path.with_name('execution.json').read_text(encoding='utf-8'))
    with h5py.File(path, 'r') as source:
        config = json.loads(source.attrs['config_json'])
        report = json.loads(source.attrs['report_json'])
    if (config != expected or entry['parameters'] != expected
            or execution.get('status') != 'completed'
            or execution.get('config_sha256') != entry['config_sha256']
            or execution.get('case_id') != entry['id']
            or solver_signature(report) != solver_signature()):
        raise ValueError('configuration, completed execution provenance, or solver hashes do not match')
    return path


def _comparison(paths, sample_count, seed, criteria, angle=0.):
    data = [read(p) for p in paths]
    points = sample_mesh(data[0], sample_count, seed)['coordinates']
    rot = np.array([[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1.]])
    fields = {field: [] for field in ('velocity', 'pressure')}
    for i, d in enumerate(data):
        if not np.allclose(d['phase'], data[0]['phase']):
            raise ValueError('comparison phase mismatch')
        target = points @ rot.T if i and angle else points
        fields['velocity'].append(evaluate(d, target))
        fields['pressure'].append(evaluate(d, target, 'pressure'))
    if angle:
        fields['velocity'][0] = fields['velocity'][0] @ rot.T
    valid = np.logical_and.reduce([np.all(np.isfinite(a), axis=tuple(i for i in range(a.ndim) if i != 1))
                                   for group in fields.values() for a in group])
    coverage = float(valid.mean())
    if coverage < criteria['comparison_coverage']:
        raise ValueError(f'comparison covers only {coverage:.3%} of volume samples')
    return {field: [a[:, valid] for a in group] for field, group in fields.items()}, coverage


def assess(manifest_path, output):
    """Select the finest candidate only when all representative evidence passes."""
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    expected = configurations('sr_accuracy')
    entries = {case['id']: case for case in manifest['cases']}
    if manifest['study'] != 'sr_accuracy' or set(entries) != set(expected):
        raise ValueError('assessment requires the complete current sr_accuracy manifest')
    s, physical, paths, errors = specification(), {}, {}, {}
    result = dict(schema='cardiac-flow-sr-qualification-1', status='blocked',
                  campaign_sha256=file_hash(SPEC_PATH), solver_signature=solver_signature(),
                  assessment_signature=assessment_signature(),
                  criteria=s['criteria'], manifest_sha256=file_hash(manifest_path), git=manifest['git'],
                  numerical_convergence={}, conservation=physical, symmetry={}, evidence={}, errors=errors,
                  scope='Representative evidence for this fixed geometry/waveform/parameter grid; not a pointwise error bound')
    for name, entry in entries.items():
        try:
            path = _verify_run(entry, expected[name])
            physical[name] = physical_metrics(path)
            paths[name] = path
            result['evidence'][name] = dict(path=str(path), sha256=file_hash(path),
                config_sha256=entry['config_sha256'], execution_sha256=file_hash(path.with_name('execution.json')),
                execution=json.loads(path.with_name('execution.json').read_text(encoding='utf-8')))
        except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as error:
            errors[name] = str(error)
    if errors:
        return write_report(output, result)
    velocity_ok, pressure_ok = True, True
    try:
        mesh_ratios = np.array(s['mesh_sizes'][:-1])/s['mesh_sizes'][1:]
        time_ratios = np.array(s['steps_per_cycle'][1:])/s['steps_per_cycle'][:-1]
        if not (np.allclose(mesh_ratios, mesh_ratios[0]) and np.allclose(time_ratios, time_ratios[0])
                and mesh_ratios[0] > 1 and time_ratios[0] > 1):
            raise ValueError('three-level GCI requires constant mesh/time refinement ratios greater than one')
        for i, p in enumerate(s['representatives']):
            prefix = group_id(*p)
            numerical = {}
            for kind, names, ratio, cap in (
                ('mesh', [f'{prefix}__h{h}_t2' for h in range(3)], float(mesh_ratios[0]), 2.),
                ('time', [f'{prefix}__h2_t{t}' for t in range(3)], float(time_ratios[0]), 1.)):
                fields, coverage = _comparison([paths[n] for n in names], s['comparison_samples'],
                                               s['sampling_seed']+i, s['criteria'])
                numerical[kind] = dict(coverage=coverage)
                for field in fields:
                    numerical[kind][field] = convergence(fields[field], ratio, cap,
                        s['criteria'][f'{field}_{kind}_gci'], s['criteria'])
                    if field == 'velocity':
                        velocity_ok &= numerical[kind][field]['passed']
                    else:
                        pressure_ok &= numerical[kind][field]['passed']
            fields, coverage = _comparison([paths[f'{prefix}__h2_t2'], paths[f'{prefix}__periodic_tight']],
                s['comparison_samples'], s['sampling_seed']+i, s['criteria'])
            numerical['periodic'] = dict(coverage=coverage, **{
                field: relative_difference(*arrays) for field, arrays in fields.items()})
            velocity_ok &= numerical['periodic']['velocity'] <= s['criteria']['periodic_difference']
            pressure_ok &= numerical['periodic']['pressure'] <= s['criteria']['periodic_difference']
            for name in (f'{prefix}__h2_t2', f'{prefix}__periodic_tight'):
                velocity_ok &= physical[name]['passed']
            result['numerical_convergence'][prefix] = numerical
        for p in s['rotation_representatives']:
            prefix = group_id(*p)
            for i, angle in enumerate(s['rotation_angles']):
                symmetry = dict(angle_radians=angle, levels={})
                for h in (1, 2):
                    pair = [paths[f'{prefix}__h{h}_t2'], paths[f'{prefix}__rot{i}_h{h}']]
                    if pair[0].resolve() == pair[1].resolve():
                        raise ValueError('rotation requires two independent simulation outputs')
                    fields, coverage = _comparison(pair, s['comparison_samples'], s['sampling_seed'], s['criteria'], angle)
                    symmetry['levels'][str(h)] = dict(coverage=coverage, **{
                        field: relative_difference(*arrays) for field, arrays in fields.items()})
                for field in ('velocity', 'pressure'):
                    medium, fine = [symmetry['levels'][str(h)][field] for h in (1, 2)]
                    symmetry[f'{field}_passed'] = bool(fine <= s['criteria'][f'rotation_{field}_error']
                        and (fine < medium or fine <= s['criteria']['rotation_roundoff_floor']))
                velocity_ok &= symmetry['velocity_passed'] and physical[f'{prefix}__rot{i}_h2']['passed']
                pressure_ok &= symmetry['pressure_passed']
                result['symmetry'][f'{prefix}__rot{i}'] = symmetry
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        errors['comparison'] = str(error)
        velocity_ok = pressure_ok = False
    result.update(status='passed' if velocity_ok else 'failed', pressure_qualified=bool(velocity_ok and pressure_ok))
    if velocity_ok:
        result['selected_numerics'] = production_numerics()
    return write_report(output, result)


def certify(manifest_path, qualification_path):
    q = load_qualification(qualification_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    if (manifest['study'] != 'sr_sweep'
            or manifest.get('accuracy_qualification_sha256') != file_hash(qualification_path)):
        raise ValueError('sr_sweep manifest and qualification do not match')
    expected = configurations('sr_sweep', qualification_path)
    if {c['id'] for c in manifest['cases']} != set(expected):
        raise ValueError('certification requires the complete 36-case manifest')
    results = {}
    for entry in manifest['cases']:
        path = Path(entry['output']) / 'fields.h5'
        report = dict(schema='cardiac-flow-sr-acceptance-1', status='rejected', pressure_accepted=False,
                      config_sha256=entry['config_sha256'], qualification_sha256=file_hash(qualification_path),
                      numerical_convergence=q['numerical_convergence'], symmetry=q['symmetry'])
        try:
            _verify_run(entry, expected[entry['id']])
            report['source_sha256'] = file_hash(path)
            report['conservation'] = physical_metrics(path)
            if report['conservation']['passed']:
                report['status'] = 'accepted'
                report['pressure_accepted'] = q['pressure_qualified']
        except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as error:
            report['reason'] = str(error)
        if path.parent.is_dir():
            write_report(path.with_name('accuracy.json'), report)
        results[entry['id']] = report['status']
    return results
