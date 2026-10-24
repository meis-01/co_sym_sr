"""SR sampling/export/reading with NumPy and HDF5 only; no CFD imports.

Each simulation owns its mesh and all 25 phases. Samples are uniform in mesh
volume, not uniform in mesh-node index, and use exact native P2 interpolation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from .storage import EDGES, read


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sample_mesh(data, count, seed):
    if not isinstance(count, int) or count <= 0:
        raise ValueError('sample count must be a positive integer')
    vertices = data['coordinates'][data['tetra4']]
    volumes = np.abs(np.linalg.det(vertices[:, 1:] - vertices[:, :1])) / 6
    if not np.all(np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError('mesh contains invalid or degenerate tetrahedra')
    rng = np.random.default_rng(seed)
    cells = rng.choice(len(vertices), count, p=volumes / volumes.sum())
    bary = rng.dirichlet(np.ones(4), count)
    weights = np.column_stack([bary * (2 * bary - 1),
                               *[4*bary[:, a]*bary[:, b] for a, b in EDGES]])
    return dict(coordinates=np.einsum('ni,nij->nj', bary, vertices[cells]),
                cell_index=cells, barycentric=bary, weights=weights)


def sample_field(data, sampling, field='velocity'):
    if field == 'pressure':
        nodes = data['tetra4'][sampling['cell_index']]
        return np.stack([np.einsum('ni,ni->n', phase[nodes], sampling['barycentric'])
                         for phase in data[field]])
    nodes = data['tetra10'][sampling['cell_index']]
    return np.stack([np.einsum('nic,ni->nc', phase[nodes], sampling['weights'])
                     for phase in data[field]])


def _check_source(data):
    n = len(data['coordinates'])
    if (data['velocity'].shape != (25, n, 3) or data['pressure'].shape != (25, n)
            or not np.allclose(data['phase'], np.arange(25)/25)
            or not np.all(np.isfinite(data['velocity']))
            or not np.all(np.isfinite(data['pressure']))):
        raise ValueError('source must contain 25 finite, correctly indexed phases')


def _sidecars(path):
    return {name: json.loads(path.with_name(name).read_text(encoding='utf-8'))
            for name in ('execution.json', 'validation.json', 'report.json', 'accuracy.json')
            if path.with_name(name).is_file()}


def export_dataset(output, *, manifest_path=None, preliminary_paths=None,
                   samples=2048, seed=20261024, pressure=False):
    """Export accepted cases, or an explicitly preliminary format demonstration.

    Unaccepted cases are listed as exclusions. Preliminary cases have no train/
    validation/test assignment and readers must explicitly opt into them.
    """
    if (manifest_path is None) == (preliminary_paths is None):
        raise ValueError('provide a manifest OR preliminary source paths')
    preliminary = preliminary_paths is not None
    manifest = None
    if preliminary:
        records = [dict(id=f'preliminary_{i:03d}', output=str(Path(p).resolve().parent),
                        source=str(Path(p).resolve())) for i, p in enumerate(preliminary_paths)]
    else:
        manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
        if (manifest.get('study') != 'sr_sweep'
                or manifest.get('accuracy_qualification', {}).get('status') != 'passed'):
            raise ValueError('certified exports require a qualified sr_sweep manifest')
        records = manifest['cases']
    accepted, exclusions = [], []
    for record in records:
        path = Path(record.get('source', str(Path(record['output']) / 'fields.h5')))
        reason = None
        metadata = _sidecars(path)
        cert = metadata.get('accuracy.json', {})
        if not path.is_file():
            reason = 'missing source'
        elif not preliminary:
            if (cert.get('schema') != 'cardiac-flow-sr-acceptance-1'
                    or cert.get('status') != 'accepted'
                    or cert.get('source_sha256') != file_hash(path)
                    or cert.get('config_sha256') != record['config_sha256']
                    or cert.get('qualification_sha256') != manifest['accuracy_qualification_sha256']):
                reason = 'missing, rejected, or stale numerical acceptance'
            elif pressure and not cert.get('pressure_accepted'):
                reason = 'pressure did not meet acceptance criteria'
        if reason:
            exclusions.append(dict(id=record['id'], reason=reason))
        else:
            accepted.append((record, path, metadata))
    if not accepted:
        raise ValueError(f'no accepted simulations to export: {exclusions}')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, 'x') as target:
        target.attrs.update(schema='cardiac-flow-sr-1', status='writing',
                            preliminary=preliminary, accuracy_certified=not preliminary,
                            sampling='uniform tetrahedral volume; Dirichlet(1,1,1,1)', seed=seed,
                            excluded_json=json.dumps(exclusions),
                            manifest_json=json.dumps(manifest),
                            complete_campaign=not preliminary and len(accepted) == 36,
                            normalization='x/R,y/R,z/L; u/(Qmean/(pi R^2)); (p-pout)/(rho Umean^2)')
        simulations = target.create_group('simulations')
        configuration_groups = set()
        for record, path, metadata in accepted:
            before = file_hash(path)
            data = read(path)
            _check_source(data)
            c = data['config']
            split = 'preliminary' if preliminary else c['_sr']['split']
            if not preliminary and split not in ('train', 'validation', 'test'):
                raise ValueError(f'case {record["id"]} has no configuration split')
            if not preliminary:
                if c != record['parameters']:
                    raise ValueError('source configuration differs from manifest')
                key = c['_sr']['configuration_group']
                if key in configuration_groups:
                    raise ValueError('duplicate physical configuration would leak across splits')
                configuration_groups.add(key)
            # ID-derived seeds are stable under filtering, ordering, and parallel export.
            case_seed = int.from_bytes(hashlib.sha256(f'{seed}:{record["id"]}'.encode()).digest()[:8], 'little')
            sampling = sample_mesh(data, samples, case_seed)
            u = sample_field(data, sampling)
            speed = c['mean_flow']/(np.pi*c['radius']**2)
            re = 2*c['rho']*c['mean_flow']/(np.pi*c['radius']*c['mu'])
            alpha = c['radius']*np.sqrt(2*np.pi*c['rho']/(c['period']*c['mu']))
            group = simulations.create_group(record['id'])
            group.attrs.update(split=split, accepted_for_training=not preliminary and split == 'train',
                               numerically_accepted=not preliminary,
                               source=str(path), source_sha256=before,
                               config_json=json.dumps(c), metadata_json=json.dumps(metadata),
                               Re=re, alpha=alpha, eccentricity=c['eccentricity'],
                               mean_speed_m_per_s=speed, pressure_scale_Pa=c['rho']*speed**2,
                               sample_seed=case_seed)
            with h5py.File(path, 'r') as source:
                source.copy('mesh', group)
                group.attrs['original_hdf5_attributes_json'] = json.dumps(
                    {k: v.item() if isinstance(v, np.generic) else v for k, v in source.attrs.items()})
            arrays = {'coordinates': (sampling['coordinates'], 'm'),
                      'normalized_coordinates': (sampling['coordinates']/[c['radius'], c['radius'], c['length']], '1'),
                      'time': (data['time'], 's'), 'phase': (data['phase'], 'cycle'),
                      'harmonics': (np.asarray(c['harmonics']).reshape(-1, 2), 'relative amplitude,rad'),
                      'velocity': (u, 'm/s'), 'dimensionless_velocity': (u/speed, '1'),
                      'sampling/cell_index': (sampling['cell_index'], 'index'),
                      'sampling/barycentric': (sampling['barycentric'], '1')}
            if pressure:
                p = sample_field(data, sampling, 'pressure')
                arrays.update(pressure=(p, 'Pa'), dimensionless_pressure=(
                    (p-c.get('outlet_pressure', 0.))/(c['rho']*speed**2), '1'))
            for name, (array, units) in arrays.items():
                group.create_dataset(name, data=array, compression='gzip', shuffle=True).attrs['units'] = units
            if file_hash(path) != before:
                raise RuntimeError(f'source changed during export: {path}')
        target.attrs['status'] = 'ready'
    return dict(output=str(output), simulations=len(accepted), preliminary=preliminary, excluded=exclusions)


def iter_simulations(path, split=None, *, allow_preliminary=False):
    """Read one variable-size simulation at a time; original CFD runtime is unnecessary."""
    with h5py.File(path, 'r') as dataset:
        if dataset.attrs.get('schema') != 'cardiac-flow-sr-1' or dataset.attrs.get('status') != 'ready':
            raise ValueError('unsupported or incomplete SR export')
        if dataset.attrs['preliminary'] and not allow_preliminary:
            raise ValueError('preliminary data require allow_preliminary=True and are not training data')
        for name, group in dataset['simulations'].items():
            if split is not None and group.attrs['split'] != split:
                continue
            result = {k: group[k][:] for k in ('coordinates', 'normalized_coordinates', 'phase',
                      'time', 'velocity', 'dimensionless_velocity', 'harmonics')}
            for field in ('pressure', 'dimensionless_pressure'):
                if field in group:
                    result[field] = group[field][:]
            result.update(id=name, split=group.attrs['split'], config=json.loads(group.attrs['config_json']),
                          metadata=json.loads(group.attrs['metadata_json']))
            yield result


def iter_batches(path, split='train', *, allow_preliminary=False):
    """Yield one phase of one configuration as named feature/target arrays."""
    for d in iter_simulations(path, split, allow_preliminary=allow_preliminary):
        c = d['config']
        re = 2*c['rho']*c['mean_flow']/(np.pi*c['radius']*c['mu'])
        alpha = c['radius']*np.sqrt(2*np.pi*c['rho']/(c['period']*c['mu']))
        names = ['x', 'y', 'z', 'phase', 'Re', 'alpha', 'eccentricity']
        names += [f'harmonic_{i+1}_{part}' for i in range(len(c['harmonics'])) for part in ('amplitude', 'phase_rad')]
        for k, phase in enumerate(d['phase']):
            parameters = np.tile([phase, re, alpha, c['eccentricity'], *np.ravel(c['harmonics'])], (len(d['coordinates']), 1))
            yield dict(configuration=d['id'], feature_names=names,
                       feature_units=['m', 'm', 'm', 'cycle', '1', '1', '1'] + ['1', 'rad']*len(c['harmonics']),
                       normalized_feature_names=['x_over_R', 'y_over_R', 'z_over_L', *names[3:]],
                       inputs=np.column_stack((d['coordinates'], parameters)),
                       normalized_inputs=np.column_stack((d['normalized_coordinates'], parameters)),
                       velocity=d['velocity'][k], dimensionless_velocity=d['dimensionless_velocity'][k],
                       pressure=d['pressure'][k] if 'pressure' in d else None)
