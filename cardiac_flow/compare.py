"""Evaluate native quadratic tetrahedra for convergence and rotation studies."""
import numpy as np
from scipy.spatial import cKDTree
from .storage import EDGES, read


def evaluate(data, points, field='velocity'):
    vertices = data['coordinates'][data['tetra4']]
    centers = vertices.mean(axis=1)
    tree = cKDTree(centers)
    inv = np.linalg.inv(np.stack([vertices[:, j]-vertices[:, 0] for j in (1, 2, 3)], axis=-1))
    count = min(80, len(vertices))
    _, candidates = tree.query(points, k=count)
    candidates = np.asarray(candidates).reshape(len(points), count)
    source = data[field]
    output = np.full((25, len(points), *source.shape[2:]), np.nan)
    for i, (point, options) in enumerate(zip(points, candidates)):
        b = np.einsum('kij,kj->ki', inv[options], point-vertices[options, 0])
        bary = np.column_stack((1-b.sum(axis=1), b))
        found = np.flatnonzero(np.all(bary >= -1e-9, axis=1))
        if not len(found):
            continue
        local = found[0]
        l = bary[local]
        weights = np.r_[l*(2*l-1), [4*l[a]*l[b] for a, b in EDGES]]
        nodes = data['tetra10'][options[local]]
        output[:, i] = np.tensordot(source[:, nodes], weights, axes=(1, 0))
    return output


def compare(reference_path, candidate_path, angle=0.0):
    """Candidate should equal reference rotated by angle about z; report L2 nodes."""
    a, b = read(reference_path), read(candidate_path)
    assert np.allclose(a['phase'], b['phase'])
    theta = angle
    rot = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    xyz = a['coordinates']
    # Exclude boundary and end planes to avoid comparing distinct faceted walls.
    from .model import radius
    mask = ((np.hypot(xyz[:, 0], xyz[:, 1]) < .85*radius(xyz[:, 2], a['config'])) &
            (xyz[:, 2] > .05*a['config']['length']) & (xyz[:, 2] < .95*a['config']['length']))
    ids = np.flatnonzero(mask)
    ids = ids[::max(1, len(ids)//1500)]
    expected = a['velocity'][:, ids]@rot.T
    actual = evaluate(b, xyz[ids]@rot.T)
    valid = np.all(np.isfinite(actual), axis=(0, 2))
    if valid.mean() < .95:
        raise RuntimeError('Insufficient overlapping interior points')
    error = np.linalg.norm(actual[:, valid]-expected[:, valid])/np.linalg.norm(expected[:, valid])
    actual_p = evaluate(b, xyz[ids]@rot.T, field='pressure')
    expected_p = a['pressure'][:, ids]
    pressure_error = np.linalg.norm(actual_p[:, valid]-expected_p[:, valid])/max(np.linalg.norm(expected_p[:, valid]), 1e-30)
    return dict(relative_velocity_l2_nodes=float(error), relative_pressure_l2_nodes=float(pressure_error),
                points=int(valid.sum()), angle_radians=angle,
                note='Interior node-weighted norm; excludes outer 15% of radius and end 5% of length')
