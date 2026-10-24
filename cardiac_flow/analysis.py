"""Dataset checks and plots; NumPy/SciPy/HDF5 only, no CFD runtime needed."""
import json
from pathlib import Path
import h5py
import numpy as np
from .storage import read
from .model import radius, flow, womersley


def divergence_rms(d):
    """Exact degree-two volume quadrature for squared divergence of P2 fields."""
    from .storage import EDGES
    vertices = d['coordinates'][d['tetra4']]
    matrices = np.stack([vertices[:, j]-vertices[:, 0] for j in (1, 2, 3)], axis=-1)
    volumes = np.abs(np.linalg.det(matrices))/6
    inv = np.linalg.inv(matrices)
    gradients = np.concatenate((-inv.sum(axis=1)[:, None], inv), axis=1)
    aa, bb = .5854101966249685, .1381966011250105
    quadrature = np.full((4, 4), bb)
    np.fill_diagonal(quadrature, aa)
    squares = np.zeros((25, len(vertices)))
    for l in quadrature:
        vertex_gradients = (4*l-1)[None, :, None]*gradients
        edge_gradients = np.stack([4*(l[i]*gradients[:, j]+l[j]*gradients[:, i]) for i, j in EDGES], axis=1)
        basis_gradients = np.concatenate((vertex_gradients, edge_gradients), axis=1)
        values = d['velocity'][:, d['tetra10']]
        div = np.einsum('tcij,cij->tc', values, basis_gradients)
        squares += div*div/4
    return np.sqrt(squares@volumes/volumes.sum())


def validate(path):
    d = read(path)
    with h5py.File(path, 'r') as h:
        report = json.loads(h.attrs['report_json'])
        tags = h['mesh/boundary_tags'][:]
        boundary = h['mesh/boundary_triangles'][:]
        assert h.attrs['status'] == 'periodic_converged'
    assert d['velocity'].shape == (25, len(d['coordinates']), 3)
    assert d['pressure'].shape == (25, len(d['coordinates']))
    assert np.all(np.isfinite(d['velocity'])) and np.all(np.isfinite(d['pressure']))
    assert np.allclose(d['phase'], np.arange(25)/25)
    assert np.allclose(np.diff(d['time']), d['config']['period']/25)
    assert d['tetra10'].min() >= 0 and d['tetra10'].max() < len(d['coordinates'])
    x = d['coordinates'][d['tetra4']]
    determinants = np.linalg.det(np.stack([x[:, k]-x[:, 0] for k in (1, 2, 3)], axis=-1))
    assert np.all(np.abs(determinants) > 1e-18), 'degenerate cells'
    wall_nodes = np.unique(boundary[tags == 3])
    wall_max = np.max(np.linalg.norm(d['velocity'][:, wall_nodes], axis=-1))
    # Independent boundary-flux integration: exact degree-2 triangle quadrature.
    edge_lookup = {}
    from .storage import EDGES
    for tet in d['tetra10']:
        for j, (a, b) in enumerate(EDGES):
            edge_lookup[tuple(sorted((int(tet[a]), int(tet[b]))))] = int(tet[4+j])
    assert np.array_equal(d['tetra10'][:, :4], d['tetra4'])
    for j, (a, b) in enumerate(EDGES):
        ids = d['tetra10'][:, 4+j]
        corners_a, corners_b = d['tetra4'][:, a], d['tetra4'][:, b]
        assert np.allclose(d['coordinates'][ids], (d['coordinates'][corners_a]+d['coordinates'][corners_b])/2)
        assert np.allclose(d['pressure'][:, ids], (d['pressure'][:, corners_a]+d['pressure'][:, corners_b])/2,
                           rtol=1e-8, atol=1e-9), 'pressure is not the declared P1 field'
    wall_p2 = set(int(x) for x in wall_nodes)
    for triangle in boundary[tags == 3]:
        for a, b in ((0, 1), (1, 2), (2, 0)):
            wall_p2.add(edge_lookup[tuple(sorted((int(triangle[a]), int(triangle[b]))))])
    wall_p2_max = np.max(np.linalg.norm(d['velocity'][:, sorted(wall_p2)], axis=-1))
    assert wall_p2_max < 1e-8, 'quadratic wall node slip'
    fluxes = []
    for tag in (1, 2):
        total = np.zeros(25)
        for triangle in boundary[tags == tag]:
            points = d['coordinates'][triangle]
            area = np.linalg.norm(np.cross(points[1]-points[0], points[2]-points[0]))/2
            mids = [edge_lookup[tuple(sorted((int(triangle[a]), int(triangle[b]))))]
                    for a, b in ((0, 1), (1, 2), (2, 0))]
            # Integral of P2 triangle vertex bases = 0; edge bases = area/3.
            total += area*np.mean(d['velocity'][:, mids, 2], axis=1)
        fluxes.append(total)
    mismatch = np.max(np.abs(fluxes[0]-fluxes[1]))/d['config']['mean_flow']
    assert mismatch < 0.01, 'inlet/outlet imbalance'
    assert wall_max < 1e-8, 'wall slip'
    assert max(report['periodic_velocity'], report['periodic_pressure']) < d['config']['periodic_tolerance']
    result = dict(status='passed', phases=25, wall_vertex_max=float(wall_max),
                  wall_p2_node_max=float(wall_p2_max),
                  exported_flux_error=float(mismatch),
                  max_fe_divergence_scaled=report['max_divergence_scaled'],
                  periodic_velocity=report['periodic_velocity'], periodic_pressure=report['periodic_pressure'])
    result['final_cycle_max_divergence_scaled'] = max(r['divergence_scaled'] for r in report['phases'])
    result['local_divergence_below_0_1'] = result['final_cycle_max_divergence_scaled'] < .1
    independently_scaled = divergence_rms(d)*d['config']['radius']/(d['config']['mean_flow']/(np.pi*d['config']['radius']**2))
    reported = np.array([r['divergence_scaled'] for r in report['phases']])
    assert np.allclose(independently_scaled, reported, atol=1e-9, rtol=1e-7), 'exported derivative mismatch'
    result['exported_derivatives_verified'] = True
    result['accuracy_certified'] = False
    result['requested_inlet_flow_error_scaled'] = float(np.max(np.abs(
        fluxes[0]-np.array([flow(t, d['config']) for t in d['time']])))/d['config']['mean_flow'])
    Path(path).with_name('validation.json').write_text(json.dumps(result, indent=2))
    return result


def plots(path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d = read(path)
    c, xyz, u = d['config'], d['coordinates'], d['velocity']
    out = Path(path).parent
    with h5py.File(path, 'r') as h:
        report = json.loads(h.attrs['report_json'])
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    z = np.linspace(0, c['length'], 200)
    axes[0, 0].plot(z*1e3, radius(z, c)*1e3, color='navy')
    axes[0, 0].plot(z*1e3, -radius(z, c)*1e3, color='navy')
    axes[0, 0].set(xlabel='z [mm]', ylabel='radius [mm]', title='Axisymmetric vessel')
    for key, label in [('inlet_flow', 'inlet'), ('outlet_flow', 'outlet')]:
        axes[0, 1].plot(d['phase'], [r[key]*1e6 for r in report['phases']], '.-', label=label)
    axes[0, 1].set(xlabel='phase [cycle]', ylabel='Q [mL/s]', title='Boundary flow')
    axes[0, 1].legend()
    plane = np.abs(xyz[:, 1]) < c['mesh_size']/4
    k = int(np.argmax([r['inlet_flow'] for r in report['phases']]))
    colors = axes[1, 0].scatter(xyz[plane, 2]*1e3, xyz[plane, 0]*1e3,
                                c=np.linalg.norm(u[k, plane], axis=1), s=12)
    axes[1, 0].set(xlabel='z [mm]', ylabel='x [mm]', title='Velocity near y = 0')
    fig.colorbar(colors, ax=axes[1, 0], label='speed [m/s]')
    middle = np.abs(xyz[:, 2]-c['length']/2) < c['mesh_size']/2
    rr = np.hypot(xyz[middle, 0], xyz[middle, 1])
    axes[1, 1].scatter(rr*1e3, u[k, middle, 2], s=9, label='3D CFD nodes near midsection')
    if c['benchmark']:
        r = np.linspace(0, c['radius'], 200)
        axes[1, 1].plot(r*1e3, womersley(r, d['time'][k], c), label='analytical')
    axes[1, 1].set(xlabel='r [mm]', ylabel='axial velocity [m/s]', title='Radial profile')
    axes[1, 1].legend(fontsize=8)
    fig.savefig(out/'overview.png', dpi=160)
    plt.close(fig)
    fig = plt.figure(figsize=(8, 7), constrained_layout=True)
    ax = fig.add_subplot(projection='3d')
    ids = np.arange(0, len(xyz), max(1, len(xyz)//350))
    ax.quiver(xyz[ids, 0]*1e3, xyz[ids, 1]*1e3, xyz[ids, 2]*1e3,
              u[k, ids, 0], u[k, ids, 1], u[k, ids, 2], length=300, normalize=False, linewidth=0.6)
    ax.set(xlabel='x [mm]', ylabel='y [mm]', zlabel='z [mm]', title=f'3D velocity, phase {k}/25')
    fig.savefig(out/'velocity_3d.png', dpi=160)
    plt.close(fig)
