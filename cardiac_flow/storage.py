"""Portable P2 tetrahedral HDF5, independent of DOLFIN numbering."""
import json
import numpy as np
import h5py

# VTK/meshio tetra10 convention: vertices then these six edges.
EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]


def quadratic_grid(vertices, tetra):
    coords = list(vertices.copy())
    edges = {}
    connectivity = []
    for tet in tetra:
        row = list(tet)
        for i, j in EDGES:
            edge = tuple(sorted((int(tet[i]), int(tet[j]))))
            if edge not in edges:
                edges[edge] = len(coords)
                coords.append((vertices[edge[0]]+vertices[edge[1]])/2)
            row.append(edges[edge])
        connectivity.append(row)
    return np.asarray(coords), np.asarray(connectivity, dtype=np.int64)


def export(directory, mesh, tags, snapshots, c, cycle, report):
    import dolfin as df
    vertices, tetra = mesh.coordinates().copy(), mesh.cells().copy()
    coords, tetra10 = quadratic_grid(vertices, tetra)
    mesh.init(2, 0)
    boundary = [(f.entities(0), int(tags[f])) for f in df.facets(mesh) if f.exterior()]
    triangles = np.array([r[0] for r in boundary], dtype=np.int64)
    markers = np.array([r[1] for r in boundary], dtype=np.int32)
    temporary = directory/'fields.partial.h5'
    with h5py.File(temporary, 'w') as h:
        h.attrs['schema'] = 'cardiac-flow-1.0'
        h.attrs['config_json'] = json.dumps(c)
        h.attrs['report_json'] = json.dumps(report)
        h.attrs['status'] = 'periodic_converged'
        h.attrs['cell_order'] = 'VTK tetra10'
        h.attrs['boundary_tags'] = '1=inlet,2=outlet,3=wall'
        datasets = {
            'mesh/coordinates': (coords, 'm'), 'mesh/tetra10': (tetra10, 'index'),
            'mesh/tetra4': (tetra, 'index'), 'mesh/boundary_triangles': (triangles, 'index'),
            'mesh/boundary_tags': (markers, 'tag'),
            'time': ((cycle-1)*c['period']+np.arange(25)*c['period']/25, 's'),
            'phase': (np.arange(25)/25, 'cycle')}
        for name, (data, units) in datasets.items():
            h.create_dataset(name, data=data).attrs['units'] = units
        u = h.create_dataset('velocity', (25, len(coords), 3), dtype='f8', compression='gzip', shuffle=True)
        p = h.create_dataset('pressure', (25, len(coords)), dtype='f8', compression='gzip', shuffle=True)
        u.attrs['units'], p.attrs['units'] = 'm/s', 'Pa'
        u.attrs['basis'], p.attrs['basis'] = 'P2 Lagrange tetrahedron', 'P1 Lagrange evaluated at P2 nodes'
        for k, (velocity, pressure) in enumerate(snapshots):
            u[k] = np.array([velocity(df.Point(*x)) for x in coords])
            p[k] = np.array([pressure(df.Point(*x)) for x in coords])
    # Standard visualization file per phase; native P2 connectivity is retained.
    import meshio
    for k, (velocity, pressure) in enumerate(snapshots):
        with h5py.File(temporary, 'r') as h:
            meshio.write(directory/f'phase_{k:02d}.vtu', meshio.Mesh(coords, [('tetra10', tetra10)],
                         point_data={'velocity': h['velocity'][k], 'pressure': h['pressure'][k]}))
    temporary.replace(directory/'fields.h5')


def read(path):
    """Return ordinary NumPy arrays and metadata, without importing the solver."""
    with h5py.File(path, 'r') as h:
        return dict(coordinates=h['mesh/coordinates'][:], tetra10=h['mesh/tetra10'][:],
                    tetra4=h['mesh/tetra4'][:], time=h['time'][:], phase=h['phase'][:],
                    velocity=h['velocity'][:], pressure=h['pressure'][:],
                    config=json.loads(h.attrs['config_json']))
