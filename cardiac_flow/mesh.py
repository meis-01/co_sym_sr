"""Straight reference mesh mapped to a smooth axisymmetric expansion."""
from pathlib import Path
import numpy as np


def build(c, directory):
    import gmsh
    import meshio
    import dolfin as df
    from .model import radius
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    gmsh.initialize()
    try:
        gmsh.option.setNumber('General.Terminal', 0)
        gmsh.option.setNumber('General.NumThreads', 1)
        gmsh.model.add('tube')
        gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, c['length'], c['radius'])
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber('Mesh.MeshSizeMin', c['mesh_size'])
        gmsh.option.setNumber('Mesh.MeshSizeMax', c['mesh_size'])
        gmsh.model.mesh.generate(3)
        gmsh.write(str(directory/'mesh.msh'))
    finally:
        gmsh.finalize()
    raw = meshio.read(directory/'mesh.msh')
    points = raw.points.copy()
    points[:, :2] *= (radius(points[:, 2], c)/c['radius'])[:, None]
    cells = raw.get_cells_type('tetra')
    meshio.write(directory/'mesh.xml', meshio.Mesh(points, [('tetra', cells)]))
    mesh = df.Mesh(str(directory/'mesh.xml'))
    facets = df.MeshFunction('size_t', mesh, 2, 0)
    class Wall(df.SubDomain):
        def inside(self, x, boundary):
            return boundary
    class Inlet(df.SubDomain):
        def inside(self, x, boundary):
            return boundary and df.near(x[2], 0)
    class Outlet(df.SubDomain):
        def inside(self, x, boundary):
            return boundary and df.near(x[2], c['length'])
    Wall().mark(facets, 3)
    Inlet().mark(facets, 1)
    Outlet().mark(facets, 2)
    return mesh, facets
