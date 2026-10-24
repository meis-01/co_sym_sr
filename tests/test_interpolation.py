import numpy as np
from cardiac_flow.storage import quadratic_grid
from cardiac_flow.compare import evaluate
from cardiac_flow.analysis import divergence_rms


def test_quadratic_field_evaluation():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tet = np.array([[0, 1, 2, 3]])
    points, conn = quadratic_grid(vertices, tet)
    f = lambda x: np.column_stack((x[:, 0]**2+2*x[:, 1]*x[:, 2], x[:, 1], x[:, 2]**2))
    d = dict(coordinates=points, tetra4=tet, tetra10=conn, velocity=np.repeat(f(points)[None], 25, axis=0))
    target = np.array([[.1, .2, .3], [.25, .1, .05]])
    actual = evaluate(d, target)
    assert np.allclose(actual, f(target)[None])


def test_exported_divergence():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    tet = np.array([[0, 1, 2, 3]])
    points, conn = quadratic_grid(vertices, tet)
    # div(x^2, -2xy, z) = 1 everywhere: tests quadratic cross derivatives.
    values = np.column_stack((points[:, 0]**2, -2*points[:, 0]*points[:, 1], points[:, 2]))
    data = dict(coordinates=points, tetra4=tet, tetra10=conn, velocity=np.repeat(values[None], 25, axis=0))
    assert np.allclose(divergence_rms(data), 1.)
