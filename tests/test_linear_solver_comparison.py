"""Check the FE field comparison used to accept the five-cycle experiment."""
import numpy as np
import pytest

pytest.importorskip('petsc4py')
pytest.importorskip('dolfin')

from cardiac_flow.storage import quadratic_grid
from scripts.validate_linear_solvers import field_errors


def test_fe_comparison_detects_relative_velocity_and_pressure_change():
    vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1, 0], [0., 0, 1]])
    tetra = np.array([[0, 1, 2, 3]])
    xyz, tetra10 = quadratic_grid(vertices, tetra)
    velocity = np.broadcast_to(1+xyz**2, (25, len(xyz), 3)).copy()
    pressure = np.broadcast_to(1+xyz[:, 0], (25, len(xyz))).copy()
    a = dict(coordinates=xyz, tetra4=tetra, tetra10=tetra10, time=np.arange(25),
             velocity=velocity, pressure=pressure)
    b = dict(a, velocity=velocity*1.01, pressure=pressure*1.02)
    errors = field_errors(a, b)
    assert errors['velocity']['relative_space_time_l2'] == pytest.approx(.01)
    assert errors['velocity']['max_phase_relative_l2'] == pytest.approx(.01)
    assert errors['pressure']['relative_space_time_l2'] == pytest.approx(.02)
    assert errors['pressure']['max_phase_relative_l2'] == pytest.approx(.02)
    with pytest.raises(ValueError, match='identical coordinates'):
        field_errors(a, dict(b, coordinates=xyz*2))
