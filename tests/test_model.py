import copy
import numpy as np
import pytest
from scipy.integrate import quad
from cardiac_flow.model import load, flow, womersley, inlet, radius, pressure_gradient
from cardiac_flow.storage import quadratic_grid
from cardiac_flow.reference import ReferenceQuadrature


@pytest.mark.parametrize('name', ['straight', 'womersley'])
def test_analytical_flux_and_wall(name):
    c = load(f'configs/{name}.json')
    for t in np.linspace(0, c['period'], 9):
        q = quad(lambda r: 2*np.pi*r*womersley(r, t, c), 0, c['radius'], epsabs=1e-14)[0]
        assert q == pytest.approx(flow(t, c), rel=1e-10)
        assert abs(womersley(c['radius'], t, c)) < 1e-14


def test_analytical_pde():
    c = load('configs/womersley.json')
    r, t, dr, dt = .006, .137, 1e-6, 1e-6
    u = lambda rad, tm: womersley(rad, tm, c)
    ut = (u(r, t+dt)-u(r, t-dt))/(2*dt)
    lap = (u(r+dr, t)-2*u(r, t)+u(r-dr, t))/dr**2 + (u(r+dr, t)-u(r-dr, t))/(2*dr*r)
    assert c['rho']*ut == pytest.approx(pressure_gradient(t, c)+c['mu']*lap, rel=1e-5)


def test_eccentric_rotation_and_flux():
    c = load('configs/expansion.json')
    c['eccentricity'] = .7
    theta = .83
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated = dict(c, orientation=theta)
    x = np.array([.003, .005])
    assert inlet([*x, 0], .12, c) == pytest.approx(inlet([*(rot@x), 0], .12, rotated))
    n = 300
    rs = (np.arange(n)+.5)*c['radius']/n
    ts = (np.arange(n)+.5)*2*np.pi/n
    integral = sum(inlet([r*np.cos(t), r*np.sin(t), 0], .12, rotated)*r
                   for r in rs for t in ts)*c['radius']/n*2*np.pi/n
    assert integral == pytest.approx(flow(.12, c), rel=1e-4)


def test_expansion_smooth_ends():
    c = load('configs/expansion.json')
    assert radius(0, c) == pytest.approx(c['radius'])
    assert radius(c['length'], c) == pytest.approx(c['radius'])
    assert radius(c['length']/2, c) == pytest.approx(c['radius']*(1+c['expansion']))


def test_quadratic_connectivity():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    points, cells = quadratic_grid(vertices, np.array([[0, 1, 2, 3]]))
    assert cells.shape == (1, 10)
    assert np.allclose(points[cells[0, 4]], [.5, 0, 0])


def test_independent_reference_quadrature():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    quad = ReferenceQuadrature(vertices, np.array([[0, 1, 2, 3]]))
    assert quad.weights.sum() == pytest.approx(1/6)
    assert np.sum(quad.weights*quad.xyz[..., 0]**2) == pytest.approx(1/60)
    assert np.sum(quad.weights*quad.xyz[..., 0]*quad.xyz[..., 1]) == pytest.approx(1/120)
