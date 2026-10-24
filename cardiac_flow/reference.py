"""Vectorized independent volume quadrature for analytical benchmark errors."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from .storage import quadratic_grid, EDGES
from .model import womersley, pressure_gradient


class ReferenceQuadrature:
    def __init__(self, vertices, cells):
        self.nodes, self.cells = quadratic_grid(vertices, cells)
        g, w = leggauss(5)
        g, w = (g+1)/2, w/2
        a, b, c = np.meshgrid(g, g, g, indexing='ij')
        wa, wb, wc = np.meshgrid(w, w, w, indexing='ij')
        a, b, c = a.ravel(), b.ravel(), c.ravel()
        l = np.column_stack((1-a, a*(1-b), a*b*(1-c), a*b*c))
        self.basis = np.column_stack((l*(2*l-1), *[4*l[:, i]*l[:, j] for i, j in EDGES]))
        corners = vertices[cells]
        self.xyz = np.einsum('qi,cij->cqj', l, corners)
        det = np.abs(np.linalg.det(np.stack([corners[:, i]-corners[:, 0] for i in (1, 2, 3)], axis=-1)))
        self.weights = det[:, None]*(wa*wb*wc).ravel()[None, :]*(a*a*b)[None, :]

    def errors(self, velocity, pressure, t, config):
        import dolfin as df
        values = np.array([velocity(df.Point(*x)) for x in self.nodes])
        pvalues = np.array([pressure(df.Point(*x)) for x in self.nodes])
        uq = np.einsum('qi,cij->cqj', self.basis, values[self.cells])
        pq = np.einsum('qi,ci->cq', self.basis, pvalues[self.cells])
        exact = womersley(np.hypot(self.xyz[..., 0], self.xyz[..., 1]), t, config)
        diff2 = uq[..., 0]**2+uq[..., 1]**2+(uq[..., 2]-exact)**2
        uerror = np.sqrt(np.sum(self.weights*diff2)/np.sum(self.weights*exact**2))
        pexact = config.get('outlet_pressure', 0.0)+pressure_gradient(t, config)*(config['length']-self.xyz[..., 2])
        perror = np.sqrt(np.sum(self.weights*(pq-pexact)**2)/np.sum(self.weights))
        return float(uerror), float(perror)
