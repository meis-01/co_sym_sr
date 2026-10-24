"""Configuration and independent analytical references (no CFD dependency)."""
import json
from pathlib import Path
import numpy as np
from scipy.special import jv


def load(path):
    c = json.loads(Path(path).read_text())
    for key in ('radius', 'length', 'mesh_size', 'rho', 'mu', 'period', 'mean_flow'):
        if not np.isfinite(c[key]) or c[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if c['geometry'] not in ('straight', 'expansion'):
        raise ValueError('unknown geometry')
    if not 0 <= c['expansion'] <= 1 or (c['geometry'] == 'straight' and c['expansion'] != 0):
        raise ValueError('expansion must be in [0,1], and zero for a straight tube')
    if not 0 <= c['eccentricity'] < 1:
        raise ValueError('eccentricity must be in [0,1)')
    if c['inlet'] not in ('parabolic', 'womersley'):
        raise ValueError('unknown inlet')
    if c['inlet'] == 'womersley' and c['eccentricity']:
        raise ValueError('eccentric Womersley inlet is not supported')
    if c['initial'] not in ('zero', 'analytical', 'dataset'):
        raise ValueError('unknown initialization')
    if c['initial'] == 'dataset' and not c.get('initial_dataset'):
        raise ValueError('dataset initialization requires initial_dataset')
    if c['initial'] == 'analytical' and (c['geometry'] != 'straight' or c['inlet'] != 'womersley'):
        raise ValueError('analytical initialization requires a compatible straight tube')
    if c['benchmark'] and (c['geometry'] != 'straight' or c['inlet'] != 'womersley'):
        raise ValueError('benchmark requires a compatible straight tube')
    n = c['steps_per_cycle']
    if not isinstance(n, int) or n < 50 or n % 25:
        raise ValueError('steps_per_cycle must be an integer multiple of 25, >= 50')
    if not 2 <= c['min_cycles'] <= c['max_cycles']:
        raise ValueError('require 2 <= min_cycles <= max_cycles')
    if not 0 < c['periodic_tolerance'] < 1:
        raise ValueError('invalid periodic tolerance')
    if not np.isfinite(c.get('grad_div_ratio', 0.0)) or c.get('grad_div_ratio', 0.0) < 0:
        raise ValueError('grad_div_ratio must be finite and nonnegative')
    if not np.isfinite(c['orientation']):
        raise ValueError('invalid orientation')
    if not np.isfinite(c.get('outlet_pressure', 0.0)):
        raise ValueError('invalid outlet pressure')
    if any(len(h) != 2 or not np.all(np.isfinite(h)) for h in c['harmonics']):
        raise ValueError('harmonics are [relative amplitude, phase radians]')
    return c


def dimensionless(c):
    r = c['radius']
    return dict(Re=2*c['rho']*c['mean_flow']/(np.pi*r*c['mu']),
                alpha=r*np.sqrt(2*np.pi*c['rho']/(c['period']*c['mu'])))


def radius(z, c):
    return c['radius']*(1+c['expansion']*np.sin(np.pi*np.asarray(z)/c['length'])**2)


def flow(t, c):
    return c['mean_flow']*(1+sum(a*np.cos(2*np.pi*k*t/c['period']+phi)
                               for k, (a, phi) in enumerate(c['harmonics'], 1)))


def womersley(r, t, c):
    """Fully developed axial velocity for a prescribed Fourier volume flow."""
    R = c['radius']
    s = np.asarray(r)/R
    u = 2*c['mean_flow']/(np.pi*R**2)*(1-s*s)
    alpha = dimensionless(c)['alpha']
    for k, (a, phi) in enumerate(c['harmonics'], 1):
        lam = (1j**1.5)*alpha*np.sqrt(k)
        shape = (1-jv(0, lam*s)/jv(0, lam))/(1-2*jv(1, lam)/(lam*jv(0, lam)))
        u = u + np.real(c['mean_flow']*a/(np.pi*R**2)*shape*
                        np.exp(1j*(2*np.pi*k*t/c['period']+phi)))
    return u


def pressure_gradient(t, c):
    """Positive driving gradient -dp/dz, Pa/m."""
    R = c['radius']
    value = 8*c['mu']*c['mean_flow']/(np.pi*R**4)
    for k, (a, phi) in enumerate(c['harmonics'], 1):
        omega = 2*np.pi*k/c['period']
        lam = (1j**1.5)*dimensionless(c)['alpha']*np.sqrt(k)
        value += np.real(1j*omega*c['rho']*c['mean_flow']*a/(np.pi*R**2)/
                         (1-2*jv(1, lam)/(lam*jv(0, lam)))*np.exp(1j*(omega*t+phi)))
    return value


def inlet(x, t, c):
    r = np.hypot(x[0], x[1])
    if c['inlet'] == 'womersley':
        return float(womersley(r, t, c))
    s2 = (r/c['radius'])**2
    tilt = (x[0]*np.cos(c['orientation'])+x[1]*np.sin(c['orientation']))/c['radius']
    # The odd tilt integrates to zero on a disk; total flux remains Q(t).
    return 2*flow(t, c)/(np.pi*c['radius']**2)*max(0, 1-s2)*(1+c['eccentricity']*tilt)
