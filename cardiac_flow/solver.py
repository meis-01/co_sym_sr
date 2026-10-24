"""Serial, mixed P2/P1, backward-Euler/Picard incompressible Navier–Stokes."""
import json
import platform
import hashlib
import time
from pathlib import Path
import numpy as np
import dolfin as df
from .mesh import build
from .model import inlet, womersley, pressure_gradient, dimensionless


def run(c, output):
    if df.MPI.size(df.MPI.comm_world) != 1:
        raise RuntimeError('This exporter requires serial execution')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    (output/'config.json').write_text(json.dumps(c, indent=2))
    df.set_log_level(40)
    df.parameters['form_compiler']['quadrature_degree'] = 6
    mesh, tags = build(c, output)
    cell = mesh.ufl_cell()
    W = df.FunctionSpace(mesh, df.MixedElement([
        df.VectorElement('Lagrange', cell, 2), df.FiniteElement('Lagrange', cell, 1)]))
    V = W.sub(0).collapse()
    Q = W.sub(1).collapse()
    previous = df.Function(V)
    advector = df.Function(V)
    result = df.Function(W)
    initial_p = df.Function(Q)
    class Profile(df.UserExpression):
        t = 0.0
        def eval(self, values, x):
            values[:] = (0, 0, inlet(x, self.t, c))
        def value_shape(self):
            return (3,)
    profile = Profile(degree=2)
    if c['initial'] == 'analytical':
        previous.interpolate(profile)
        initial_p.interpolate(df.Expression('pout+g*(L-x[2])', pout=c.get('outlet_pressure', 0.0),
                                            g=pressure_gradient(0, c), L=c['length'], degree=1))
    elif c['initial'] == 'dataset':
        from .storage import read
        from .compare import evaluate
        data = read(c['initial_dataset'])
        for key in ('geometry', 'radius', 'length', 'expansion', 'rho', 'mu', 'period',
                    'mean_flow', 'harmonics', 'inlet', 'eccentricity', 'orientation'):
            if data['config'][key] != c[key]:
                raise ValueError(f'Warm-start parameter mismatch: {key}')
        for component in range(3):
            scalar_space = V.sub(component).collapse()
            field = df.Function(scalar_space)
            values = evaluate(data, scalar_space.tabulate_dof_coordinates())[0, :, component]
            if not np.all(np.isfinite(values)):
                raise ValueError('Warm-start mesh does not cover target velocity nodes')
            field.vector().set_local(values)
            field.vector().apply('insert')
            df.assign(previous.sub(component), field)
        values = evaluate(data, Q.tabulate_dof_coordinates(), field='pressure')[0]
        if not np.all(np.isfinite(values)):
            raise ValueError('Warm-start mesh does not cover target pressure nodes')
        initial_p.vector().set_local(values)
        initial_p.vector().apply('insert')
    assign = df.FunctionAssigner(W, [V, Q])
    assign.assign(result, [previous, initial_p])
    # Wall condition wins at the circular inlet rim; both traces are zero there.
    bcs = [df.DirichletBC(W.sub(0), profile, tags, 1),
           df.DirichletBC(W.sub(0), df.Constant((0., 0., 0.)), tags, 3)]
    u, p = df.TrialFunctions(W)
    v, q = df.TestFunctions(W)
    dt = c['period']/c['steps_per_cycle']
    rho, mu = df.Constant(c['rho']), df.Constant(c['mu'])
    n = df.FacetNormal(mesh)
    ds = df.Measure('ds', domain=mesh, subdomain_data=tags)
    dx = df.Measure('dx', domain=mesh)
    # Laplacian form gives pseudo-traction mu grad(u)n - p n = 0.
    # Energy-stable backflow penalty is active only for u_adv.n < 0.
    negative_normal = (abs(df.dot(advector, n))-df.dot(advector, n))/2
    a = (rho/dt*df.inner(u, v) + mu*df.inner(df.grad(u), df.grad(v))
         + rho*df.inner(df.dot(df.grad(u), advector), v)
         - p*df.div(v) + q*df.div(u))*dx
    # Consistent grad-div stabilization; exact incompressible fields add zero.
    a += df.Constant(c.get('grad_div_ratio', 0.0)*c['mu'])*df.div(u)*df.div(v)*dx
    a += rho/2*negative_normal*df.inner(u, v)*ds(2)
    L = rho/dt*df.inner(previous, v)*dx - df.Constant(c.get('outlet_pressure', 0.0))*df.dot(n, v)*ds(2)
    volume = df.assemble(df.Constant(1)*dx)
    speed = c['mean_flow']/(np.pi*c['radius']**2)
    pscale = max(c['rho']*speed**2, c['mu']*speed*c['length']/c['radius']**2)
    diagnostics = []
    last_cycle = None
    started = time.time()
    if c['benchmark']:
        from .reference import ReferenceQuadrature
        reference = ReferenceQuadrature(mesh.coordinates(), mesh.cells())

    def measures(velocity, pressure, t, iteration_count, benchmark=False):
        qi = -df.assemble(df.dot(velocity, n)*ds(1))
        qo = df.assemble(df.dot(velocity, n)*ds(2))
        div_rms = np.sqrt(df.assemble(df.div(velocity)**2*dx)/volume)
        wall_area = df.assemble(df.Constant(1)*ds(3))
        wall = np.sqrt(df.assemble(df.inner(velocity, velocity)*ds(3))/wall_area)
        row = dict(time=t, inlet_flow=qi, outlet_flow=qo,
                   flux_error=abs(qo-qi)/c['mean_flow'],
                   divergence_scaled=div_rms*c['radius']/speed,
                   wall_rms=wall, max_speed=float(np.max(np.abs(velocity.vector().get_local()))),
                   picard_iterations=iteration_count)
        if benchmark and c['benchmark']:
            ue, pe = reference.errors(velocity, pressure, t, c)
            row['velocity_relative_l2'] = ue
            row['pressure_error_scaled'] = pe/pscale
        return row

    try:
        for cycle in range(1, c['max_cycles']+1):
            snapshots = []
            phase_rows = []
            phase_stride = c['steps_per_cycle']//25
            for step in range(c['steps_per_cycle']):
                t0 = ((cycle-1)*c['steps_per_cycle']+step)*dt
                if step % phase_stride == 0:
                    velocity, pressure = result.split(deepcopy=True)
                    snapshots.append((velocity, pressure))
                    phase_rows.append(measures(velocity, pressure, t0, 0, benchmark=True))
                profile.t = t0+dt
                advector.assign(previous)
                converged = False
                for iteration in range(20):
                    df.solve(a == L, result, bcs, solver_parameters={'linear_solver': 'mumps'})
                    velocity, pressure = result.split(deepcopy=True)
                    delta = np.sqrt(df.assemble(df.inner(velocity-advector, velocity-advector)*dx))
                    norm = max(np.sqrt(df.assemble(df.inner(velocity, velocity)*dx)), speed*np.sqrt(volume)*1e-8)
                    advector.assign(velocity)
                    if delta/norm < 1e-7:
                        converged = True
                        break
                if not converged or not np.all(np.isfinite(result.vector().get_local())):
                    raise RuntimeError(f'Nonlinear solve failed at t={t0+dt}')
                previous.assign(velocity)
                row = measures(velocity, pressure, t0+dt, iteration+1)
                if row['flux_error'] > 0.01 or row['wall_rms'] > 1e-8:
                    raise RuntimeError(f'Physical validation failed: {row}')
                diagnostics.append(row)
            periodic_u = periodic_p = float('inf')
            if last_cycle is not None:
                du = sum(df.assemble(df.inner(s[0]-old[0], s[0]-old[0])*dx)
                         for s, old in zip(snapshots, last_cycle))
                un = sum(df.assemble(df.inner(s[0], s[0])*dx) for s in snapshots)
                dp = sum(df.assemble((s[1]-old[1])**2*dx) for s, old in zip(snapshots, last_cycle))
                pn = sum(df.assemble(s[1]**2*dx) for s in snapshots)
                periodic_u = np.sqrt(du/max(un, 1e-30))
                periodic_p = np.sqrt(dp/max(pn, volume*pscale**2*1e-12))
            print(f'cycle={cycle} velocity_periodic={periodic_u:.4g} pressure_periodic={periodic_p:.4g} elapsed={time.time()-started:.1f}s', flush=True)
            last_cycle = snapshots
            if cycle >= c['min_cycles'] and max(periodic_u, periodic_p) < c['periodic_tolerance']:
                from .storage import export
                report = dict(status='periodic_converged', cycles=cycle,
                              periodic_velocity=float(periodic_u), periodic_pressure=float(periodic_p),
                              dimensionless=dimensionless(c), vertices=mesh.num_vertices(), cells=mesh.num_cells(),
                              mixed_dofs=W.dim(), wall_seconds=time.time()-started,
                              dolfin=df.__version__, python=platform.python_version(),
                              max_flux_error=max(r['flux_error'] for r in diagnostics),
                              max_wall_rms=max(r['wall_rms'] for r in diagnostics),
                              max_divergence_scaled=max(r['divergence_scaled'] for r in diagnostics),
                              final_cycle_max_divergence_scaled=max(r['divergence_scaled'] for r in phase_rows),
                              phases=phase_rows)
                report['numerics'] = dict(method='P2/P1 Taylor-Hood; backward Euler; Picard',
                    dt=dt, linear_solver='PETSc MUMPS LU', nonlinear_tolerance=1e-7,
                    max_nonlinear_iterations=20, quadrature_degree=6,
                    outlet='mu grad(u)n - p n + rho/2 max(-u_adv.n,0)u = -p_out n',
                    outlet_pressure_Pa=c.get('outlet_pressure', 0.0),
                    pressure_reference='outlet pseudo-traction at configured constant gauge pressure',
                    spatial_accuracy='piecewise planar tetrahedra; quadratic velocity; linear pressure',
                    temporal_accuracy='first order')
                report['numerics']['grad_div_ratio'] = c.get('grad_div_ratio', 0.0)
                report['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                           for p in Path(__file__).parent.glob('*.py')}
                (output/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
                export(output, mesh, tags, snapshots, c, cycle, report)
                (output/'report.json').write_text(json.dumps(report, indent=2))
                return report
        raise RuntimeError('Periodic convergence not reached; no final dataset exported')
    except Exception as error:
        (output/'failure.json').write_text(json.dumps(dict(status='failed', reason=str(error)), indent=2))
        (output/'diagnostics.json').write_text(json.dumps(diagnostics, indent=2))
        raise
