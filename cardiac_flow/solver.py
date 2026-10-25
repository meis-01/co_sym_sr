"""Serial, mixed P2/P1, backward-Euler/Picard incompressible Navier–Stokes."""
import json
import platform
import hashlib
import time
from pathlib import Path
import numpy as np
import dolfin as df
from .mesh import build
from .model import inlet, womersley, pressure_gradient, dimensionless, flow


_FIELDSPLIT_MAX_IT = 300
_FIELDSPLIT_OPTIONS = dict(
    ksp_type='fgmres', ksp_pc_side='right', ksp_gmres_restart=100,
    pc_type='fieldsplit', pc_fieldsplit_type='schur',
    pc_fieldsplit_schur_fact_type='full', pc_fieldsplit_schur_precondition='selfp',
    fieldsplit_u_ksp_type='preonly', fieldsplit_u_pc_type='ilu',
    fieldsplit_u_pc_factor_levels=0,
    fieldsplit_p_ksp_type='preonly', fieldsplit_p_pc_type='lu',
    fieldsplit_p_pc_factor_mat_solver_type='mumps')


class _FieldSplitSolver:
    """Opt-in serial Schur solver; fresh preconditioner for each Picard matrix."""

    def __init__(self, W):
        from petsc4py import PETSc
        self.PETSc = PETSc
        fields = [np.asarray(W.sub(i).dofmap().dofs(), dtype=PETSc.IntType) for i in (0, 1)]
        if not np.array_equal(np.sort(np.concatenate(fields)), np.arange(W.dim())):
            raise RuntimeError('Velocity/pressure indices do not partition the mixed space')
        self.index_sets = [PETSc.IS().createGeneral(f, comm=PETSc.COMM_SELF) for f in fields]

    def close(self):
        for index_set in self.index_sets:
            index_set.destroy()

    def solve(self, a, L, result, bcs):
        PETSc = self.PETSc
        start = time.perf_counter()
        A_df, b_df = df.assemble(a), df.assemble(L)
        for bc in bcs:
            bc.apply(A_df, b_df)
        assembly_seconds = time.perf_counter()-start
        A, b = df.as_backend_type(A_df).mat(), df.as_backend_type(b_df).vec()
        prefix = 'cardiac_flow_fieldsplit_'
        opts = PETSc.Options()
        previous_options = opts.getAll()
        start = time.perf_counter()
        ksp = PETSc.KSP().create(PETSc.COMM_SELF)
        x = A.createVecRight()
        x.set(0)
        try:
            for key, value in _FIELDSPLIT_OPTIONS.items():
                opts[prefix+key] = value
            ksp.setOptionsPrefix(prefix)
            ksp.setOperators(A)
            ksp.getPC().setType('fieldsplit')
            ksp.getPC().setFieldSplitIS(('u', self.index_sets[0]), ('p', self.index_sets[1]))
            ksp.setFromOptions()
            ksp.setTolerances(rtol=1e-14, atol=0.0, divtol=1e8, max_it=_FIELDSPLIT_MAX_IT)
            ksp.setNormType(PETSc.KSP.NormType.UNPRECONDITIONED)
            ksp.setInitialGuessNonzero(False)
            ksp.setUp()
            ksp.solve(b, x)
            solver_seconds = time.perf_counter()-start
            reason = int(ksp.getConvergedReason())
            iterations = ksp.getIterationNumber()
            residual = b.duplicate()
            try:
                A.mult(x, residual)
                residual.axpy(-1, b)
                bnorm = b.norm()
                absolute_residual = residual.norm()
                relative_residual = absolute_residual/bnorm if bnorm else absolute_residual
            finally:
                residual.destroy()
            values = x.getArray(readonly=True)
            if (reason <= 0 or not np.all(np.isfinite(values))
                    or not np.isfinite(relative_residual) or relative_residual > 1e-12):
                raise RuntimeError(f'FieldSplit failed: reason={reason}, iterations={iterations}, '
                                   f'true_relative_residual={relative_residual}')
            result.vector().set_local(values)
            result.vector().apply('insert')
            return dict(assembly_seconds=assembly_seconds, solver_seconds=solver_seconds,
                        iterations=iterations, reason=reason,
                        true_relative_residual=relative_residual)
        except PETSc.Error as error:
            raise RuntimeError(f'FieldSplit PETSc failure: {error}') from error
        finally:
            ksp.destroy()
            x.destroy()
            for key in _FIELDSPLIT_OPTIONS:
                full_key = prefix+key
                if full_key in previous_options:
                    opts[full_key] = previous_options[full_key]
                else:
                    del opts[full_key]


def _lu_seconds():
    """DOLFIN's existing LU timer includes factorization/solve, excludes assembly."""
    try:
        return df.timing('LU solver', df.TimingClear.keep)[1]
    except RuntimeError:  # No LU solve in this process yet.
        return 0.0


def run(c, output):
    setup_started = time.perf_counter()
    if df.MPI.size(df.MPI.comm_world) != 1:
        raise RuntimeError('This exporter requires serial execution')
    linear_solver = c.get('linear_solver', 'mumps')
    if linear_solver not in ('mumps', 'fieldsplit'):
        raise ValueError('linear_solver must be mumps or fieldsplit')
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
    cycle_history = []
    last_cycle = None
    split_solver = _FieldSplitSolver(W) if linear_solver == 'fieldsplit' else None
    lu_started = _lu_seconds() if split_solver is None else 0.0
    linear_system_seconds = assembly_seconds = solver_seconds = 0.0
    linear_solves = total_ksp_iterations = max_ksp_iterations = 0
    max_true_residual = 0.0
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
                   inlet_flow_error=abs(qi-flow(t, c))/c['mean_flow'],
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
                    linear_started = time.perf_counter()
                    if split_solver is None:
                        # Preserve the original high-level MUMPS path and its defaults.
                        df.solve(a == L, result, bcs, solver_parameters={'linear_solver': 'mumps'})
                    else:
                        linear = split_solver.solve(a, L, result, bcs)
                        assembly_seconds += linear['assembly_seconds']
                        solver_seconds += linear['solver_seconds']
                        total_ksp_iterations += linear['iterations']
                        max_ksp_iterations = max(max_ksp_iterations, linear['iterations'])
                        max_true_residual = max(max_true_residual, linear['true_relative_residual'])
                    linear_system_seconds += time.perf_counter()-linear_started
                    linear_solves += 1
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
            cycle_history.append(dict(cycle=cycle,
                periodic_velocity=float(periodic_u) if np.isfinite(periodic_u) else None,
                periodic_pressure=float(periodic_p) if np.isfinite(periodic_p) else None))
            last_cycle = snapshots
            if cycle >= c['min_cycles'] and max(periodic_u, periodic_p) < c['periodic_tolerance']:
                from .storage import export
                report = dict(status='periodic_converged', cycles=cycle,
                              periodic_velocity=float(periodic_u), periodic_pressure=float(periodic_p),
                              dimensionless=dimensionless(c), vertices=mesh.num_vertices(), cells=mesh.num_cells(),
                              mixed_dofs=W.dim(), wall_seconds=time.time()-started,
                              setup_and_simulation_seconds=time.perf_counter()-setup_started,
                              cycle_history=cycle_history,
                              dolfin=df.__version__, python=platform.python_version(),
                              max_flux_error=max(r['flux_error'] for r in diagnostics),
                              max_inlet_flow_error=max(r['inlet_flow_error'] for r in diagnostics),
                              max_wall_rms=max(r['wall_rms'] for r in diagnostics),
                              max_divergence_scaled=max(r['divergence_scaled'] for r in diagnostics),
                              final_cycle_max_divergence_scaled=max(r['divergence_scaled'] for r in phase_rows),
                              phases=phase_rows)
                report['numerics'] = dict(method='P2/P1 Taylor-Hood; backward Euler; Picard',
                    dt=dt, linear_solver=('PETSc MUMPS LU' if split_solver is None else
                        'PETSc FGMRES FieldSplit Schur full/selfp; velocity ILU(0); pressure MUMPS LU'),
                    linear_solver_key=linear_solver, linear_solver_experimental=split_solver is not None,
                    nonlinear_tolerance=1e-7,
                    max_nonlinear_iterations=20, quadrature_degree=6,
                    outlet='mu grad(u)n - p n + rho/2 max(-u_adv.n,0)u = -p_out n',
                    outlet_pressure_Pa=c.get('outlet_pressure', 0.0),
                    pressure_reference='outlet pseudo-traction at configured constant gauge pressure',
                    spatial_accuracy='piecewise planar tetrahedra; quadratic velocity; linear pressure',
                    temporal_accuracy='first order')
                report['numerics']['grad_div_ratio'] = c.get('grad_div_ratio', 0.0)
                report['linear_solver_statistics'] = dict(solves=linear_solves,
                    linear_system_seconds=linear_system_seconds,
                    solver_seconds=solver_seconds if split_solver else _lu_seconds()-lu_started,
                    solver_timer=('perf_counter: KSP creation/setup/factorization/solve' if split_solver else
                                  'DOLFIN LU solver wall timer: factorization/solve'),
                    assembly_seconds=assembly_seconds if split_solver else None)
                if split_solver:
                    report['numerics']['linear_solver_options'] = dict(_FIELDSPLIT_OPTIONS,
                        ksp_rtol=1e-14, ksp_atol=0.0, ksp_divtol=1e8,
                        ksp_max_it=_FIELDSPLIT_MAX_IT, ksp_norm_type='unpreconditioned',
                        true_relative_residual_limit=1e-12)
                    report['linear_solver_statistics'].update(total_iterations=total_ksp_iterations,
                        max_iterations=max_ksp_iterations, max_true_relative_residual=max_true_residual)
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
    finally:
        if split_solver is not None:
            split_solver.close()
