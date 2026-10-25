"""Exercise the actual opt-in solve branch, including convergence failures."""
import json

import numpy as np
import pytest

df = pytest.importorskip('dolfin')
PETSc = pytest.importorskip('petsc4py.PETSc')

from cardiac_flow import solver
from cardiac_flow.model import load


class _Stopped(BaseException):
    pass


def test_production_fieldsplit_matches_mumps_and_restores_options(tmp_path, monkeypatch):
    c = dict(load('configs/cluster-smoke.json'), linear_solver='fieldsplit')
    original = solver._FieldSplitSolver.solve
    calls = []
    opts = PETSc.Options()
    option = 'cardiac_flow_fieldsplit_pc_type'
    opts[option] = 'none'

    def compare(self, a, L, result, bcs):
        if len(calls) == 3:
            raise _Stopped()
        info = original(self, a, L, result, bcs)
        assert info['reason'] > 0
        assert info['true_relative_residual'] < 1e-12
        reference = df.Function(result.function_space())
        df.solve(a == L, reference, bcs, solver_parameters={'linear_solver': 'mumps'})
        for actual, expected in zip(result.split(deepcopy=True), reference.split(deepcopy=True)):
            x, y = actual.vector().get_local(), expected.vector().get_local()
            assert np.linalg.norm(x-y)/np.linalg.norm(y) < 1e-8
        assert opts[option] == 'none'
        calls.append(info)
        return info

    monkeypatch.setattr(solver._FieldSplitSolver, 'solve', compare)
    try:
        with pytest.raises(_Stopped):
            solver.run(c, tmp_path/'fields')
    finally:
        del opts[option]
    assert len(calls) == 3
    assert not (tmp_path/'fields/fields.h5').exists()


def test_default_keeps_original_mumps_call(tmp_path, monkeypatch):
    def stop(*args, **kwargs):
        assert kwargs == {'solver_parameters': {'linear_solver': 'mumps'}}
        raise _Stopped()

    monkeypatch.setattr(df, 'solve', stop)
    with pytest.raises(_Stopped):
        solver.run(load('configs/cluster-smoke.json'), tmp_path/'default')


def test_fieldsplit_failure_is_reported_without_fallback(tmp_path, monkeypatch):
    c = dict(load('configs/cluster-smoke.json'), linear_solver='fieldsplit')
    monkeypatch.setattr(solver, '_FIELDSPLIT_MAX_IT', 1)
    with pytest.raises(RuntimeError, match='FieldSplit failed'):
        solver.run(c, tmp_path/'failed')
    failure = json.loads((tmp_path/'failed/failure.json').read_text())
    assert 'FieldSplit failed' in failure['reason']
    assert not (tmp_path/'failed/fields.h5').exists()


def test_solver_selection_is_validated(tmp_path):
    config = load('configs/cluster-smoke.json')
    assert 'linear_solver' not in config  # Existing configs keep their default.
    path = tmp_path/'config.json'
    for method in ('mumps', 'fieldsplit'):
        path.write_text(json.dumps(dict(config, linear_solver=method)))
        assert load(path)['linear_solver'] == method
    path.write_text(json.dumps(dict(config, linear_solver='unknown')))
    with pytest.raises(ValueError, match='linear_solver'):
        load(path)
    with pytest.raises(ValueError, match='linear_solver'):
        solver.run(dict(config, linear_solver='unknown'), tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
