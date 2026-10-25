"""Real CFD assembly / block-solver regression; skip without the native stack."""
import json

import pytest

pytest.importorskip('dolfin')
pytest.importorskip('petsc4py')

from scripts.benchmark_linear_solvers import benchmark


def test_schur_matches_production_mumps_on_oseen_system(tmp_path):
    report = benchmark('configs/cluster-smoke.json', tmp_path/'benchmark',
                       systems=1, repeats=1, expected_cells=184,
                       methods=('mumps', 'schur_ilu_lu', 'schur_ilu_ilu'))
    assert report['mixed_dofs'] == 1364
    assert report['systems'][0]['original_df_solve_relative_difference'] < 1e-12
    assert all(method['validated'] for method in report['summary'].values())
    saved = json.loads((tmp_path/'benchmark/linear_benchmark.json').read_text())
    assert saved['summary'] == report['summary']
    assert not (tmp_path/'benchmark/fields.h5').exists()
    assert not (tmp_path/'benchmark/failure.json').exists()


def test_benchmark_refuses_fine_mesh_before_creating_output(tmp_path):
    with pytest.raises(ValueError, match='forbids meshes finer'):
        benchmark('configs/expansion.json', tmp_path/'forbidden', systems=1, repeats=1)
    assert not (tmp_path/'forbidden').exists()
