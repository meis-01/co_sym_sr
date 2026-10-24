from pathlib import Path
import json
import pytest
import numpy as np
from cardiac_flow.analysis import validate
from cardiac_flow.storage import read


@pytest.mark.parametrize('name', ['poiseuille', 'womersley', 'womersley_low_alpha', 'expansion', 'expansion_stabilized', 'expansion_strict'])
def test_completed_datasets(name):
    path = Path('results')/name/'fields.h5'
    if not path.exists() or not path.with_name('report.json').exists():
        pytest.skip('Generate the integration dataset first')
    assert validate(path)['status'] == 'passed'
    report = json.loads(path.with_name('report.json').read_text())
    if name == 'poiseuille':
        assert max(r['velocity_relative_l2'] for r in report['phases']) < .12
    if name.startswith('womersley'):
        assert max(r['velocity_relative_l2'] for r in report['phases']) < .20
    if name == 'expansion_strict':
        assert validate(path)['local_divergence_below_0_1']
    if name.startswith('expansion'):
        velocity = read(path)['velocity']
        assert np.max(np.abs(velocity[..., 0])) > 1e-7
        assert np.max(np.abs(velocity[..., 1])) > 1e-7
