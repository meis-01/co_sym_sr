"""Plot executed convergence evidence, never configuration-only cases."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('studies/convergence')
result = json.loads((root/'comparison.json').read_text())
names = ['base', 'mesh_medium', 'mesh_fine']
h = [json.loads((root/f'{name}.json').read_text())['mesh_size']*1000 for name in names]
errors = [result[name]['analytical_velocity_error']*100 for name in names]
fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
axes[0].loglog(h, errors, 'o-')
axes[0].set(xlabel='Mesh size [mm]', ylabel='Maximum velocity L2 error [%]', title='Womersley spatial error')
axes[0].grid(True, which='both', alpha=.3)
period = json.loads((root/'base.json').read_text())['period']
dt = [period/100, period/150]
t = result['temporal_to_finest']
axes[1].loglog(dt, [100*t['coarse'], 100*t['medium']], 'o-', label='velocity')
axes[1].loglog(dt, [100*t['pressure_coarse'], 100*t['pressure_medium']], 's-', label='pressure')
axes[1].set(xlabel='Time step [s]', ylabel='Difference to 200-step solution [%]', title='Temporal refinement')
axes[1].legend()
axes[1].grid(True, which='both', alpha=.3)
fig.savefig(root/'convergence.png', dpi=160)
plt.close(fig)
