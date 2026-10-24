"""Build a concise, evidence-only inventory of completed and failed runs."""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cardiac_flow.analysis import validate


def main():
    rows = ['# Executed CFD results', '',
            'Only directories with a final dataset and report appear in the completed table.',
            'Errors below are numerical measurements, not claims of physiological fidelity.', '',
            '| Run | Cells | Cycles | Periodic velocity | Final scaled divergence | Max analytical velocity L2 error |',
            '|---|---:|---:|---:|---:|---:|']
    completed = []
    for root in ('results', 'studies'):
        for file in sorted(Path(root).rglob('report.json')):
            dataset = file.with_name('fields.h5')
            if not dataset.exists():
                continue
            r = json.loads(file.read_text())
            v = validate(dataset)
            ae = max((p.get('velocity_relative_l2', 0) for p in r['phases']), default=0)
            analytical = f'{100*ae:.3f}%' if ae else 'n/a'
            rows.append(f"| `{file.parent.as_posix()}` | {r['cells']} | {r['cycles']} | {r['periodic_velocity']:.4g} | {v['final_cycle_max_divergence_scaled']:.4g} | {analytical} |")
            completed.append(dict(directory=file.parent.as_posix(), validation=v))
    rows += ['', 'The initial unstabilized expansion is a coarse diagnostic dataset. Its local divergence',
             'requires refinement/stabilization despite excellent global flux balance.',
             'All datasets retain `accuracy_certified: false`; use study results and application-specific tolerances.',
             'The ratio-1000 expansion missed the 0.1 local-divergence check (0.1016). The ratio-10000',
             'follow-up passed (0.0866), but changes velocity by 11.4% and pressure by 30.5% relative to',
             'ratio 1000 on the interior comparison nodes. This sensitivity requires finer aortic meshes;',
             'passing the divergence threshold alone does not make these resolution-independent training fields.', '']
    for study in ('convergence', 'rotation', 'stabilization'):
        file = Path('studies')/study/'comparison.json'
        if file.exists():
            rows += [f'## {study.capitalize()} study', '', '```json', file.read_text(), '```', '']
    rows += ['## Interrupted or failed runs', '']
    for file in sorted(Path('results').rglob('failure.json')):
        r = json.loads(file.read_text())
        rows.append(f"- `{file.parent.as_posix()}`: {r['status']}: {r['reason']}")
    Path('RESULTS.md').write_text('\n'.join(rows)+'\n')
    Path('results/inventory.json').write_text(json.dumps(completed, indent=2))
    print(f'{len(completed)} completed datasets inventoried')


if __name__ == '__main__':
    main()
