import argparse
import json
from .model import load

parser = argparse.ArgumentParser(description='Pulsatile 3D CFD dataset pipeline')
sub = parser.add_subparsers(dest='command', required=True)
run = sub.add_parser('run')
run.add_argument('config')
run.add_argument('output')
plot = sub.add_parser('plot')
plot.add_argument('dataset')
validate = sub.add_parser('validate')
validate.add_argument('dataset')
compare_parser = sub.add_parser('compare')
compare_parser.add_argument('reference')
compare_parser.add_argument('candidate')
compare_parser.add_argument('--angle', type=float, default=0.0)
args = parser.parse_args()
if args.command == 'run':
    from .solver import run
    report = run(load(args.config), args.output)
    keys = ('status', 'cycles', 'periodic_velocity', 'periodic_pressure', 'dimensionless', 'cells', 'wall_seconds')
    print(json.dumps(dict(output=args.output, **{key: report[key] for key in keys}), indent=2))
elif args.command == 'plot':
    from .analysis import plots
    plots(args.dataset)
elif args.command == 'validate':
    from .analysis import validate
    print(json.dumps(validate(args.dataset), indent=2))
else:
    from .compare import compare
    print(json.dumps(compare(args.reference, args.candidate, args.angle), indent=2))
