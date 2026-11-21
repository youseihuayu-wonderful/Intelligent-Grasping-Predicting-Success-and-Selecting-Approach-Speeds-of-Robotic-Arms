"""Execute declared preflight specs; archive inputs before observing outcomes."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grasp_study_physics import provenance, run_one, save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('round')
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent / args.round
    specs = json.loads((directory / 'specs.json').read_text())
    save(directory / 'environment.json', provenance())
    for name in ('grasp.py', 'collisions.py', 'grasp_study_physics.py', 'grasp_study_objects.py',
                 'assets/grasp_study/spoon.xml', 'assets/grasp_study/glass.xml',
                 'assets/grasp_study/pen.xml'):
        target = directory / 'sources' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x') as stream:
            stream.write((ROOT / name).read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(run_spec, [(spec, directory) for spec in specs]))
    save(directory / 'trials.json', {'rows': rows})
    summary = {
        'attempted': len(rows),
        'successes': sum(row['success'] is True for row in rows),
        'failures': sum(row['success'] is False for row in rows),
        'invalid_initial_states': sum(row['success'] is None for row in rows),
        'failure_reasons': dict(Counter(row['failure_reason'] for row in rows if not row['success'])),
        'numerical_warning_trials': sum(bool(row['warnings']) for row in rows),
        'unexpected_contact_trials': sum(bool(row['collisions']['episodes']) for row in rows),
    }
    save(directory / 'summary.json', summary)
    print(json.dumps(summary), flush=True)


def run_spec(item):
    spec, directory = item
    row = run_one(spec, directory)
    print(json.dumps({key: row[key] for key in ('trial_id', 'success', 'failure_reason', 'verified_hold_s')}), flush=True)
    return row


if __name__ == '__main__':
    main()
