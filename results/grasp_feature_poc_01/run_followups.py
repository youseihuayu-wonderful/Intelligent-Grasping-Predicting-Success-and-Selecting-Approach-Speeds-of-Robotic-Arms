"""Follow-up physical probes selected AFTER exploratory screening; not hold-out evaluation."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

from run_exploration import BASE, jitter_trials, provenance, run_one, save

ROOT = Path(__file__).resolve().parent


def main():
    directory = ROOT / 'followups'
    directory.mkdir(exist_ok=False)
    specs = jitter_trials()
    for target in ('bowl', 'bottle'):
        for scale in (.70, 1., 1.30):
            for opening in (.04, .08):
                specs.append((f'{target}.geometry.scale_{scale:.2f}.opening_{opening:.2f}', {
                    **BASE, 'target': target, 'mass_kg': .18, 'scale': scale, 'opening_m': opening,
                }, {'design': 'size_opening_factorial', 'selection': 'Use a mass whose reference succeeds for both assets, to avoid size contrasts against a failed reference.'}))
        for speed in (.04, .10, .20):
            for trigger in (0., .01):
                specs.append((f'{target}.timing.speed_{speed:.2f}.trigger_{trigger:.2f}', {
                    **BASE, 'target': target, 'mass_kg': .18,
                    'descent_speed_m_s': speed, 'close_trigger_m': trigger,
                }, {'design': 'speed_closure_timing_factorial', 'selection': 'Use successful nominal-mass references and separate timing from the speed setting.'}))
    save(directory / 'protocol.json', {
        'stage': 'Post-screening targeted verification', 'trials': specs,
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT / 'run_exploration.py')},
        'jitter_scope': 'Ten independently generated XYZ grasp errors per object, shared across three configurations; an exploratory conditional frequency over this specific error distribution, not a calibrated learned probability.',
        'label_rule': 'Same strict contact/clearance/lift/relative stability plus continuous two-second hold as screening; initial invalid states unlabelled; numerical errors abort.',
        'physics_and_control': 'Same declared compiled scene and contact settings as screening, not the original demo settings.',
    })
    save(directory / 'environment.json', provenance())
    rows = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=3) as pool:
        pending = [pool.submit(run_one, spec, directory) for spec in specs]
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            print(json.dumps({'trial': row['trial_id'], 'success': row['success'],
                              'kind': row['failure_kind'], 'reason': row['failure_reason']}, ensure_ascii=False), flush=True)
    order = {spec[0]: i for i, spec in enumerate(specs)}
    rows.sort(key=lambda row: order[row['trial_id']])
    save(directory / 'trials.json', {'rows': rows})
    summary = {
        'attempted': len(rows), 'successes': sum(r['success'] is True for r in rows),
        'failures': sum(r['success'] is False for r in rows),
        'invalid_initial_states': sum(r['success'] is None for r in rows),
        'warning_trials': sum(bool(r['warnings']) for r in rows),
        'wall_s': time.monotonic() - started,
    }
    save(directory / 'summary.json', summary)
    print('SUMMARY ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
