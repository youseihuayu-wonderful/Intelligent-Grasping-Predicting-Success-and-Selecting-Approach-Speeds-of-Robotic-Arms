"""Summarize physical feature probes without fitting or evaluating a classifier."""
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(phase):
    return json.loads((ROOT / phase / 'trials.json').read_text())['rows']


def contact_context(row):
    target = row['settings']['target'] + '_object'
    # Separate target/table recontact and finger-to-finger closure from an
    # obstacle-blocked path. Preserve both in the raw collision records.
    closure_pairs = ({target, 'table'}, {'robot_left_finger', 'robot_right_finger'})
    obstacle_pairs = [p for p in row['collisions']['pairs']
                      if set((p['body_a'], p['body_b'])) not in closure_pairs]
    if obstacle_pairs:
        return 'other_unplanned_contact'
    if row['success'] is False and row['last_active_phase'] in ('close', 'lift', 'hold'):
        return 'grasp_contact_or_retention'
    return row['failure_kind']


def wilson(successes, n):
    if not n:
        return None
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n)) / denominator
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denominator
    return [center-half, center+half]


def main():
    phases = ['smoke', 'screening', 'interactions', 'jitter', 'followups']
    rows = {phase: load(phase) for phase in phases if (ROOT / phase / 'trials.json').exists()}
    result = {
        'phase_counts': {},
        'scope': 'Exploration only; no fitted classifier, no untouched model-evaluation set, no calibrated probabilities or unseen-object claim.',
        'category_limit': 'Only two existing geometry families, with shape-specific rim/neck grasp offsets. Position, mass and opening are controlled, but semantic class is not separable from geometry and grasp strategy.',
        'label_limit': 'Controller fails immediately on loss of a required hold predicate. A brief bilateral-contact loss may count as failure even if the object remains elevated at the next physics step. Do not equate every negative with an actual drop.',
    }
    for phase, selected in rows.items():
        completed = [r for r in selected if r['success'] is not None]
        result['phase_counts'][phase] = {
            'attempted': len(selected), 'valid': len(completed),
            'successes': sum(r['success'] is True for r in selected),
            'failures': sum(r['success'] is False for r in selected),
            'invalid': sum(r['success'] is None for r in selected),
            'failure_contexts': dict(Counter(contact_context(r) for r in selected if r['success'] is False)),
            'warnings': sum(bool(r['warnings']) for r in selected),
        }
    if 'screening' in rows:
        selected = rows['screening']
        references = {r['settings']['target']: r for r in selected if r['metadata']['factor'] == 'reference'}
        contrasts = []
        for row in selected:
            factor = row['metadata']['factor']
            if factor == 'reference':
                continue
            reference = references[row['settings']['target']]
            contrasts.append({
                'target': row['settings']['target'], 'factor': factor,
                'reference_value': reference['settings'][factor], 'changed_value': row['settings'][factor],
                'reference_success': reference['success'], 'changed_success': row['success'],
                'valid_pair': reference['success'] is not None and row['success'] is not None,
                'outcome_changed': reference['success'] is not None and row['success'] is not None and reference['success'] != row['success'],
                'failure_context': contact_context(row), 'trial_id': row['trial_id'],
            })
        result['screening_contrasts'] = contrasts
        result['screening_factors_with_outcome_changes'] = sorted({r['factor'] for r in contrasts if r['outcome_changed']})
    if 'interactions' in rows:
        groups = defaultdict(list)
        for row in rows['interactions']:
            s = row['settings']
            groups[(s['target'], s['mass_kg'], s['sliding_friction'])].append(row)
        force_pairs = []
        for (target, mass, friction), group in groups.items():
            group.sort(key=lambda r: r['settings']['closing_force_n'])
            force_pairs.append({
                'target': target, 'mass_kg': mass, 'mu': friction,
                'success_by_tendon_force_n': {str(r['settings']['closing_force_n']): r['success'] for r in group},
                'initial_fingerprints_match': len({r['initial_state_fingerprint'] for r in group}) == 1,
            })
        result['interaction_force_pairs'] = force_pairs
    jitter_rows = rows.get('jitter', []) + [
        r for r in rows.get('followups', [])
        if r['metadata']['design'] == 'paired_execution_error_jitter'
    ]
    if jitter_rows:
        groups = defaultdict(list)
        for row in jitter_rows:
            groups[(row['settings']['target'], row['metadata']['condition'])].append(row)
        result['jitter_empirical_frequencies'] = []
        for (target, condition), group in groups.items():
            valid = [r for r in group if r['success'] is not None]
            k = sum(r['success'] is True for r in valid)
            result['jitter_empirical_frequencies'].append({
                'target': target, 'condition': condition, 'successes': k, 'n': len(valid),
                'empirical_frequency': k/len(valid) if valid else None,
                'wilson_95_interval': wilson(k, len(valid)),
                'interpretation': 'Frequency over declared uniform XYZ grasp-error distribution at this configuration; NOT a learned or calibrated probability, and not a universal real-world success rate.',
            })
    for design in ('size_opening_factorial', 'speed_closure_timing_factorial'):
        result[design] = [
            {
                'trial_id': r['trial_id'], 'settings': r['settings'],
                'success': r['success'], 'failure_context': contact_context(r),
                'aabb_size_m': r['variant_features']['collision_world_aabb_size_m'],
            }
            for r in rows.get('followups', []) if r['metadata']['design'] == design
        ]
    all_rows = [r for selected in rows.values() for r in selected]
    result['physical_integrity'] = {
        'attempted': len(all_rows),
        'workspace_obstacle_contact_trials': sum(
            contact_context(r) == 'other_unplanned_contact' for r in all_rows),
        'successful_trials_meet_two_second_hold': all(
            r['verified_hold_s'] >= 2.0 and all(r['postconditions'].values())
            for r in all_rows if r['success'] is True),
        'actual_mass_matches_setting': all(
            math.isclose(r['variant_features']['actual_mass_kg'], r['settings']['mass_kg'], abs_tol=1e-12)
            for r in all_rows),
        'observed_contact_mu_matches_setting': all(
            all(math.isclose(mu, r['settings']['sliding_friction'], abs_tol=1e-6)
                for mu in r['actual_finger_target_sliding_frictions'])
            for r in all_rows),
        'trials_with_observed_finger_target_contacts': sum(
            bool(r['actual_finger_target_sliding_frictions']) for r in all_rows),
        'unplanned_contact_pairs': sorted({
            (p['body_a'], p['body_b'])
            for r in all_rows for p in r['collisions']['pairs']
        }),
    }
    output = ROOT / 'analysis.json'
    with output.open('w') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'screening_contrasts'}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
