"""Archive explicit Experiment 02 recordings without changing their raw evidence."""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np


ROOT = Path(__file__).resolve().parent
TARGETS = {'spoon', 'pen', 'bowl', 'bottle', 'glass'}
BASELINES = ('five_object_cabinet_20261122', 'five_object_tabletop')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    with path.open(encoding='utf-8') as stream:
        return json.load(stream)


def repository_file(name):
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError(f'Not a repository file: {name}')
    return path


def require_final(record):
    if record.get('outcome') is not True or record.get('controller_status') != 'succeeded':
        raise ValueError('Final run must explicitly report outcome=true and succeeded')
    objects = record.get('objects', [])
    if len(objects) != 5 or {item.get('target') for item in objects} != TARGETS:
        raise ValueError('Final run must contain exactly the five distinct target objects')
    for item in objects:
        if any(item.get(key) is not True for key in
               ('inside', 'supported', 'detached', 'stable', 'stored')):
            raise ValueError(f'Final physical predicates failed: {item.get("target")}')
        force = item.get('support_force_n')
        if isinstance(force, bool) or not isinstance(force, (int, float)) or not math.isfinite(force) or force <= 0:
            raise ValueError('Final floor support must have a positive measured force')
    for key in ('drawer_closed', 'hand_empty'):
        if record.get(key) is not True:
            raise ValueError(f'Final run must explicitly report {key}=true')
    hold = record.get('verified_hold_s')
    if isinstance(hold, bool) or not isinstance(hold, (int, float)) or not math.isfinite(hold) or hold < 3:
        raise ValueError('Final verified hold must be at least 3 simulation seconds')
    resets = record.get('scene_resets_during_mission')
    if type(resets) is not int or resets != 0:
        raise ValueError('Final run must measure zero scene resets during the mission')
    if record.get('warnings') != []:
        raise ValueError('Final run must explicitly report an empty numerical warnings list')
    hashes = record.get('source_sha256')
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError('Final run requires source_sha256 provenance')
    required = {'tabletop_storage.py', 'drawer_storage_checks.py', 'grasp_tabletop.py',
                'grasp.py', 'simulation.py', 'grasp_study_physics.py',
                'grasp_study_objects.py', 'assets/grasp_study/tabletop_cabinet.xml'}
    if not required.issubset(hashes):
        raise ValueError(f'Final source hashes missing: {sorted(required - hashes.keys())}')
    for name, expected in hashes.items():
        if sha256(repository_file(name)) != expected:
            raise ValueError(f'Final source hash differs from repository: {name}')


def npz_inventory(path):
    with np.load(path, allow_pickle=False) as arrays:
        fields = {name: {'shape': list(arrays[name].shape), 'dtype': str(arrays[name].dtype)}
                  for name in arrays.files}
        span = None
        if 'time_s' in arrays.files:
            times = arrays['time_s']
            if times.ndim != 1 or not len(times) or not np.isfinite(times).all():
                raise ValueError(f'Invalid simulation samples in {path}')
            if np.any(np.diff(times) <= 0):
                raise ValueError(f'Non-increasing simulation samples in {path}')
            span = {'first_s': float(times[0]), 'last_s': float(times[-1]),
                    'span_s': float(times[-1] - times[0]), 'sample_count': len(times)}
            for name in ('qpos', 'qvel', 'ctrl'):
                if name in arrays.files and (arrays[name].ndim != 2 or arrays[name].shape[0] != len(times)):
                    raise ValueError(f'Mismatched {name} simulation samples in {path}')
        return {'fields': fields, 'simulation_samples': span}


def copy_file(source, destination):
    if source.is_symlink():
        raise ValueError(f'Symlink evidence is not supported: {source}')
    expected = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.is_symlink() or sha256(destination) != expected:
            raise FileExistsError(f'Refusing to overwrite differing evidence: {destination}')
    else:
        with source.open('rb') as src, destination.open('xb') as dst:
            shutil.copyfileobj(src, dst)
    if sha256(destination) != expected or sha256(source) != expected:
        raise ValueError(f'Copy SHA256 verification failed: {source}')
    result = {'bytes': destination.stat().st_size, 'sha256': expected}
    if source.suffix == '.npz':
        result['npz'] = npz_inventory(destination)
    return result


def encode_model(source, destination):
    """Preserve the logical model while storing deterministic, verified gzip."""
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'model.mjb').exists():
        raise FileExistsError(f'Refusing mixed raw/encoded model storage: {destination}')
    logical_sha256 = sha256(source)
    logical_bytes = source.stat().st_size
    with tempfile.TemporaryDirectory(prefix='.model-encoding-', dir=destination) as temporary:
        encoded = Path(temporary) / 'model.mjb.gz'
        with source.open('rb') as src, encoded.open('xb') as dst:
            with gzip.GzipFile(filename='', mode='wb', fileobj=dst, mtime=0) as compressed:
                shutil.copyfileobj(src, compressed, length=1024 * 1024)
        if encoded.stat().st_size >= 100 * 1024 * 1024:
            raise ValueError('Compressed model is >=100MiB and cannot be published on GitHub')
        digest = hashlib.sha256()
        decompressed_bytes = 0
        with gzip.open(encoded, 'rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
                decompressed_bytes += len(block)
        if (digest.hexdigest() != logical_sha256 or decompressed_bytes != logical_bytes
                or sha256(source) != logical_sha256):
            raise ValueError(f'Lossless model verification failed: {source}')
        info = copy_file(encoded, destination / 'model.mjb.gz')
    descriptor = {
        'logical_path': 'model.mjb', 'logical_sha256': logical_sha256,
        'encoding': 'gzip', 'stored_path': 'model.mjb.gz',
        'stored_sha256': info['sha256'], 'uncompressed_bytes': logical_bytes,
        'stored_bytes': info['bytes']}
    descriptor_path = destination / 'model_storage.json'
    write_report(descriptor_path, json.dumps(descriptor, indent=2, sort_keys=True) + '\n')
    info['logical_original'] = {'path': 'model.mjb', 'sha256': logical_sha256,
                                'bytes': logical_bytes}
    info['encoding'] = 'gzip'
    return {'model.mjb.gz': info,
            'model_storage.json': {'sha256': sha256(descriptor_path),
                                   'bytes': descriptor_path.stat().st_size}}


def copy_tree(source, destination):
    if not source.is_dir() or source.is_symlink():
        raise ValueError(f'Not a plain run directory: {source}')
    files = {}
    if (source / 'model.mjb').is_file():
        for reserved in ('model.mjb.gz', 'model_storage.json'):
            if (source / reserved).exists():
                raise ValueError(f'Raw source contains reserved archive model storage: {source / reserved}')
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError(f'Symlink evidence is not supported: {path}')
        relative = path.relative_to(source)
        if path.is_dir():
            (destination / relative).mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            if relative == Path('model.mjb'):
                files.update(encode_model(path, destination))
            else:
                files[relative.as_posix()] = copy_file(path, destination / relative)
        else:
            raise ValueError(f'Unsupported evidence file: {path}')
    verification = source / 'verification.json'
    if 'model.mjb.gz' in files and verification.is_file():
        recorded_model = load_json(verification).get('compiled_model')
        if recorded_model is not None:
            if (recorded_model.get('path') != 'model.mjb'
                    or recorded_model.get('sha256') != files['model.mjb.gz']['logical_original']['sha256']):
                raise ValueError(f'Original compiled model metadata mismatch: {source}')
    return files


def write_report(path, text):
    data = text.encode('utf-8')
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != data:
            raise FileExistsError(f'Refusing to overwrite differing report: {path}')
        return
    with path.open('xb') as stream:
        stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', action='append', type=Path, default=[])
    parser.add_argument('--final-run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--check-log', action='append', type=Path, default=[])
    args = parser.parse_args()
    output = args.output.resolve()
    final = args.final_run.resolve()
    sources = list(dict.fromkeys(path.resolve() for path in args.source_run))
    names = [path.name for path in sources]
    if len(set(names)) != len(names):
        raise ValueError('Source runs with the same basename cannot share an archive')
    for source in [*sources, final]:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise ValueError('Archive output and source runs must not overlap')
    final_record = load_json(final / 'verification.json')
    require_final(final_record)
    compiled = final_record.get('compiled_model')
    if not isinstance(compiled, dict) or compiled.get('path') != 'model.mjb':
        raise ValueError('Final recording requires the exact compiled model.mjb; no model fallback')
    if sha256(final / 'model.mjb') != compiled.get('sha256'):
        raise ValueError('Final compiled model SHA256 mismatch')
    final_trajectory = npz_inventory(final / 'trajectory.npz')
    if not {'time_s', 'qpos'}.issubset(final_trajectory['fields']):
        raise ValueError('Final trajectory requires time_s and qpos')
    records = {name: load_json(ROOT / 'results' / name / 'verification.json') for name in BASELINES}
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'schema_version': 1, 'experiment': 2, 'runs': [], 'final': {},
                'baselines': {}, 'check_logs': [], 'source_snapshot': {},
                'reproduction': 'Partial source snapshot only. Reproduce from the repository checkout with uv.lock; this snapshot is not standalone.'}
    for source in sources:
        files = copy_tree(source, output / 'runs' / source.name)
        verification = source / 'verification.json'
        record = load_json(verification) if verification.is_file() else {}
        limitations = []
        if not verification.is_file():
            limitations.append('Original verification.json unavailable')
        if not record.get('source_sha256'):
            limitations.append('Original source hashes unavailable; historical source version cannot be reconstructed from this run')
        if not (source / 'trajectory.npz').is_file():
            limitations.append('Original trajectory.npz unavailable')
        if not record.get('compiled_model'):
            limitations.append('Original compiled model unavailable; exact historical model replay unavailable')
        if not record.get('events'):
            limitations.append('Original initialization/phase event measurements unavailable')
        if not (source / 'trajectory.npz').is_file():
            limitations.append('Recorded initialization/sample span unavailable')
        missing = sorted({'outcome', 'controller_status', 'objects', 'drawer_closed',
                          'hand_empty', 'verified_hold_s', 'scene_resets_during_mission',
                          'warnings', 'events'} - record.keys())
        if missing:
            limitations.append('Original verification fields unavailable: ' + ', '.join(missing))
        manifest['runs'].append({'source': str(source), 'archive': f'runs/{source.name}',
                                 'files': files, 'recorded_outcome': record.get('outcome'),
                                 'metadata_limitations': limitations})
    manifest['final'] = {'source': str(final), 'archive': 'final',
                         'files': copy_tree(final, output / 'final'),
                         'strict_acceptance_verified': True}
    if manifest['final']['files']['model.mjb.gz']['logical_original']['sha256'] != compiled['sha256']:
        raise ValueError('Compiled model changed during archival')
    manifest['final']['compiled_model'] = compiled
    manifest['final']['recording_limits'] = (
        'Replay covers only recorded simulation samples, not any unrecorded initialization. '
        'See original events and NPZ first/last simulation timestamps; no missing samples are synthesized. '
        'Exact replay decodes final/model.mjb.gz using model_storage.json; never reconstructs a model fallback.')
    for name in BASELINES:
        path = ROOT / 'results' / name / 'verification.json'
        manifest['baselines'][name] = copy_file(path, output / 'baselines' / name / path.name)
    for name, expected in final_record['source_sha256'].items():
        source = repository_file(name)
        if source.suffix.lower() in {'.py', '.xml', '.json', '.toml', '.lock', '.yaml', '.yml', '.ini', '.cfg', '.txt'}:
            info = copy_file(source, output / 'source_snapshot' / name)
            info['capture'] = 'inspection_text'
        else:
            info = {'sha256': sha256(source), 'bytes': source.stat().st_size,
                    'capture': 'repository_dependency_reference_only',
                    'repository_path': name}
        if info['sha256'] != expected:
            raise ValueError(f'Source changed during capture: {name}')
        manifest['source_snapshot'][name] = info
    log_names = set()
    for log in args.check_log:
        log = log.resolve()
        if log.name in log_names:
            raise ValueError(f'Duplicate check-log basename: {log.name}')
        log_names.add(log.name)
        manifest['check_logs'].append({'source': str(log), 'archive': f'check_logs/{log.name}',
                                       **copy_file(log, output / 'check_logs' / log.name),
                                       'interpretation': 'Verbatim supplied log; no pass/fail status inferred by archiver'})
    frozen = records[BASELINES[0]].get('unchanged_frozen_study_sha256', {})
    dataset_evidence = {}
    for name, expected in frozen.items():
        matches = [path for path in (ROOT / 'results').rglob(Path(name).name)
                   if path.is_file() and not path.is_relative_to(output)
                   and path.as_posix().endswith('/' + name) and sha256(path) == expected]
        if not matches:
            raise ValueError(f'Cannot verify unchanged historical classifier artifact: {name}')
        dataset_evidence[name] = {'sha256': expected,
                                  'matching_repository_paths': [str(path.relative_to(ROOT)) for path in sorted(matches)]}
    final_cabinet = final_record.get('cabinet', {})
    prior_cabinet = records[BASELINES[0]].get('cabinet', {})
    protocol_rows = [
        {'fact': 'Task', 'previous': 'Five independent grasp-and-hold attempts; reset between targets',
         'experiment_02': 'Five continuous storage placements, drawer closure and home; no inter-item resets'},
        {'fact': 'Housing width (configured mm)', 'previous': 300, 'experiment_02': 400},
        {'fact': 'Conservative cavity width (configured mm)', 'previous': 254, 'experiment_02': 354},
        {'fact': 'Drawer travel (configured mm)', 'previous': 160, 'experiment_02': 240},
        {'fact': 'Drawer joint position (baseline initial / Experiment 02 final), m',
         'previous': prior_cabinet.get('initial_drawer_position_m'),
         'experiment_02': final_cabinet.get('drawer_position_m')},
        {'fact': 'Final drawer joint velocity (measured m/s)', 'previous': 'Not recorded in cabinet baseline',
         'experiment_02': final_cabinet.get('drawer_velocity_m_s')},
        {'fact': 'Verified all-five closed-storage hold (simulation s)',
         'previous': 'Not performed', 'experiment_02': final_record['verified_hold_s']},
        {'fact': 'Scene resets during continuous storage mission',
         'previous': 'No continuous storage mission', 'experiment_02': final_record['scene_resets_during_mission']},
    ]
    comparison = {
        'schema_version': 1, 'experiment': 2,
        'protocol_differences': protocol_rows,
        'protocol_notes': 'Dimensions are configuration specifications, not measured object poses. Previous joint position is its recorded initially open position, not a storage end state. Null means unavailable.',
        'prior_demonstrations': {name: {'verification': f'baselines/{name}/verification.json',
                                      'recorded_scope': records[name].get('scope'),
                                      'recorded_status': records[name].get('status')}
                                 for name in BASELINES},
        'task_difference': 'Prior demonstrations independently grasped and held five objects with resets between targets; the cabinet demonstration did not place objects or close the drawer. Experiment 02 continuously stores all five, physically closes the drawer, returns home and holds the final state without inter-item resets.',
        'final_acceptance': {key: final_record[key] for key in
                             ('outcome', 'controller_status', 'objects', 'drawer_closed', 'hand_empty',
                              'verified_hold_s', 'scene_resets_during_mission', 'warnings')},
        'final_simulation_samples': final_trajectory['simulation_samples'],
        'recorded_final_protocol': {
            'release_strategy': final_record.get('release_strategy'),
            'events': final_record.get('events'),
            'metadata_note': 'Values copied from final verification only; null means unavailable. Development-only opening-force experiments are not asserted as the final protocol.'},
        'classifier_data': {'new_training_data_collected_here': False,
                            'statement': 'Existing classifier datasets are unchanged by this archive and were not newly collected or trained here. Historical frozen artifacts below match the prior recorded hashes; this is not a new classifier evaluation.',
                            'separate_existing_batch_comparison': 'dataset_batch_comparison.json (maintained separately; historical batches, not data collected by this nominal mission)',
                            'verified_historical_artifacts': dataset_evidence},
        'limitations': ['One nominal continuous mission does not establish a success rate or speed improvement.',
                        'No comparison of unmatched simulation and wall clocks is made.',
                        'Historical one-bowl storage and classifier datasets are separate tasks, not the direct five-object baseline.',
                        'Missing historical metadata is unavailable, not reconstructed.',
                        'Check logs are preserved verbatim, including failures; no logs are fabricated.'],
        'reproduction': manifest['reproduction'],
    }
    markdown = '\n'.join([
        '# Experiment 02 — continuous five-object storage', '',
        comparison['task_difference'], '',
        '## Protocol differences', '',
        '| Fact | Previous demonstration | Experiment 02 |',
        '| --- | --- | --- |',
        *[f'| {row["fact"]} | {row["previous"]} | {row["experiment_02"]} |'
          for row in protocol_rows], '',
        comparison['protocol_notes'], '',
        '## Recorded result',
        'The explicitly supplied final verification passes all five strict physical predicates for every object: inside, floor-supported, detached, stable and stored. The drawer is closed, the hand is empty, the verified hold is at least 3 simulation seconds, there are zero inter-item scene resets and no numerical warnings.', '',
        'Development runs remain under `runs/`; the explicit successful run is under `final/`. Original files are byte-for-byte except `model.mjb`, stored losslessly as deterministic gzip with a `model_storage.json` descriptor. Original verification JSON and its logical model hash remain unchanged. File sizes, stored/logical SHA256 checks, NPZ fields/shapes and simulation sample spans are in `manifest.json`. Missing metadata is identified there, never rewritten.', '',
        '## Evidence and limits',
        'Prior records are copied under `baselines/` and compared in `comparison.json`. Independent grasp-and-hold demonstrations are not equivalent to continuous storage. No success-rate or speed improvement is inferred, and simulation clocks are not compared to wall clocks.', '',
        comparison['classifier_data']['statement'], '',
        'Historical classifier batch comparison is documented separately in `dataset_batch_comparison.json`; those batches were not collected by this nominal storage task.', '',
        'Optional check logs are copied verbatim under `check_logs/`; their actual content, including any failures, is authoritative. No pass status or missing log is invented.', '',
        manifest['reproduction'], '',
    ])
    # Recheck provenance after all copies; a concurrently edited final source is an error.
    require_final(final_record)
    for name, value in (('manifest.json', manifest), ('comparison.json', comparison)):
        write_report(output / name, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    write_report(output / 'COMPARISON.md', markdown)
    print(output)


if __name__ == '__main__':
    main()
