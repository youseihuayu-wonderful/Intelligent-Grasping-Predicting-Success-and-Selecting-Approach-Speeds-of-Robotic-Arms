"""Restore the six hash-matched Experiment01 sources over a fresh shared runtime.

The destination must be new or empty. The current working checkout is never
modified. This restores source, not an unavailable original compiled-model image.
"""

import argparse
import json
from pathlib import Path
import shutil

from simulation_delivery import destination_path, regular_file, restore, sha256, verify

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'archives/experiment_01_sources'
REFERENCE = 'results/five_object_tabletop/verification.json'


def restore_original(capsule, destination):
    destination_path(destination, 'EXPERIMENT01_RESTORED.json')
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f'Destination must be new or empty: {destination}')
    archived = json.loads(regular_file(SOURCE, 'manifest.json').read_text())
    if archived.get('schema') != 'experiment01-source-recovery-v1':
        raise ValueError('Unexpected original-source manifest')
    reference_path = regular_file(ROOT, REFERENCE)
    reference = json.loads(reference_path.read_text())
    recorded = reference['viewer_source_sha256']
    if set(archived['files']) != set(recorded):
        raise ValueError('Recovered file set differs from the original verification')
    for name, info in archived['files'].items():
        source = regular_file(SOURCE, name)
        if (info['sha256'] != recorded[name] or sha256(source) != recorded[name]
                or source.stat().st_size != info['bytes']):
            raise ValueError(f'Original source hash/size mismatch: {name}')
    runtime = verify(capsule)
    if runtime['files'][REFERENCE]['sha256'] != sha256(reference_path):
        raise ValueError('Shared runtime contains a different historical reference')
    restore(capsule, destination)
    expected = {name: info['sha256'] for name, info in runtime['files'].items()}
    for name, digest in recorded.items():
        shutil.copyfile(regular_file(SOURCE, name), destination_path(destination, name))
        expected[name] = digest
    for name, digest in expected.items():
        if sha256(regular_file(destination, name)) != digest:
            raise ValueError(f'Assembled first-experiment source differs: {name}')
    record = {
        'mode': 'original_experiment01_sources_with_shared_runtime',
        'source_manifest': archived,
        'shared_runtime_archive_sha256': runtime['archive']['sha256'],
        'assembled_source_sha256': expected,
        'restore_program_sha256': sha256(Path(__file__)),
        'limits': [
            'All six original viewer/tabletop/test source files exactly match the archived verification.',
            'Other dependencies come from the verified shared runtime; this is not an original compiled-model or full operating-system snapshot.',
            'Restoration does not execute a trial or create a new physical success label.',
        ],
    }
    with (destination / 'EXPERIMENT01_RESTORED.json').open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(f'Restored all {len(recorded)} exact Experiment01 source files and verified {len(expected)} assembled runtime files in {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capsule', type=Path, default=ROOT / 'assets/simulation_delivery_v1')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    restore_original(args.capsule.absolute(), args.destination.absolute())


if __name__ == '__main__':
    main()
