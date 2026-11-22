"""Build, verify and safely restore the two-task runtime source capsule (stdlib only)."""

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPOSITORY = 'https://github.com/youseihuayu-wonderful/Intelligent-Grasping-Predicting-Success-and-Selecting-Approach-Speeds-of-Robotic-Arms'
RECORD_COMMIT = '7a2c5642113d0d3b9c5b5d5205feda843b2c043a'
FINAL = 'results/experiment_02_five_object_storage/final/verification.json'
ENVIRONMENT = 'results/grasp_classifier_five_objects_20261121/environment.json'
RECORDS = (
    FINAL, ENVIRONMENT,
    'results/five_object_tabletop/verification.json',
    'results/five_object_tabletop/initial_layout_verification.json',
    'results/five_object_cabinet_20261122/verification.json',
    'results/experiment_02_five_object_storage/COMPARISON.md',
)
SEEDS = (
    'simulation_delivery.py', 'run_tabletop_grasps.py', 'tabletop_storage.py',
    'serve_grasp_study.py', 'tests/test_grasp_tabletop.py', 'pyproject.toml', 'uv.lock',
    'web/study.html', 'web/study.js', 'web/study.css',
    'vendor/libero/LICENSE', 'vendor/panda/LICENSE',
)
# Only loaded model roots are dependency roots. Recorded input inventories also
# contain unused original model variants (some reference absent template textures).
MODEL_ROOTS = (
    'scene.xml', 'vendor/panda/panda.xml',
    'assets/grasp_study/tabletop_cabinet.xml',
    'assets/grasp_study/spoon.xml', 'assets/grasp_study/glass.xml', 'assets/grasp_study/pen.xml',
    'vendor/libero/stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml',
    'vendor/libero/turbosquid_objects/wine_bottle/wine_bottle.xml',
    'vendor/libero/turbosquid_objects/wine_rack/wine_rack.xml',
    'vendor/libero/articulated_objects/white_cabinet.xml',
)
MAX_FILE = 100 * 1024 * 1024
ARCHIVE = 'runtime.tar.gz'


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (not name or '\\' in name or ':' in name or path.is_absolute() or path.as_posix() != name
            or any(part in ('.', '..') or part.startswith('.') for part in path.parts)
            or any(part in ('__pycache__', 'venv') for part in path.parts)
            or path.suffix == '.pyc'):
        raise ValueError(f'Unsafe capsule path: {name!r}')
    return path


def regular_file(root, name):
    relative = safe_name(name)
    path = root.joinpath(*relative.parts)
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f'Symlink is not allowed: {item}')
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size >= MAX_FILE:
        raise ValueError(f'File must be smaller than 100 MiB: {name}')
    return path


def source_inventory(root):
    """Recorded inputs plus transitive local imports and MJCF file references."""
    expected = {}
    bases = {}
    for record, key in ((FINAL, 'source_sha256'), (ENVIRONMENT, 'input_sha256')):
        content = json.loads(regular_file(root, record).read_text())
        for name, digest in content[key].items():
            safe_name(name)
            if name in expected and expected[name] != digest:
                raise ValueError(f'Conflicting recorded hashes: {name}')
            expected[name] = digest
            bases.setdefault(name, []).append(f'{record}#{key}')
    xml_roots = set(MODEL_ROOTS)
    names = set(expected) | set(RECORDS) | set(SEEDS) | xml_roots
    processed = set()
    while names - processed:
        name = min(names - processed)
        path = regular_file(root, name)
        processed.add(name)
        if name in expected and sha256(path) != expected[name]:
            raise ValueError(f'Recorded source hash mismatch: {name}')
        if path.suffix == '.py':
            for node in ast.walk(ast.parse(path.read_text(), filename=name)):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    modules = [node.module] if node.module else []
                else:
                    continue
                for module in modules:
                    candidate = module.split('.')[0] + '.py'
                    if (root / candidate).is_file():
                        names.add(candidate)
                        bases.setdefault(candidate, []).append(f'import:{name}')
        elif name in xml_roots:
            document = ET.parse(path).getroot()
            compiler = document.find('compiler')
            options = {} if compiler is None else compiler.attrib
            for element in document.iter():
                filename = element.get('file')
                if filename is None:
                    continue
                if Path(filename).is_absolute() or '\\' in filename:
                    raise ValueError(f'Nonportable XML reference in {name}: {filename}')
                directory = ''
                if element.tag in ('mesh', 'texture', 'hfield', 'skin'):
                    key = 'texturedir' if element.tag == 'texture' else 'meshdir'
                    directory = options.get(key, options.get('assetdir', ''))
                # Resolve ./ components, but never permit references outside the source root.
                referenced = (path.parent / directory / filename).resolve()
                relative = referenced.relative_to(root.resolve()).as_posix()
                names.add(relative)
                if referenced.suffix == '.xml':
                    xml_roots.add(relative)
                    processed.discard(relative)
                bases.setdefault(relative, []).append(f'xml:{name}')
    files = {}
    for name in sorted(names):
        path = regular_file(root, name)
        files[name] = {
            'sha256': sha256(path), 'size_bytes': path.stat().st_size,
            'basis': sorted(set(bases.get(name, ['delivery supplement']))),
        }
    return files


def build(root, output):
    files = source_inventory(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'Output must be new: {output}')
    with tempfile.TemporaryDirectory(prefix='simulation-delivery-', dir=output.parent) as temporary:
        staging = Path(temporary)
        archive = staging / ARCHIVE
        with archive.open('xb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', filename='', mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                    for name, info in files.items():
                        path = regular_file(root, name)
                        entry = tarfile.TarInfo(name)
                        entry.size = info['size_bytes']
                        entry.mode = 0o644
                        entry.mtime = 0
                        with path.open('rb') as stream:
                            tar.addfile(entry, stream)
        if archive.stat().st_size >= MAX_FILE:
            raise ValueError('Compressed capsule must be smaller than 100 MiB')
        manifest = {
            'schema': 'simulation-delivery-v1',
            'repository': REPOSITORY,
            'recorded_experiment02_commit': RECORD_COMMIT,
            'recorded_experiment02_tag': 'experiment-02-five-object-storage',
            'scope': 'Current independent grasp reproduction and recorded Experiment02 storage runtime; not historical Experiment01 exact source.',
            'archive': {'path': ARCHIVE, 'sha256': sha256(archive),
                        'size_bytes': archive.stat().st_size},
            'files': files,
            'model_dependency_roots': list(MODEL_ROOTS),
            'licenses': {'vendor/libero/LICENSE': 'LIBERO bundled MIT license',
                         'vendor/panda/LICENSE': 'MuJoCo Menagerie Panda bundled Apache-2.0 license'},
            'limits': [
                'Generated source capsule, not the recorded compiled model snapshot.',
                'Logical source bytes are unchanged; tar/gzip metadata is normalized.',
                'Recorded verifications are provenance, not results of building this capsule.',
                'Additional imported modules/XML references are hashed delivery supplements.',
                'Unused recorded XML variants are preserved as provenance, not treated as task model roots.',
                'Dependencies are installed separately from the unchanged pyproject.toml and uv.lock.',
            ],
        }
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
        verify(staging)
        # The final directory is only exposed after complete archive/hash validation.
        output.mkdir()
        for path in staging.iterdir():
            shutil.move(str(path), output / path.name)
    print(f'Built {output}: {len(files)} source files')


def destination_path(destination, name):
    path = destination.joinpath(*safe_name(name).parts)
    for item in (path, *path.parents):
        if item.is_symlink():
            raise ValueError(f'Refusing destination symlink: {item}')
        if item.exists() and item != path and not item.is_dir():
            raise ValueError(f'Destination ancestor is not a directory: {item}')
    return path


def verify(capsule, destination=None):
    manifest_path = regular_file(capsule, 'manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'simulation-delivery-v1':
        raise ValueError('Unsupported capsule schema')
    archive_info = manifest['archive']
    if archive_info['path'] != ARCHIVE:
        raise ValueError('Unexpected capsule archive name')
    archive = regular_file(capsule, ARCHIVE)
    if sha256(archive) != archive_info['sha256'] or archive.stat().st_size != archive_info['size_bytes']:
        raise ValueError('Archive hash/size mismatch')
    files = manifest['files']
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty/invalid source manifest')
    for name, info in files.items():
        safe_name(name)
        if not isinstance(info['size_bytes'], int) or not 0 <= info['size_bytes'] < MAX_FILE:
            raise ValueError(f'Invalid member size: {name}')
    seen = set()
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar:
            name = member.name
            safe_name(name)
            if not member.isfile() or member.issparse() or member.linkname or name in seen or name not in files:
                raise ValueError(f'Unsafe, duplicate or unlisted tar member: {name}')
            info = files[name]
            if member.size != info['size_bytes']:
                raise ValueError(f'Tar member size mismatch: {name}')
            with tar.extractfile(member) as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != info['sha256']:
                raise ValueError(f'Tar member hash mismatch: {name}')
            seen.add(name)
    if seen != set(files):
        raise ValueError(f'Missing archive members: {sorted(set(files) - seen)}')
    if destination is not None:
        for name, info in files.items():
            path = destination_path(destination, name)
            if not path.is_file() or path.stat().st_size != info['size_bytes'] or sha256(path) != info['sha256']:
                raise ValueError(f'Restored source hash/size mismatch: {path}')
    return manifest


def restore(capsule, destination):
    manifest = verify(capsule)
    # Preflight every member before creating anything; identical files are idempotent.
    for name, info in manifest['files'].items():
        path = destination_path(destination, name)
        if path.exists() and (not path.is_file() or path.stat().st_size != info['size_bytes']
                              or sha256(path) != info['sha256']):
            raise FileExistsError(f'Refusing to overwrite differing existing file: {path}')
    with tarfile.open(capsule / ARCHIVE, 'r:gz') as tar:
        for member in tar:
            path = destination_path(destination, member.name)
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target)
    verify(capsule, destination)
    print(f'Restored and verified {len(manifest["files"])} files in {destination}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    builder = commands.add_parser('build', help='Require original recorded sources; write a new capsule directory')
    builder.add_argument('--source-root', type=Path, default=ROOT)
    builder.add_argument('--output', type=Path, default=ROOT / 'assets/simulation_delivery_v1')
    checker = commands.add_parser('verify', help='Verify archive and optionally restored source bytes')
    checker.add_argument('--capsule', type=Path, default=ROOT / 'assets/simulation_delivery_v1')
    checker.add_argument('--destination', type=Path)
    restorer = commands.add_parser('restore', help='Verify then restore; refuse differing existing files and symlinks')
    restorer.add_argument('--capsule', type=Path, default=ROOT / 'assets/simulation_delivery_v1')
    restorer.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.source_root.absolute(), args.output.absolute())
    elif args.command == 'restore':
        restore(args.capsule.absolute(), args.destination.absolute())
    else:
        manifest = verify(args.capsule.absolute(), None if args.destination is None else args.destination.absolute())
        print(f'Verified {len(manifest["files"])} files')


if __name__ == '__main__':
    main()
