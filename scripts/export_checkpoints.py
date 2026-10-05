"""Export stored checkpoints as archives after verifying every reference hash."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--source-root', type=Path, required=True, help='Original Dreamer workspace containing runs/')
parser.add_argument('--output', type=Path, required=True, help='Destination for .tar.gz files (outside Git)')
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / 'reference/manifest.json').read_text())
args.output.mkdir(parents=True, exist_ok=True)
for model, record in manifest['models'].items():
    source = args.source_root / 'runs' / f'elevated-track-{model}-seed0' / 'ckpt' / record['criterion']['checkpoint']
    destination = args.output / f'{model}-criterion.tar.gz'
    if destination.exists():
        raise FileExistsError(destination)
    for name, expected in record['checkpoint_sha256'].items():
        with (source / name).open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != expected:
            raise ValueError(f'Hash mismatch: {source / name}')
    with tarfile.open(destination, 'x:gz') as archive:
        for name in record['checkpoint_sha256']:
            archive.add(source / name, arcname=f'{model}/{source.name}/{name}')
        archive.add(root / 'reference' / model / 'config.yaml', arcname=f'{model}/config.yaml')
        archive.add(root / 'reference' / model / 'criterion_reached.json', arcname=f'{model}/criterion_reached.json')
    print(destination)
