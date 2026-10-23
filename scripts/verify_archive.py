"""Verify the published archive without fitting a model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads((ROOT / 'publication/file_manifest.json').read_text(encoding='utf-8'))
    errors = []
    for row in manifest['files']:
        path = ROOT / row['path']
        if not path.is_file() or sha256(path) != row['sha256']:
            errors.append(row['path'])
    feature = ROOT / 'preparation/20261022T182444Z/execution_full/max1000_20261022T203000Z/aim_features.npz'
    if sha256(feature) != 'ee7f4c5e74f9bb3b1950c98395249948f0910348858fea0df3f4384e195fbd34':
        errors.append('feature_bundle_hash')
    if errors:
        raise ValueError('Archive hash mismatch: ' + json.dumps(errors))
    print(json.dumps({'status': 'PASS', 'files_verified': len(manifest['files']),
                      'estimator_fits': 0, 'encoder_calls': 0}))


if __name__ == '__main__':
    main()
