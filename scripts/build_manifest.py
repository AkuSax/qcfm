"""Record the files in the compact collaborator repository."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    scope = json.loads((ROOT / 'publication/compact_scope.json').read_text(encoding='utf-8'))
    for row in scope['selected_files']:
        if sha256(ROOT / row['path']) != row['sha256']:
            raise ValueError('Copied research file changed: ' + row['path'])
    manifest = ROOT / 'publication/file_manifest.json'
    entries = []
    for path in sorted(ROOT.rglob('*')):
        relative = path.relative_to(ROOT)
        if not path.is_file() or path == manifest:
            continue
        if set(relative.parts) & {'.git', '__pycache__', '.pytest_cache', '.venv'}:
            continue
        if relative.parts[0] == 'results':
            continue
        entries.append({'path': relative.as_posix(), 'bytes': path.stat().st_size,
                        'sha256': sha256(path)})
    total = sum(row['bytes'] for row in entries)
    if total >= 100_000_000:
        raise ValueError('Compact files exceed 100 MB')
    record = {'algorithm': 'SHA-256', 'manifest_excludes_itself': True,
              'edition': 'compact collaborator repository', 'file_count': len(entries),
              'total_bytes': total, 'maximum_bytes': 100_000_000, 'files': entries}
    manifest.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    if total + manifest.stat().st_size >= 100_000_000:
        raise ValueError('Manifest makes the compact files exceed 100 MB')
    print(json.dumps({'status': 'PASS', 'files': len(entries) + 1,
                      'tracked_bytes_including_manifest': total + manifest.stat().st_size}))


if __name__ == '__main__':
    main()
