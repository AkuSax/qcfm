"""Verify saved evidence without model calls."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PPI = Path('C:/Users/Saxen/ppi_IG')
COL = PPI / 'results/representation_diagnosis/20261015/attempt02/gcp_a100_retry_01/collected'
BUNDLE = PPI / 'results/qhts_feasibility/20261012/independent_data_arm/capture_capacity_retries_20261012/request_0002/collected/bundle'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))

def blobs(value):
    if isinstance(value, dict):
        if value.get('type') == 'blob':
            yield value
        for child in value.values():
            yield from blobs(child)
    elif isinstance(value, list):
        for child in value:
            yield from blobs(child)

def main():
    records = read_rows(COL / 'records.csv')
    checks = []
    refs = {}
    conflicts = []
    for row in records:
        path = PPI / row['capture_path']
        checks.append({'path': str(path), 'expected': row['capture_sha256'], 'actual': sha(path)})
        for ref in blobs(json.loads(path.read_text())):
            name = ref['file']
            if name in refs and refs[name] != ref:
                conflicts.append(name)
            refs[name] = ref
    blob_checks = []
    for name, ref in sorted(refs.items()):
        path = BUNDLE / 'captures' / name
        actual = sha(path) if path.exists() else None
        blob_checks.append({'file': name, 'expected_sha256': ref['file_sha256'],
                            'actual_sha256': actual, 'expected_bytes': ref['stored_bytes'],
                            'actual_bytes': path.stat().st_size if path.exists() else None,
                            'pass': actual == ref['file_sha256'] and path.stat().st_size == ref['stored_bytes']})
    source_checks = []
    for row in read_rows(COL / 'source_hashes.csv'):
        path = PPI / row['path']
        actual = sha(path) if path.exists() else None
        source_checks.append({'path': str(path), 'expected': row['sha256'], 'actual': actual,
                              'pass': actual == row['sha256']})
    expected = {'records.csv': 'b6c9473de5dd8d09af35f5b5e42983498ee468a15e96c19b0ba066c3cc20791c',
                'splits.csv': '137c76616c1940cdddabd021bb6d5d1f9b1ead0a78dddd450de73ac88dbb0888',
                'exact_inputs.npz': '5dcbd0011ed01995fe76a71c12f62073be5a7905c7cb6e16e9cee7402fea1540'}
    retained = [{'name': name, 'actual': sha(COL / name), 'expected': digest} for name, digest in expected.items()]
    groups = Counter(r['chemical_group'] for r in records)
    sensitivity = Counter(r['chemical_group'] for r in records if r['pubchem_cid'] not in {'899202', '976329'})
    gate = json.loads((COL / 'capture_gate.json').read_text())
    final_mlp = json.loads((COL.parent / 'CPU_final_predictor_parity.json').read_text())
    report = {
        'checked_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if all(x['pass'] for x in blob_checks + source_checks) and all(x['actual'] == x['expected'] for x in checks + retained) and not conflicts else 'BLOCKED',
        'records': len(records), 'groups': len(groups),
        'nonsingleton_groups': sum(n > 1 for n in groups.values()),
        'within_group_pairs': sum(n*(n-1)//2 for n in groups.values()),
        'sensitivity_records': sum(sensitivity.values()),
        'sensitivity_pairs': sum(n*(n-1)//2 for n in sensitivity.values()),
        'capture_manifest_checks': checks, 'retained_file_checks': retained,
        'source_hash_checks': source_checks,
        'blob_count': len(refs), 'blob_metadata_conflicts': sorted(set(conflicts)),
        'blob_hash_failures': [x for x in blob_checks if not x['pass']],
        'stored_blob_bytes': sum(x['actual_bytes'] or 0 for x in blob_checks),
        'saved_A100_full_affinity_gate': gate,
        'saved_CPU_final_MLP_gate': final_mlp,
        'replay_limit': 'Saved receipts only. This script does not replay either boundary.',
        'model_calls': 0, 'training_runs': 0, 'cloud_calls': 0,
    }
    (HERE / 'blob_hash_checks.json').write_text(json.dumps(blob_checks, indent=2) + '\n')
    (HERE / 'primary_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ['status','records','groups','nonsingleton_groups','within_group_pairs','sensitivity_records','sensitivity_pairs','blob_count','stored_blob_bytes','blob_hash_failures']}))
    return 0 if report['status'] == 'PASS' else 1

if __name__ == '__main__':
    raise SystemExit(main())
