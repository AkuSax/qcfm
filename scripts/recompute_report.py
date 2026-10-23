"""Recompute the saved-prediction report without changing frozen evidence."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'exploration/20261022T232117Z'


def compare(left, right, location='root'):
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ValueError(f'Different fields: {location}')
        for key in left:
            compare(left[key], right[key], f'{location}.{key}')
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ValueError(f'Different length: {location}')
        for i, (a, b) in enumerate(zip(left, right)):
            compare(a, b, f'{location}[{i}]')
    elif isinstance(left, (int, float)) and not isinstance(left, bool):
        if not isinstance(right, (int, float)) or not math.isclose(left, right, rel_tol=0, abs_tol=1e-8):
            raise ValueError(f'Different number: {location}')
    elif left != right:
        raise ValueError(f'Different value: {location}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('results/recomputed'))
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(RUN) or output.is_relative_to(ROOT / 'preparation'):
        raise ValueError('Write outside the frozen experiment directories')
    if output.exists():
        raise ValueError('Use a new output directory')
    output.mkdir(parents=True)
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = '1'
    command = [sys.executable, str(RUN / 'interaction/postrun_decision.py'),
               '--predictions', str(RUN / 'classical/predictions.csv'),
               '--output-dir', str(output)]
    with (output / 'stdout.log').open('w', encoding='utf-8') as stdout, (output / 'stderr.log').open('w', encoding='utf-8') as stderr:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr,
                                timeout=600, check=False)
    if result.returncode:
        raise RuntimeError(f'Report failed with exit code {result.returncode}; see {output / "stderr.log"}')
    archived = json.loads((RUN / 'interaction/postrun/decision_report.json').read_text(encoding='utf-8'))
    rebuilt = json.loads((output / 'decision_report.json').read_text(encoding='utf-8'))
    for field in ('metric_rows', 'candidate_metric_means_and_seed_ranges', 'frozen_acceptance'):
        compare(archived[field], rebuilt[field], field)
    def read_bootstrap(path):
        with path.open(newline='', encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    expected = read_bootstrap(RUN / 'interaction/postrun/paired_bootstrap.csv')
    observed = read_bootstrap(output / 'paired_bootstrap.csv')
    if len(expected) != 210 or len(observed) != 210:
        raise ValueError('Expected 210 bootstrap rows')
    numeric = {'gain_mean', 'ci95_lower', 'ci95_upper'}
    for i, (a, b) in enumerate(zip(expected, observed)):
        for key in a:
            compare(float(a[key]) if key in numeric else a[key],
                    float(b[key]) if key in numeric else b[key], f'bootstrap[{i}].{key}')
    receipt = {'status': 'PASS', 'metric_rows': len(rebuilt['metric_rows']),
               'bootstrap_rows': len(observed), 'tolerance': 1e-8,
               'estimator_fits': 0, 'encoder_calls': 0,
               'scientific_decision': 'NO_PROMOTION'}
    (output / 'reproduction_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
