"""Select the collaborator files from the complete local archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


PACKAGE = 'preparation/20261022T182444Z'
PREVIOUS = PACKAGE + '/execution_full/max1000_20261022T203000Z'
RUN = 'exploration/20261022T232117Z'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    src, dst = args.source.resolve(), args.destination.resolve()
    if src == dst or not (dst / '.git').is_dir():
        raise ValueError('Use a separate Git checkout')
    selected = set()
    def select(folder, patterns):
        for pattern in patterns:
            selected.update(p.relative_to(src).as_posix() for p in (src / folder).glob(pattern) if p.is_file())
    select('', ['.gitignore', '.gitattributes', 'requirements-reports.txt',
                'FULL_DATASET_RESULT.md', 'FULL_DATASET_RUN_PLAN_20261022.md', 'RESEARCH_PLAN.md'])
    select('scripts', ['verify_archive.py', 'recompute_report.py'])
    select('data/frozen_inputs', ['records.csv', 'splits.csv', 'exact_inputs.npz',
                                  'source_hashes.csv', 'boundary.json', 'protocol.json'])
    select(PACKAGE + '/arm3_environment', ['*.py', '*.ps1', '*.in', '*.lock', 'observed_freeze.txt', 'README.md'])
    select(PACKAGE + '/arm4_protocol', ['*.py', '*.csv', 'comparison_config.json', 'README.md'])
    select(PACKAGE + '/orchestrator', ['*.py'])
    select(PACKAGE + '/arm5_integration', ['*.py'])
    select(PACKAGE + '/execution/guarded_fit_20261022', ['*.py', '*.ps1', '*.diff', 'guard_receipt.json'])
    select(PREVIOUS, ['aim_features.npz', 'aim_features.schema.json', 'feature_bundle_audit.json',
                      'protocol_amendment_v3.json', 'RESULTS.md', 'promotion_evaluation.json',
                      'evaluator_review.json', 'downstream_result.json'])
    select(PREVIOUS + '/fit_retry01', ['predictions.csv', 'outer_fit_manifest.csv',
                                      'metrics.csv', 'bootstrap_corrected.csv',
                                      'bootstrap_seed_repair_receipt.json'])
    select(RUN, ['RESULTS.md', 'FROZEN_PLAN.md', 'REPORT_CONTRACT.md', 'result_summary.json',
                 'plan_freeze_receipt.json', 'LEAD_REVIEW.md'])
    select(RUN + '/classical', ['*.py', 'predictions.csv', 'selection_manifest.jsonl',
                                'inner_candidate_scores.jsonl', 'fit_schedule.csv',
                                'stack_crossfit_splits.json', 'complete_attempt_accounting.*',
                                'coverage_annotation.*', 'metadata_amendment_receipt.json'])
    select(RUN + '/interaction', ['*.py', 'TRANSFORM_SPEC.md', 'POSTRUN_SPEC.md'])
    select(RUN + '/interaction/postrun', ['decision_report.json', 'decision_report.md',
                                         'paired_bootstrap.csv', 'final_postrun_receipt.json'])
    select(RUN + '/interaction/postrun/postrun_repair01', ['repair_receipt.json', 'source_diff.patch'])
    select(RUN + '/audit', ['independent_audit.*', 'numeric_reconciliation.json',
                            'independent_metrics.csv', 'independent_bootstrap_210.csv',
                            'independent_group_errors.csv', 'base_error_complementarity_18.csv'])
    entries = []
    for relative in sorted(selected):
        original, target = src / relative, dst / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        digest = sha256(original)
        if sha256(target) != digest:
            raise ValueError('Copy differs: ' + relative)
        entries.append({'path': relative, 'bytes': target.stat().st_size, 'sha256': digest})
    total = sum(row['bytes'] for row in entries)
    if total >= 100_000_000:
        raise ValueError('Selected files exceed 100 MB')
    record = {'scope': 'Collaborator code, frozen inputs, cached features, compact results, and audit summaries',
              'source_files_unchanged': True, 'selected_files': entries,
              'selected_bytes': total, 'maximum_bytes': 100_000_000,
              'omitted': ['trained model checkpoints', 'per-fit design matrices and coefficient archives',
                          'base-prediction cache files', 'raw fit journals and redundant partial predictions',
                          'per-molecule extraction intermediates', 'synthetic run outputs',
                          'virtual environments and downloaded dependencies', 'cloud login files']}
    (dst / 'publication').mkdir(exist_ok=True)
    (dst / 'publication/compact_scope.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'files_selected': len(entries), 'selected_bytes': total}))


if __name__ == '__main__':
    main()
