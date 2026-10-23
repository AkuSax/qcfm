"""Bind fitting gates to actual evidence. Do not call a model."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

PACKAGE = Path(__file__).resolve().parent.parent
WEIGHT = 'f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28'
COMMIT = 'd14692839f4e5544f4eb38c15cf01298c27960f8'
RECORDS = 'b6c9473de5dd8d09af35f5b5e42983498ee468a15e96c19b0ba066c3cc20791c'
INPUTS = '5dcbd0011ed01995fe76a71c12f62073be5a7905c7cb6e16e9cee7402fea1540'
SPLITS = '137c76616c1940cdddabd021bb6d5d1f9b1ead0a78dddd450de73ac88dbb0888'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def evidence(gate, name):
    path = Path(gate[f'{name}_path']).resolve()
    if not path.is_relative_to(PACKAGE):
        raise ValueError(f'Keep {name} evidence under this preparation package.')
    if sha(path) != gate[f'{name}_sha256']:
        raise ValueError(f'{name} evidence hash changed.')
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    if data.get('synthetic_only'):
        raise ValueError('Synthetic evidence cannot authorize fitting.')
    return data

def verify(gate, features):
    for key in ['execution_authorized','joins_verified','aim_runtime_verified','feature_manifest_frozen']:
        if gate.get(key) is not True: raise ValueError(f'Gate {key} did not pass.')
    features = Path(features).resolve()
    if not features.is_relative_to(PACKAGE) or sha(features) != gate['feature_sha256']:
        raise ValueError('Frozen feature hash or path changed.')
    if gate.get('split_sha256') != SPLITS: raise ValueError('Frozen split hash changed.')
    schema = evidence(gate,'feature_schema')
    if schema['output_hashes']['npz_sha256'] != gate['feature_sha256']:
        raise ValueError('Feature schema does not bind this NPZ.')
    expected_sources = {'records_sha256':RECORDS,'exact_inputs_sha256':INPUTS,
                        'chemistry_sha256':sha(PACKAGE/'arm2_chemistry/per_record_coverage.csv')}
    if any(schema['input_hashes'].get(key) != value for key,value in expected_sources.items()):
        raise ValueError('Feature source hashes changed.')
    primary = evidence(gate,'primary_verification')
    joins = evidence(gate,'join_verification')
    if primary.get('status') != 'PASS' or joins.get('status') != 'PASS' or joins.get('candidate_exports') != 62:
        raise ValueError('Source integrity or atom correspondence did not pass.')
    allowed = {row['cid'] for row in joins['records'] if all(row['checks'].values())}
    with np.load(features,allow_pickle=False) as a:
        valid = a['feature_valid'].astype(bool)
        if valid.shape != (65,) or valid.sum() < 59:
            raise ValueError('Feature coverage did not reach 90 percent of 65 records.')
        actual = set(a['cids'][valid].tolist())
        if not actual.issubset(allowed): raise ValueError('A feature row lacks a verified atom join.')
    smoke = evidence(gate,'aim_smoke')
    if smoke.get('status') != 'passed' or smoke.get('model') != 'aimnet2-wb97m-d3_0' or smoke.get('weight_sha256') != WEIGHT:
        raise ValueError('The selected AIM model smoke did not pass.')
    if smoke.get('metadata_parity') != 'passed' or smoke.get('molecule_evaluations',999) > 18 or smoke.get('api_calls',999) > 14:
        raise ValueError('AIM smoke parity or budget did not pass.')
    if smoke.get('input_ids') != ['57341_bound','68261_bound','570059_bound']:
        raise ValueError('AIM smoke identities changed.')
    dependency = evidence(gate,'dependency')
    if dependency.get('aimnet_source_commit') != COMMIT or dependency['weight']['actual_sha256'] != WEIGHT:
        raise ValueError('The pinned extraction dependency changed.')
    parity = evidence(gate,'final_readout_parity')
    if parity.get('status') != 'passed' or parity.get('input_sha256') != INPUTS or parity.get('records_sha256') != RECORDS:
        raise ValueError('The new runtime final-readout parity did not pass.')
    return gate

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--aim-smoke',type=Path,required=True)
    parser.add_argument('--final-readout-parity',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--execution-authorized',action='store_true')
    args = parser.parse_args()
    if not args.execution_authorized:
        parser.error('Record execution authorization only after the user grants it.')
    gate = {'execution_authorized':True,'joins_verified':True,'aim_runtime_verified':True,
            'feature_manifest_frozen':True,'feature_sha256':sha(args.features),'split_sha256':SPLITS}
    paths = {'feature_schema':args.features.with_suffix('.schema.json'),
             'primary_verification':PACKAGE/'orchestrator/primary_verification.json',
             'join_verification':PACKAGE/'orchestrator/export_verification.json',
             'aim_smoke':args.aim_smoke,'final_readout_parity':args.final_readout_parity,
             'dependency':PACKAGE/'arm3_environment/dependency_receipt.json'}
    for name,path in paths.items():
        gate[f'{name}_path'] = str(path.resolve())
        gate[f'{name}_sha256'] = sha(path)
    verify(gate,args.features)
    output = args.output.resolve()
    if not output.is_relative_to(PACKAGE) or output.exists():
        raise ValueError('Use a new gate file under this preparation package.')
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(gate,indent=2)+'\n')
    print('PASS: Evidence binds the future fitting command.')

if __name__=='__main__':
    main()
