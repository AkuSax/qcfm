"""Validate the prepared package. Do not load model weights or fit models."""
import hashlib
import json
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent.parent
ROOT=PACKAGE.parents[1]

def read(relative):
    return json.loads((PACKAGE/relative).read_text(encoding='utf-8-sig'))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    checks=[]
    for path in ['orchestrator/primary_verification.json','orchestrator/export_verification.json',
                 'orchestrator/prepared_input_verification.json','orchestrator/chemistry_synthetic_test.json',
                 'orchestrator/assembler_synthetic_test.json','orchestrator/batch_journal_synthetic_test.json',
                 'orchestrator/fit_gate_synthetic_rejection.json','orchestrator/bounded_process_synthetic_test.json',
                 'orchestrator/fit_authorization_refusal.json','orchestrator/preserved_input_verification.json']:
        data=read(path)
        checks.append({'check':path,'status':'PASS' if data['status']=='PASS' else 'BLOCKED'})
    dependency=read('arm3_environment/dependency_receipt.json')
    weight=Path(dependency['weight']['path'])
    checks.append({'check':'staged_weight','status':'PASS' if weight.stat().st_size==8836941 and sha(weight)==dependency['weight']['expected_sha256'] else 'BLOCKED'})
    v1=json.loads((ROOT/'FROZEN_EXPERIMENT_PROPOSAL_v1_20261022.json').read_text())
    v2=read('arm5_integration/FROZEN_EXPERIMENT_PROPOSAL_v2_20261022.json')
    checks.append({'check':'frozen_decision_thresholds','status':'PASS' if v1['practical_decision_rules']==v2['practical_decision_rules'] else 'BLOCKED'})
    primary=read('orchestrator/primary_verification.json')
    checks.append({'check':'primary_pair_count','status':'PASS' if primary['within_group_pairs']==136 and primary['sensitivity_pairs']==129 else 'BLOCKED'})
    for row in read('orchestrator/prepared_input_verification.json')['records']:
        manifest=PACKAGE/'arm5_integration/prepared_manifests'/f"{row['cid']}_bound.json"
        checks.append({'check':f"prepared_manifest_{row['cid']}",'status':'PASS' if sha(manifest)==row['manifest_sha256'] else 'BLOCKED'})
    readiness=read('arm5_integration/readiness.json')
    checks.append({'check':'readiness_states','status':'PASS' if all(x['status'] in {'PASS','BLOCKED','NOT_REQUIRED'} for x in readiness['items'].values()) else 'BLOCKED'})
    checks.append({'check':'zero_pretrained_execution','status':'PASS' if all(v==0 for v in readiness['execution_now'].values()) else 'BLOCKED'})
    report={'status':'PASS' if all(x['status']=='PASS' for x in checks) else 'BLOCKED',
            'preparation_checks':checks,'execution_ready':False,'pretrained_model_calls':0,'fits':0,'VM_starts':0,
            'next_check':'A separately authorized 3-ligand CPU AIM smoke. Use at most 18 molecular evaluations and 1800 seconds.'}
    (PACKAGE/'orchestrator/dry_run_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','execution_ready','pretrained_model_calls','fits','VM_starts']}))
    return 0 if report['status']=='PASS' else 1

if __name__=='__main__': raise SystemExit(main())
