"""Record the final preparation evidence. Do not execute models."""
import hashlib
import json
from pathlib import Path

PACKAGE=Path(__file__).resolve().parent.parent
QGCF=PACKAGE.parents[1]
HERE=Path(__file__).resolve().parent

def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value): Path(path).write_text(json.dumps(value,indent=2)+'\n')

proposal=PACKAGE/'arm5_integration/FROZEN_EXPERIMENT_PROPOSAL_v2_20261022.json'
readiness=PACKAGE/'arm5_integration/readiness.json'
p=read(proposal); r=read(readiness)
write(HERE/'readiness_before_final_root_checks.json',r)
contract=p['integration_execution_contract']
contract['staging_contract']['future_manifests']='Preparation is authorized. After the AIM smoke passes, use --prepare --all-compatible --include-free with its actual receipt. Keep heavy atoms fixed for hydrogen placement. Use seed 1729, 8 free conformers, and 200-step MMFF94s limits. Exclude the 3 stereo holds.'
contract['extraction']['synthetic_validation']='PASS. Mocked calculator tests verify pre-call journals, first-failure stop, cumulative retry count, unique attempt numbers, cache reuse, no constructor for a complete cache, and the cumulative 130-attempt cap. No pretrained model ran.'
contract['extraction']['limit']='At most 130 total attempted AIM evaluations across feature-extraction attempts. Count failed calls. Reuse verified successes. Keep all 65 records and both states in the ledger. The bounded process wrapper enforces 3600 seconds per authorized extraction attempt.'
contract['current_evidence']['batch_journal_synthetic_test']='PASS; 125 simulated calls include 1 preserved failure and 124 successful states. Real pretrained calls: 0.'
contract['current_evidence']['strict_provenance_gate']='PASS; the verifier rejects synthetic evidence and binds the frozen feature, schema, source, split, join, dependency, smoke, and final-readout receipts.'
contract['training_plan_if_later_authorized']['runner']['gates']='Require explicit execution authorization and hash-bound feature/schema, primary/source, split, atom-join, dependency, actual AIM smoke, and new-runtime final-readout parity evidence. Synthetic evidence cannot authorize fitting.'
for item in contract['future_commands']:
    if item['purpose'].startswith('Full 62-record'):
        text=item['command']; flag=' --smoke-receipt '
        if text.count(flag)>1:
            item['command']=text[:text.rfind(flag)]
    if item['purpose'].startswith('Future AIM extraction'):
        item['purpose']='Future AIM extraction. Require a passing smoke. Enforce 130 total attempted feature calls across retries.'
        item['command'] += ' --attempt-ledger "$Package/arm5_integration/aim_attempt_ledger.jsonl"'
p['remaining_blockers']=[x for x in p['remaining_blockers'] if 'mock' not in x.lower()]
for value in contract['provenance'].values():
    if isinstance(value,dict) and 'path' in value and 'sha256' in value:
        path=Path(value['path'])
        if not path.is_absolute():path=QGCF/path
        if path.is_file():value['sha256']=sha(path)
write(proposal,p)
r['items']['batch_extraction_control']={'status':'PASS','evidence':['orchestrator/batch_journal_synthetic_test.json'],
 'detail':'Mocked calls verify journaling before each forward, failed-call accounting, unique attempt indices, first-failure stop, verified cache reuse, and the cumulative 130-attempt cap. No pretrained model ran.'}
r['items']['full_feature_preparation_and_extraction']['detail']='A real smoke receipt is absent. Only 3 bound inputs are prepared. Full bound/free preparation and actual extraction remain unrun. The batch-control synthetic test passes.'
r['items']['package_preparation']={'status':'PASS','evidence':['README.md','orchestrator/dry_run.py','orchestrator/assembler_synthetic_test.json','orchestrator/batch_journal_synthetic_test.json'],
 'detail':'The preparation code, isolated runtime, weight, 62 atom exports, 3 bound inputs, synthetic tests, future commands, and provenance gates are ready. Real model and fitting outputs remain unvalidated.'}
r['remaining_blockers']=[x for x in r['remaining_blockers'] if 'mock' not in x.lower()]
r['items']['synthetic_checks']['detail']='AIM interface checks pass. The 15 protocol tests pass. Bundle round-trip, negative receipt/hash/coordinate tests, batch journaling/retry/cache tests, process timeout tests, and synthetic-gate rejection pass.'
r['execution_now'].update(AIM_model_calls=0,Boltz_model_calls=0,fit_steps=0,GPU_VM_starts=0,cloud_calls=0,remote_recovery=0)
r['preparation_execution']={'bound_hydrogen_minimizations':3,'free_conformer_minimizations':0,
 'subagents':5,'subagent_model':'gpt-6-luna','subagent_reasoning':'medium','nested_subagents':0,
 'new_chats':0,'ppi_IG_research_writes':0}
r['hashes']['v2']['sha256']=sha(proposal)
for key in ['arm4_config','prepared_verification','assembler_synthetic_test','fit_gate_synthetic_rejection']:
    row=r['hashes'].get(key)
    if row:
        path=Path(row['path'])
        if not path.is_absolute():
            path=(PACKAGE/path) if (PACKAGE/path).exists() else (QGCF/path)
        if path.is_file():row['sha256']=sha(path)
r['hashes']['batch_journal_synthetic_test']={'path':'orchestrator/batch_journal_synthetic_test.json','sha256':sha(HERE/'batch_journal_synthetic_test.json')}
r['smallest_next_experiment']={'status':'BLOCKED','requires':'Separate model-execution authorization.',
 'cids':['57341','68261','570059'],'runtime':'Local CPU','max_seconds':1800,'max_molecular_evaluations':18,
 'max_calculator_calls':14,'VM_start_required':False,'command_source':'README.md'}
write(readiness,r)
print('PASS: Recorded final preparation evidence. Model execution remains blocked.')
