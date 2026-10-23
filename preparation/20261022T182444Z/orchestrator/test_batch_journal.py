"""Test batch control flow with a fake calculator. Do not load model weights."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import types
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'orchestrator/synthetic_batch_test'
OUT.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('batch',ROOT/'arm5_integration/extract_feature_batch.py')
batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
sys.path.insert(0,str(ROOT/'arm3_environment'))
import extract_aim_features as real_interface
fixture=ROOT/'orchestrator/synthetic_assembler_test/attempt01'
manifests=fixture/'synthetic_manifests'
records=Path('C:/Users/Saxen/ppi_IG/results/representation_diagnosis/20261015/attempt02/gcp_a100_retry_01/collected/records.csv')
journal=OUT/'attempts.jsonl'
calls=[]
state={'fail_call':2,'constructors':0}

class FakeModel:
    def eval(self): return self
    def parameters(self): return []

class FakeCalculator:
    def __init__(self,**kwargs):
        self.model=FakeModel();state['constructors']+=1

class FakeFactory:
    @staticmethod
    def calculator_class(): return FakeCalculator

def fake_extract(path,output,device,force_grad,calc=None):
    manifest=json.loads(Path(path).read_text())
    attempts=[x for x in map(json.loads,journal.read_text().splitlines()) if 'attempt' in x]
    # The journal must exist before every simulated forward.
    assert attempts[-1]['status']=='running'
    assert attempts[-1]['input_sha256']==batch.sha(path)
    calls.append(manifest['id'])
    if len(calls)==state['fail_call']: raise RuntimeError('Intentional synthetic call failure.')
    z,xyz,heavy=real_interface.validate_manifest(manifest)
    target=Path(output)/manifest['id'];target.mkdir()
    fp=target/'features.npz'
    np.savez_compressed(fp,features=np.zeros((len(heavy),256),dtype='f4'),
        source_atom_index=np.asarray(manifest['source_atom_index']),heavy_atom_indices=heavy,
        atomic_numbers=z,coord_angstrom=xyz,energy_native=np.array([0.0]),charges=np.zeros(len(z)),
        forces_native=np.zeros_like(xyz))
    receipt={'synthetic_only':True,'id':manifest['id'],'model':batch.MODEL,'source_commit':batch.COMMIT,
        'weight_sha256':batch.WEIGHT_SHA,'weight_bytes':8836941,'input_sha256':batch.sha(path),
        'feature_channels':256,'force_gradient_requested':True}
    (target/'receipt.json').write_text(json.dumps(receipt))

fake_module=types.SimpleNamespace(validate_manifest=real_interface.validate_manifest,
    AIMFeaturesCalculator=FakeFactory,extract=fake_extract)
sys.modules['extract_aim_features']=fake_module
# This test isolates batch control flow. The real smoke gate rejects synthetic evidence.
batch.smoke_gate=lambda path: {'synthetic_only':True,'status':'passed'}
base=['batch','--execute','--smoke-receipt',str(fixture/'synthetic_smoke_receipt.json'),
      '--manifest-dir',str(manifests),'--output',str(OUT/'outputs'),'--ledger',str(OUT/'states.csv'),
      '--attempt-ledger',str(journal),'--records',str(records)]

def invoke(args):
    sys.argv=args
    with contextlib.redirect_stdout(io.StringIO()): batch.main()

try:
    try: invoke(base)
    except RuntimeError as exc: assert 'First AIM feature call failed' in str(exc)
    else: raise AssertionError('The failed synthetic call did not stop the batch.')
    assert len(calls)==2
    first_receipt=json.loads((OUT/'outputs/batch_run_receipt.json').read_text())
    (OUT/'preserved_first_failure.json').write_text(json.dumps(first_receipt,indent=2)+'\n')
    assert first_receipt['status']=='failed' and first_receipt['attempt_count_total']==2
    # Concrete remedy: remove the intentional fake-calculator failure.
    state['fail_call']=-1
    invoke(base)
    assert len(calls)==125
    attempts=[x for x in map(json.loads,journal.read_text().splitlines()) if 'attempt' in x]
    assert [x['attempt'] for x in attempts]==list(range(1,126))
    assert len({x['attempt'] for x in attempts})==125
    constructors=state['constructors']
    invoke(base)
    assert len(calls)==125 and state['constructors']==constructors
    # Reject exhausted attempt budgets before any calculator operation.
    exhausted=OUT/'exhausted_attempts.jsonl'
    exhausted.write_text(''.join(json.dumps({'attempt':i+1})+'\n' for i in range(131)))
    capped=base.copy();capped[capped.index('--attempt-ledger')+1]=str(exhausted)
    try: invoke(capped)
    except ValueError as exc: assert 'Cumulative' in str(exc)
    else: raise AssertionError('The exhausted synthetic budget passed.')
    assert len(calls)==125 and state['constructors']==constructors
    result={'status':'PASS','synthetic_only':True,'simulated_calls':125,'failed_simulated_calls':1,
        'successful_states':124,'checks':['pre-call journal','first-failure stop','cumulative retry count',
        'unique attempt numbers','verified cache reuse','no model construction for full cache','cumulative 130-call cap'],
        'pretrained_model_calls':0,'weights_loaded':False,'fits':0}
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'orchestrator/batch_journal_synthetic_test.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
except Exception as exc:
    (OUT/'FAILED_test_receipt.json').write_text(json.dumps({'status':'BLOCKED','synthetic_only':True,
        'error':str(exc),'pretrained_model_calls':0,'fits':0},indent=2)+'\n')
    raise
