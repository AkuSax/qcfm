"""Gate a saved final-readout parity check. Never loads a Boltz checkpoint."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
RECORDS=Path(r'C:\Users\Saxen\ppi_IG\results\representation_diagnosis\20261015\attempt02\gcp_a100_retry_01\collected\records.csv')
INPUTS=Path(r'C:\Users\Saxen\ppi_IG\results\representation_diagnosis\20261015\attempt02\gcp_a100_retry_01\collected\exact_inputs.npz')
RECORDS_SHA='b6c9473de5dd8d09af35f5b5e42983498ee468a15e96c19b0ba066c3cc20791c'
INPUTS_SHA='5dcbd0011ed01995fe76a71c12f62073be5a7905c7cb6e16e9cee7402fea1540'
ATOL=1e-5; RTOL=1e-5

def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_rows(p):
    with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def main():
    ap=argparse.ArgumentParser(); m=ap.add_mutually_exclusive_group(required=True); m.add_argument('--dry-run',action='store_true');m.add_argument('--execute',action='store_true')
    ap.add_argument('--records',type=Path,default=RECORDS);ap.add_argument('--exact-inputs',type=Path,default=INPUTS);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    for p,expected in ((a.records,RECORDS_SHA),(a.exact_inputs,INPUTS_SHA)):
        if not p.is_file(): raise FileNotFoundError(p)
        if digest(p)!=expected: raise ValueError(f'Pinned SHA-256 mismatch: {p}')
    rows=load_rows(a.records)
    if len(rows)!=65 or not {'member1','member2','unchanged'}.issubset(rows[0]): raise ValueError('Saved final-readout fields/row count changed')
    if not a.execute:
        print(json.dumps({'status':'PASS','dry_run_only':True,'records':65,'atol':ATOL,'rtol':RTOL,'torch_minimum':'2.10','model_calls':0,'boltz_checkpoint_loaded':False,'future_output':str(a.output)},indent=2));return
    if a.output.exists(): raise FileExistsError(f'Output path exists: {a.output}')
    import torch
    if tuple(map(int,torch.__version__.split('+')[0].split('.')[:2])) < (2,10): raise RuntimeError('Run in torch >=2.10 environment')
    torch.set_grad_enabled(False)
    with np.load(a.exact_inputs,allow_pickle=False) as d:
        x1=np.asarray(d['x1'],np.float32);x2=np.asarray(d['x2'],np.float32)
        weights={k:np.asarray(d[k],np.float32) for k in d.files if k.startswith(('m1::','m2::'))}
    if x1.shape!=(65,384) or x2.shape!=(65,384):raise ValueError('Saved input arrays have changed')
    def head(prefix,x):
        t=torch.as_tensor(x,dtype=torch.float32)
        for layer in ('0','2','4'):
            w=torch.as_tensor(weights[f'{prefix}::{layer}.weight']);b=torch.as_tensor(weights[f'{prefix}::{layer}.bias'])
            t=torch.nn.functional.linear(t,w,b)
            if layer!='4':t=torch.relu(t)
        return t.squeeze(-1).cpu().numpy()
    p1=head('m1',x1);p2=head('m2',x2);ens=(p1+p2)/2
    refs=[np.asarray([float(r[k]) for r in rows],np.float64) for k in ('member1','member2','unchanged')]
    preds=[p1,p2,ens]; names=['member1','member2','unchanged ensemble']; outcomes=[]
    for name,pred,ref in zip(names,preds,refs):
        ok=np.isfinite(pred).all() and np.allclose(pred,ref,atol=ATOL,rtol=RTOL)
        outcomes.append({'name':name,'passed':bool(ok),'max_abs_error':float(np.max(np.abs(pred-ref))),'atol':ATOL,'rtol':RTOL})
        if not ok: break
    a.output.parent.mkdir(parents=True,exist_ok=True)
    receipt={'status':'passed' if len(outcomes)==3 and all(x['passed'] for x in outcomes) else 'failed','model_calls':2,'scope':'saved final MLPs only; no Boltz checkpoint','records':65,'input_sha256':INPUTS_SHA,'records_sha256':RECORDS_SHA,'torch':torch.__version__,'checks':outcomes}
    a.output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt,indent=2))
    if receipt['status']!='passed':raise SystemExit(1)
if __name__=='__main__': main()
