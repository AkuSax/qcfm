"""Smoke-gated AIM feature batch runner. It does not contain model code."""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; PACKAGE=HERE.parent; ENV=PACKAGE/'arm3_environment'
EXPORTS=PACKAGE/'arm1_atoms'/'safe_exports'; COVERAGE=PACKAGE/'arm2_chemistry'/'per_record_coverage.csv'
EXTRACTOR=ENV/'extract_aim_features.py'; WEIGHT=ENV/'weights'/'aimnet2_wb97m_d3_0.pt'
WEIGHT_SHA='f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28'; COMMIT='d14692839f4e5544f4eb38c15cf01298c27960f8'; MODEL='aimnet2-wb97m-d3_0'
MAX_CALLS=130

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows_csv(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def verify_smoke(path):
 r=json.loads(Path(path).read_text(encoding='utf-8'))
 if r.get('status')!='passed' or r.get('model')!=MODEL or r.get('weight_sha256')!=WEIGHT_SHA or r.get('metadata_parity')!='passed': raise ValueError('Three-ligand smoke receipt is not a verified pass')
 if r.get('input_ids')!=['57341_bound','68261_bound','570059_bound'] or r.get('molecule_evaluations',999)>18 or r.get('api_calls',999)>14: raise ValueError('Smoke receipt exceeds the frozen subset or budget')
 return r
def verify_cached(cid,state,manifest,outdir):
 d=Path(outdir)/f'{cid}_{state}'; feature=d/'features.npz'; receipt=d/'receipt.json'
 if not feature.exists() and not receipt.exists():return None
 if not feature.is_file() or not receipt.is_file(): raise ValueError(f'Incomplete cached output; preserve and inspect: {d}')
 r=json.loads(receipt.read_text(encoding='utf-8'))
 raw=Path(manifest).read_bytes()
 for k,v in {'status':'passed','id':f'{cid}_{state}','model':MODEL,'source_commit':COMMIT,'weight_sha256':WEIGHT_SHA,'weight_bytes':8836941,'input_sha256':hashlib.sha256(raw).hexdigest(),'feature_channels':256,'force_gradient_requested':True}.items():
  if r.get(k)!=v:raise ValueError(f'Cached receipt mismatch {d}: {k}')
 with np.load(feature,allow_pickle=False) as a:
  if a['features'].shape!=(len(a['source_atom_index']),256) or not np.isfinite(a['features']).all():raise ValueError(f'Cached features invalid: {d}')
  if r.get('features_npz_sha256')!=sha(feature):raise ValueError(f'Cached output hash missing/mismatch: {d}')
  for key in ('energy_native','charges','forces_native'):
   if key not in a.files or not np.isfinite(a[key]).all():raise ValueError(f'Cached {key} invalid: {d}')
 return {'feature_sha256':sha(feature),'receipt_sha256':sha(receipt),'manifest_sha256':sha(manifest),'state':state,'cid':cid}
def main():
 ap=argparse.ArgumentParser(); m=ap.add_mutually_exclusive_group(required=True);m.add_argument('--dry-run',action='store_true');m.add_argument('--execute',action='store_true')
 ap.add_argument('--smoke-receipt',type=Path,required=True);ap.add_argument('--manifest-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--ledger',type=Path,required=True);ap.add_argument('--records',type=Path,required=True);ap.add_argument('--device',choices=('cpu','cuda'),default='cpu')
 a=ap.parse_args()
 if a.execute: ap.error('Execution is blocked until each AIM attempt is journaled before the call and a cumulative 130-attempt cap is enforced across retries.')
 smoke=verify_smoke(a.smoke_receipt)
 records=rows_csv(a.records); cids=[str(r['pubchem_cid']) for r in records]
 if len(cids)!=65 or len(set(cids))!=65:raise ValueError('Frozen source record count/order changed')
 supported={p.stem for p in EXPORTS.glob('*.json')}
 if len(supported)!=62:raise ValueError('Compatible atom export count changed')
 all_inputs=[]; ledger=[]; cached=[]; planned=0
 for cid in cids:
  if cid not in supported:
   for st in ('bound','free'):ledger.append({'cid':cid,'state':st,'status':'HELD','failure_code':'STEREOCHEMISTRY_HOLD','reason':'Source stereochemistry conflict; not an AIM extraction input'})
   continue
  side=json.loads((EXPORTS/f'{cid}.json').read_text(encoding='utf-8')); source_npz=EXPORTS/f'{cid}.npz'
  if sha(source_npz)!=side['npz_sha256']:raise ValueError(f'Atom export hash mismatch: {cid}')
  for st in ('bound','free'):
   mp=Path(a.manifest_dir)/f'{cid}_{st}.json'
   if not mp.is_file():raise FileNotFoundError(f'Missing future prepared manifest: {mp}')
   man=json.loads(mp.read_text(encoding='utf-8'))
   if man.get('id')!=f'{cid}_{st}' or man.get('state')!=st or man.get('parent_cid')!=cid or man.get('coordinate_unit')!='angstrom' or man.get('multiplicity')!=1:raise ValueError(f'Manifest identity/schema mismatch: {mp}')
   z=np.asarray(man['atomic_numbers'],dtype=np.int64);xyz=np.asarray(man['coord_angstrom'],dtype=np.float64);heavy=np.asarray(man['heavy_atom_indices'],dtype=np.int64);src=np.asarray(man['source_atom_index'],dtype=np.int64)
   if xyz.shape!=(len(z),3) or not np.isfinite(xyz).all() or not np.array_equal(np.sort(heavy),np.flatnonzero(z>1)) or len(set(src.tolist()))!=len(src):raise ValueError(f'Manifest atom arrays invalid: {mp}')
   with np.load(source_npz,allow_pickle=False) as e:
    if not np.array_equal(z[heavy],e['atomic_numbers']) or not np.array_equal(src,e['source_atom_indices']):raise ValueError(f'Manifest/source atom map mismatch: {cid}/{st}')
    if st=='bound' and not np.array_equal(xyz[heavy],e['x_pred_heavy_coords']):raise ValueError(f'Bound heavy coordinates changed: {cid}')
   got=verify_cached(cid,st,mp,a.output)
   if got:
    cached.append(got);ledger.append({'cid':cid,'state':st,'status':'CACHED_PASS','failure_code':'','reason':''})
   else:
    all_inputs.append(str(mp));planned+=1;ledger.append({'cid':cid,'state':st,'status':'PLANNED' if a.dry_run else 'PENDING','failure_code':'','reason':''})
 if planned>MAX_CALLS:raise ValueError(f'Planned {planned} evaluations exceeds {MAX_CALLS}')
 if not WEIGHT.is_file() or WEIGHT.stat().st_size!=8836941 or sha(WEIGHT)!=WEIGHT_SHA:raise ValueError('Pinned weight path/size/hash check failed')
 if a.dry_run:
  print(json.dumps({'status':'PASS','dry_run_only':True,'smoke_status':smoke['status'],'compatible_records':62,'held_records':3,'cached_evaluations':len(cached),'planned_new_evaluations':planned,'max_new_evaluations':MAX_CALLS,'force_grad':True,'model_calls':0,'input_manifests':all_inputs},indent=2));return
 # Persist planned work before launch. Parent run_bounded.py owns hard timeout/child cleanup.
 a.output.mkdir(parents=True,exist_ok=True);a.ledger.parent.mkdir(parents=True,exist_ok=True)
 with a.ledger.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['cid','state','status','failure_code','reason']);w.writeheader();w.writerows(ledger)
 run={'status':'running','model':MODEL,'source_commit':COMMIT,'weight_sha256':WEIGHT_SHA,'smoke_receipt_sha256':sha(a.smoke_receipt),'planned_new_evaluations':planned,'evaluation_cap':MAX_CALLS,'input_manifests':[{ 'path':p,'sha256':sha(p)} for p in all_inputs],'force_grad':True,'device':a.device,'hard_timeout_seconds':'Set by parent run_bounded.py'}
 (a.output/'batch_run_receipt.json').write_text(json.dumps(run,indent=2)+'\n',encoding='utf-8')
 proc=None; error=None
 try:
  if all_inputs:
   cmd=[sys.executable,str(EXTRACTOR),'--execute','--input',*all_inputs,'--output',str(a.output),'--device',a.device,'--force-grad']
   proc=subprocess.run(cmd,check=False)
   if proc.returncode: error=f'extractor exit code {proc.returncode}'
 except Exception as exc:error=f'{type(exc).__name__}: {exc}'
 finally:
  # Re-scan both states. Retain all prior successes and mark each incomplete state.
  final=[]
  for entry in ledger:
   if entry['status']=='HELD':final.append(entry);continue
   cid,st=entry['cid'],entry['state'];mp=Path(a.manifest_dir)/f'{cid}_{st}.json'
   try:
    feature=Path(a.output)/f'{cid}_{st}'/'features.npz'; receipt=feature.parent/'receipt.json'
    if feature.is_file() and receipt.is_file():
     rr=json.loads(receipt.read_text(encoding='utf-8'))
     with np.load(feature,allow_pickle=False) as qa:
      if qa['features'].shape!=(len(qa['source_atom_index']),256) or not np.isfinite(qa['features']).all() or not np.isfinite(qa['energy_native']).all() or not np.isfinite(qa['charges']).all() or not np.isfinite(qa['forces_native']).all(): raise ValueError('Post-run AIM arrays are invalid')
     rr['status']='passed'; rr['features_npz_sha256']=sha(feature); receipt.write_text(json.dumps(rr,indent=2)+'\n',encoding='utf-8')
    cache=verify_cached(cid,st,mp,a.output)
    if cache:final.append({'cid':cid,'state':st,'status':'PASS','failure_code':'','reason':''})
    else:final.append({'cid':cid,'state':st,'status':'FAILED','failure_code':'NO_OUTPUT','reason':error or 'Extractor produced no feature output'})
   except Exception as exc:
    rp=Path(a.output)/f'{cid}_{st}'/'receipt.json'
    if rp.is_file():
     try:
      rr=json.loads(rp.read_text(encoding='utf-8'));rr['status']='failed';rr['failure']=str(exc);rp.write_text(json.dumps(rr,indent=2)+'\n',encoding='utf-8')
     except Exception: pass
    final.append({'cid':cid,'state':st,'status':'FAILED','failure_code':type(exc).__name__,'reason':str(exc)})
  with a.ledger.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=['cid','state','status','failure_code','reason']);w.writeheader();w.writerows(final)
  run.update(status='failed' if error else 'passed',failure=error,completed_states=sum(x['status'] in ('PASS','CACHED_PASS') for x in final),failed_states=sum(x['status']=='FAILED' for x in final),held_states=sum(x['status']=='HELD' for x in final))
  (a.output/'batch_run_receipt.json').write_text(json.dumps(run,indent=2)+'\n',encoding='utf-8')
 if error:raise RuntimeError(error)
 print(json.dumps(run,indent=2))
if __name__=='__main__':main()
