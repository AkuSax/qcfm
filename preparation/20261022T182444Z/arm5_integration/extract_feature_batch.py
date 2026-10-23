"""Smoke-gated AIM feature batch. This module performs model calls only with --execute."""
from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; PACKAGE=HERE.parent; ENV=PACKAGE/'arm3_environment'
EXPORTS=PACKAGE/'arm1_atoms'/'safe_exports'; COVERAGE=PACKAGE/'arm2_chemistry'/'per_record_coverage.csv'
EXTRACTOR=ENV/'extract_aim_features.py'; WEIGHT=ENV/'weights'/'aimnet2_wb97m_d3_0.pt'
WEIGHT_SHA='f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28'; COMMIT='d14692839f4e5544f4eb38c15cf01298c27960f8'; MODEL='aimnet2-wb97m-d3_0'; MAX_CALLS=130

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def loadcsv(p):
 with Path(p).open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def smoke_gate(path):
 r=json.loads(Path(path).read_text(encoding='utf-8'))
 if r.get('synthetic_only'):raise ValueError('Synthetic smoke evidence cannot authorize extraction')
 if r.get('status')!='passed' or r.get('model')!=MODEL or r.get('weight_sha256')!=WEIGHT_SHA or r.get('metadata_parity')!='passed' or r.get('input_ids')!=['57341_bound','68261_bound','570059_bound'] or r.get('molecule_evaluations',999)>18 or r.get('api_calls',999)>14:raise ValueError('Three-CID AIM smoke gate failed')
 return r
def validate_output(cid,state,manifest,outdir,mark=False):
 d=Path(outdir)/f'{cid}_{state}'; fp=d/'features.npz'; rp=d/'receipt.json'
 if not fp.is_file() or not rp.is_file():return None
 r=json.loads(rp.read_text(encoding='utf-8')); raw=Path(manifest).read_bytes()
 for k,v in {'id':f'{cid}_{state}','model':MODEL,'source_commit':COMMIT,'weight_sha256':WEIGHT_SHA,'weight_bytes':8836941,'input_sha256':hashlib.sha256(raw).hexdigest(),'feature_channels':256,'force_gradient_requested':True}.items():
  if r.get(k)!=v:raise ValueError(f'Output receipt mismatch for {cid} {state}: {k}')
 with np.load(fp,allow_pickle=False) as a:
  feats=np.asarray(a['features']);src=np.asarray(a['source_atom_index']);h=np.asarray(a['heavy_atom_indices']);z=np.asarray(a['atomic_numbers']);xyz=np.asarray(a['coord_angstrom']);e=np.asarray(a['energy_native']);q=np.asarray(a['charges']);force=np.asarray(a['forces_native'])
  if feats.shape!=(len(src),256) or h.shape!=src.shape or not np.isfinite(feats).all() or not np.isfinite(xyz).all() or not np.isfinite(e).all() or not np.isfinite(q).all() or not np.isfinite(force).all() or force.shape!=xyz.shape:raise ValueError(f'Output arrays invalid for {cid} {state}')
  manifest_obj=json.loads(raw)
  if not np.array_equal(z,np.asarray(manifest_obj['atomic_numbers'])) or not np.array_equal(xyz,np.asarray(manifest_obj['coord_angstrom'])) or not np.array_equal(h,np.asarray(manifest_obj['heavy_atom_indices'])):raise ValueError(f'Output full atom array differs from prepared manifest {cid} {state}')
  if not np.array_equal(z[h],np.asarray(manifest_obj['atomic_numbers'])[np.asarray(manifest_obj['heavy_atom_indices'])]):raise ValueError(f'Output element order mismatch {cid} {state}')
  if not np.array_equal(src,np.asarray(json.loads(raw)['source_atom_index'])):raise ValueError(f'Output source map mismatch {cid} {state}')
 digest=sha(fp)
 if r.get('status')=='failed':raise ValueError(f'Preserve failed extraction receipt for {cid} {state}')
 if r.get('status')=='passed' and r.get('features_npz_sha256')!=digest:raise ValueError(f'Cached feature hash mismatch for {cid} {state}')
 if mark and r.get('status') is None:
  r['status']='passed';r['features_npz_sha256']=digest;rp.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
 if r.get('status')!='passed':raise ValueError(f'Cached extraction is not a verified pass for {cid} {state}')
 return {'cid':cid,'state':state,'feature_sha256':digest,'receipt_sha256':sha(rp),'manifest_sha256':sha(manifest)}
def main():
 ap=argparse.ArgumentParser();g=ap.add_mutually_exclusive_group(required=True);g.add_argument('--dry-run',action='store_true');g.add_argument('--execute',action='store_true')
 ap.add_argument('--smoke-receipt',type=Path,required=True);ap.add_argument('--manifest-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--ledger',type=Path,required=True);ap.add_argument('--attempt-ledger',type=Path,default=HERE/'aim_attempt_ledger.jsonl');ap.add_argument('--records',type=Path,required=True);ap.add_argument('--device',choices=('cpu','cuda'),default='cpu');a=ap.parse_args()
 records=loadcsv(a.records);cids=[str(r['pubchem_cid']) for r in records];chem={str(r['pubchem_cid']):r for r in loadcsv(COVERAGE)}
 if len(cids)!=65 or len(set(cids))!=65 or len(chem)!=65:raise ValueError('Frozen record or chemistry row count changed')
 supported={p.stem for p in EXPORTS.glob('*.json')}
 if len(supported)!=62:raise ValueError('Compatible export count changed')
 smoke=None; smoke_error=None
 try:smoke=smoke_gate(a.smoke_receipt)
 except Exception as e:smoke_error=f'{type(e).__name__}: {e}'
 if a.execute and smoke is None:raise ValueError(smoke_error)
 sys.path.insert(0,str(ENV));import extract_aim_features as aim
 manifests=[];missing=[];ledger=[];cache=[]
 for cid in cids:
  states=('bound','free') if cid in supported else ()
  if cid not in supported:
   for st in ('bound','free'):ledger.append({'cid':cid,'state':st,'status':'HELD','failure_code':'STEREOCHEMISTRY_HOLD','reason':'Source stereo conflict'})
   continue
  side=json.loads((EXPORTS/f'{cid}.json').read_text(encoding='utf-8'));ep=EXPORTS/f'{cid}.npz'
  if sha(ep)!=side['npz_sha256']:raise ValueError(f'Boltz export SHA mismatch {cid}')
  for st in states:
   mp=a.manifest_dir/f'{cid}_{st}.json'
   if not mp.is_file():missing.append(str(mp));ledger.append({'cid':cid,'state':st,'status':'MISSING','failure_code':'NO_MANIFEST','reason':'Future bound/free preparation is incomplete'});continue
   m=json.loads(mp.read_text(encoding='utf-8'));aim.validate_manifest(m)
   if m.get('id')!=f'{cid}_{st}' or m.get('parent_cid')!=cid or m.get('state')!=st or m.get('coordinate_unit','angstrom')!='angstrom' or m.get('multiplicity',1)!=1 or int(m['formal_charge'])!=int(chem[cid]['total_formal_charge']):raise ValueError(f'Manifest metadata mismatch {cid} {st}')
   with np.load(ep,allow_pickle=False) as e:
    if not np.array_equal(np.asarray(m['atomic_numbers'])[m['heavy_atom_indices']],e['atomic_numbers']) or not np.array_equal(m['source_atom_index'],e['source_atom_indices']):raise ValueError(f'Atom join mismatch {cid} {st}')
    if st=='bound' and not np.array_equal(np.asarray(m['coord_angstrom'])[m['heavy_atom_indices']],e['x_pred_heavy_coords']):raise ValueError(f'Bound x_pred changed {cid}')
   try:q=validate_output(cid,st,mp,a.output)
   except Exception as e:raise ValueError(f'Preserved output needs review; no overwrite: {cid} {st}: {e}')
   if q:cache.append(q);ledger.append({'cid':cid,'state':st,'status':'CACHED_PASS','failure_code':'','reason':''})
   else:manifests.append((cid,st,mp));ledger.append({'cid':cid,'state':st,'status':'PENDING','failure_code':'','reason':''})
 if a.dry_run:
  print(json.dumps({'status':'PASS' if smoke and not missing else 'BLOCKED','smoke_gate':'PASS' if smoke else smoke_error,'compatible_records':62,'held_records':3,'cached_states':len(cache),'missing_manifests':missing,'planned_states':len(manifests),'planned_evaluations':len(manifests),'cumulative_cap':MAX_CALLS,'model_calls':0},indent=2));return
 if missing:raise FileNotFoundError('All 124 manifests are required before extraction')
 a.output.mkdir(parents=True,exist_ok=True);a.ledger.parent.mkdir(parents=True,exist_ok=True);a.attempt_ledger.parent.mkdir(parents=True,exist_ok=True)
 prior=[json.loads(x) for x in a.attempt_ledger.read_text(encoding='utf-8').splitlines() if x.strip()] if a.attempt_ledger.exists() else []
 past_count=sum(1 for x in prior if 'attempt' in x)
 if past_count+len(manifests)>MAX_CALLS:raise ValueError('Cumulative AIM attempt budget would exceed 130')
 with a.ledger.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['cid','state','status','failure_code','reason']);w.writeheader();w.writerows(ledger)
 # One verified local calculator serves every uncached manifest. No retry occurs in this loop.
 calc=None
 if manifests:
  if not WEIGHT.is_file() or WEIGHT.stat().st_size!=8836941 or sha(WEIGHT)!=WEIGHT_SHA:raise ValueError('Pinned local weight integrity failed')
  Calc=aim.AIMFeaturesCalculator.calculator_class()
  calc=Calc(model=str(WEIGHT),device=a.device,compile_model=False,deterministic=True);calc.model.eval()
  for prm in calc.model.parameters():prm.requires_grad_(False)
 failures=[]
 used=0
 for cid,st,mp in manifests:
  used+=1
  attempt={'attempt':past_count+used,'cid':cid,'state':st,'input_sha256':sha(mp),'weight_sha256':WEIGHT_SHA,'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running'}
  with a.attempt_ledger.open('a',encoding='utf-8') as f:f.write(json.dumps(attempt)+'\n');f.flush();os.fsync(f.fileno())
  try:
   aim.extract(mp,a.output,a.device,True,calc=calc)
   cached=validate_output(cid,st,mp,a.output,mark=True)
   attempt.update(status='passed',feature_sha256=cached['feature_sha256'],receipt_sha256=cached['receipt_sha256'])
  except Exception as e:
   attempt.update(status='failed',failure=f'{type(e).__name__}: {e}');failures.append(attempt)
  with a.attempt_ledger.open('a',encoding='utf-8') as f:f.write(json.dumps({'attempt_result':attempt})+'\n');f.flush();os.fsync(f.fileno())
  if failures:break
 # Rebuild state ledger from verified cache and preserve all held/missing/failures.
 final=[]
 for cid in cids:
  if cid not in supported:
   for st in ('bound','free'):final.append({'cid':cid,'state':st,'status':'HELD','failure_code':'STEREOCHEMISTRY_HOLD','reason':'Source stereo conflict'})
   continue
  for st in ('bound','free'):
   mp=a.manifest_dir/f'{cid}_{st}.json'
   try:q=validate_output(cid,st,mp,a.output)
   except Exception as e:q=None;reason=str(e)
   else:reason=''
   status='PASS' if q else 'FAILED' if failures and failures[0]['cid']==cid and failures[0]['state']==st else 'PENDING'
   final.append({'cid':cid,'state':st,'status':status,'failure_code':'AIM_FORWARD_FAILED' if status=='FAILED' else '','reason':(failures[0]['failure'] if status=='FAILED' else reason)})
 with a.ledger.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=['cid','state','status','failure_code','reason']);w.writeheader();w.writerows(final)
 status='failed' if failures else 'passed'
 (a.output/'batch_run_receipt.json').write_text(json.dumps({'status':status,'model':MODEL,'source_commit':COMMIT,'weight_sha256':WEIGHT_SHA,'smoke_receipt_sha256':sha(a.smoke_receipt),'attempt_ledger':str(a.attempt_ledger),'attempt_count_total':past_count+used,'new_calls':used,'cap':MAX_CALLS,'failures':failures,'cached_successes':cache,'state_ledger':str(a.ledger)},indent=2)+'\n',encoding='utf-8')
 if failures:raise RuntimeError(f'First AIM feature call failed: {failures[0]}')
 print(json.dumps({'status':'passed','new_calls':used,'attempts_including_prior':past_count+used,'cap':MAX_CALLS},indent=2))
if __name__=='__main__':main()

