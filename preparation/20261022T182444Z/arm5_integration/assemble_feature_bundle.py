"""Validate future AIM outputs and assemble the arm4 bundle. No model calls."""
from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
DEFAULT_EXPORTS=PACKAGE/'arm1_atoms'/'safe_exports'
DEFAULT_PROTOCOL=PACKAGE/'arm4_protocol'/'comparison_config.json'
AIM_MEMBER='aimnet2-wb97m-d3_0'
AIM_COMMIT='d14692839f4e5544f4eb38c15cf01298c27960f8'
WEIGHT_SHA='f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28'
SIMPLE_NAMES=['mmff94s_bound_minus_free_kcal_mol','formal_charge','heavy_atom_count','molecular_weight','clogp','tpsa','hbd','hba','rotatable_bonds','ring_count','count_C','count_N','count_O','count_S','count_F','count_Cl','count_Br','count_H']
AIM_NAMES=['aim_energy_bound_minus_free_eV','bound_force_rms_eV_per_angstrom','bound_force_max_eV_per_angstrom','bound_force_median_eV_per_angstrom','bound_force_p90_eV_per_angstrom','free_force_rms_eV_per_angstrom','free_force_max_eV_per_angstrom','free_force_median_eV_per_angstrom','free_force_p90_eV_per_angstrom','bound_charge_mean_all_atoms','bound_charge_std_all_atoms','bound_charge_abs_sum_all_atoms','bound_charge_min_all_atoms','bound_charge_max_all_atoms','free_charge_mean_all_atoms','free_charge_std_all_atoms','free_charge_abs_sum_all_atoms','free_charge_min_all_atoms','free_charge_max_all_atoms']
AIM_FREE_NAMES=AIM_NAMES

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def read_records(path):
    with Path(path).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))

def force_summary(force, heavy_rows):
    a=np.asarray(force,dtype=np.float64)
    if a.ndim!=2 or a.shape[1]!=3 or not np.isfinite(a).all(): raise ValueError('Forces must be finite N by 3')
    a=a[np.asarray(heavy_rows,dtype=np.int64)]
    mag=np.linalg.norm(a,axis=1)
    return [float(np.sqrt(np.mean(mag**2))),float(np.max(mag)),float(np.median(mag)),float(np.quantile(mag,.9))]

def read_aim(path, expected_cid, state, source, atomic_numbers, expected_xyz, expected_charge, manifest_dir, aim_root):
    if not path.is_file(): raise FileNotFoundError(f'{state} AIM feature export missing')
    receipt_path=path.parent/'receipt.json'
    if not receipt_path.is_file(): raise FileNotFoundError(f'{state} AIM receipt missing')
    receipt=json.loads(receipt_path.read_text(encoding='utf-8'))
    if receipt.get('status')!='passed' or receipt.get('features_npz_sha256')!=digest(path): raise ValueError(f'{state} AIM output receipt/hash is not a verified pass')
    manifest_path=Path(manifest_dir)/f'{expected_cid}_{state}.json'
    if not manifest_path.is_file(): raise FileNotFoundError(f'{state} prepared manifest missing')
    manifest_raw=manifest_path.read_bytes(); manifest=json.loads(manifest_raw)
    for k,v in {'id':f'{expected_cid}_{state}','formal_charge':expected_charge,'state':state,'parent_cid':expected_cid}.items():
        if manifest.get(k)!=v: raise ValueError(f'{state} manifest {k} mismatch')
    prep_path=Path(manifest_dir)/f'{expected_cid}_preparation_receipt.json'
    if prep_path.is_file():
        prep_record=json.loads(prep_path.read_text(encoding='utf-8'))
        if prep_record.get(state,{}).get('manifest_sha256') != hashlib.sha256(manifest_raw).hexdigest(): raise ValueError(f'{state} manifest hash differs from preparation receipt')
    checks={'model':AIM_MEMBER,'source_commit':AIM_COMMIT,'weight_sha256':WEIGHT_SHA,'weight_bytes':8836941,
            'input_sha256':hashlib.sha256(manifest_raw).hexdigest(),'feature_channels':256}
    for k,v in checks.items():
        if receipt.get(k)!=v: raise ValueError(f'{state} AIM receipt {k} mismatch')
    with np.load(path,allow_pickle=False) as q:
        feat=np.asarray(q['features'],dtype=np.float32); smap=np.asarray(q['source_atom_index'],dtype=np.int64)
        hrows=np.asarray(q['heavy_atom_indices'],dtype=np.int64); fullz=np.asarray(q['atomic_numbers'],dtype=np.int64)
        xyz=np.asarray(q['coord_angstrom'],dtype=np.float64); energy=np.asarray(q['energy_native'],dtype=np.float64).reshape(-1)
        charges=np.asarray(q['charges'],dtype=np.float64).reshape(-1)
        forces=np.asarray(q['forces_native'],dtype=np.float64) if 'forces_native' in q.files else None
    n=len(source)
    if feat.shape!=(n,256) or smap.shape!=(n,) or not np.array_equal(smap,source): raise ValueError(f'{state} AIM heavy mapping mismatch')
    if hrows.shape!=(n,) or len(set(hrows.tolist()))!=n or np.any(hrows<0) or np.any(hrows>=len(fullz)): raise ValueError(f'{state} heavy row indices invalid')
    if not np.array_equal(fullz[hrows],atomic_numbers): raise ValueError(f'{state} AIM element order mismatch')
    if not np.array_equal(fullz,np.asarray(manifest['atomic_numbers'],dtype=np.int64)) or not np.array_equal(xyz,np.asarray(manifest['coord_angstrom'],dtype=np.float64)) or not np.array_equal(hrows,np.asarray(manifest['heavy_atom_indices'],dtype=np.int64)) or not np.array_equal(smap,np.asarray(manifest['source_atom_index'],dtype=np.int64)): raise ValueError(f'{state} AIM full atom rows differ from prepared manifest')
    if xyz.shape!=(len(fullz),3) or not np.isfinite(xyz).all() or not np.isfinite(feat).all(): raise ValueError(f'{state} AIM coords/features invalid')
    if state=='bound' and not np.array_equal(xyz[hrows],expected_xyz): raise ValueError('AIM bound heavy coordinates differ from scored x_pred')
    if len(energy)!=1 or not np.isfinite(energy).all() or len(charges)!=len(fullz) or not np.isfinite(charges).all(): raise ValueError(f'{state} AIM energy or charges invalid')
    if forces is None: raise ValueError(f'{state} AIM forces are required for frozen scalar descriptors')
    if forces.shape != xyz.shape: raise ValueError(f'{state} force array shape differs from full atom array')
    fs=force_summary(forces,hrows)
    return feat,float(energy[0]),charges,fs

def assemble(records_csv, exact_inputs, aim_dir, output, ledger, manifest_dir, smoke_receipt, exports=DEFAULT_EXPORTS, protocol=DEFAULT_PROTOCOL):
    smoke=json.loads(Path(smoke_receipt).read_text(encoding='utf-8'))
    if smoke.get('status')!='passed' or smoke.get('weight_sha256')!=WEIGHT_SHA or smoke.get('molecule_evaluations',99)>18 or smoke.get('api_calls',99)>14 or smoke.get('metadata_parity')!='passed': raise ValueError('Three-ligand smoke receipt did not pass frozen limits')
    if smoke.get('input_ids') != ['57341_bound','68261_bound','570059_bound']: raise ValueError('Three-ligand smoke IDs differ from frozen subset')
    records=read_records(records_csv); chemistry=read_records(PACKAGE/'arm2_chemistry'/'per_record_coverage.csv'); chem_by={str(r['pubchem_cid']):r for r in chemistry}; cids=[str(r.get('pubchem_cid',r.get('CID',r.get('cid','')))) for r in records]
    if len(records)!=65 or len(set(cids))!=65 or not all(cids) or len(chem_by)!=65: raise ValueError('Need 65 unique frozen records and chemistry rows')
    if any(str(r['smiles'])!=str(chem_by[cid]['smiles']) for cid,r in zip(cids,records)): raise ValueError('Chemistry rows do not match frozen record order/SMILES')
    with np.load(exact_inputs,allow_pickle=False) as d: x1=np.asarray(d['x1'],np.float32); x2=np.asarray(d['x2'],np.float32)
    if x1.shape!=(65,384) or x2.shape!=(65,384): raise ValueError('Saved x1/x2 shape or row count changed')
    config=json.loads(Path(protocol).read_text(encoding='utf-8'))
    keys=config['feature_manifest_required_before_fit']['keys']
    sidecars={p.stem:p for p in Path(exports).glob('*.json')}
    counts=[]
    for cid,sp in sidecars.items():
        m=json.loads(sp.read_text(encoding='utf-8')); ep=sp.with_suffix('.npz')
        if digest(ep)!=m['npz_sha256']: raise ValueError(f'Boltz export hash mismatch: {cid}')
        counts.append(int(m['shapes']['source_atom_indices'][0]))
    max_atoms=max(counts)
    boltz=np.full((65,max_atoms,128),np.nan,np.float32); aim=np.full((65,max_atoms,256),np.nan,np.float32); free=np.full_like(aim,np.nan)
    mask=np.zeros((65,max_atoms),bool); allowed=np.zeros((65,max_atoms,max_atoms),bool)
    simple=np.full((65,len(SIMPLE_NAMES)),np.nan,np.float32); a_scalar=np.full((65,len(AIM_NAMES)),np.nan,np.float32); a_free=np.full_like(a_scalar,np.nan)
    valid=np.zeros(65,bool); failures=[]
    for i,(cid,row) in enumerate(zip(cids,records)):
        try:
            if cid not in sidecars: raise ValueError('No verified Boltz atom export')
            side=json.loads(sidecars[cid].read_text(encoding='utf-8')); ep=sidecars[cid].with_suffix('.npz')
            with np.load(ep,allow_pickle=False) as e:
                source=np.asarray(e['source_atom_indices'],np.int64); z=np.asarray(e['atomic_numbers'],np.int64); n=len(source)
                if n>max_atoms: raise ValueError('Atom count exceeds padded width')
                boltz[i,:n]=np.asarray(e['ligand_to_receptor_z_mean'],np.float32); mask[i,:n]=True
                allowed[i,:n,:n]=np.asarray(e['graph_adjacency'],bool); np.fill_diagonal(allowed[i,:n,:n],True)
                xyz=np.asarray(e['x_pred_heavy_coords'],np.float64)
            if digest(ep)!=side['npz_sha256']: raise ValueError('Boltz export hash changed during assembly')
            q=int(chem_by[cid]['total_formal_charge'])
            prep=Path(manifest_dir)/f'{cid}_preparation_receipt.json'
            if not prep.is_file(): raise FileNotFoundError('Preparation receipt missing')
            pr=json.loads(prep.read_text(encoding='utf-8'))
            if pr.get('status')!='PASS' or pr.get('free_conformer_pending') or not pr.get('free'): raise ValueError('Bound/free MMFF preparation receipt is incomplete')
            if pr.get('formal_charge')!=q or pr.get('simple_scalar_names')!=SIMPLE_NAMES: raise ValueError('Preparation charge or scalar schema mismatch')
            simple_vals=np.asarray(pr['simple_scalars'],np.float64)
            if simple_vals.shape!=(len(SIMPLE_NAMES),) or not np.isfinite(simple_vals).all(): raise ValueError('Simple scalar values invalid')
            bound=Path(aim_dir)/f'{cid}_bound'/'features.npz'; freep=Path(aim_dir)/f'{cid}_free'/'features.npz'
            b=read_aim(bound,cid,'bound',source,z,xyz,q,manifest_dir,aim_dir)
            f=read_aim(freep,cid,'free',source,z,None,q,manifest_dir,aim_dir)
            aim[i,:n]=b[0]; free[i,:n]=f[0]; simple[i]=simple_vals
            bs=np.asarray(b[2]); fs=np.asarray(f[2]); bstat=b[3]; fstat=f[3]
            a_scalar[i]=[b[1]-f[1],*bstat,*fstat,float(bs.mean()),float(bs.std()),float(np.abs(bs).sum()),float(bs.min()),float(bs.max()),float(fs.mean()),float(fs.std()),float(np.abs(fs).sum()),float(fs.min()),float(fs.max())]
            a_free[i]=a_scalar[i].copy()
            if not np.isfinite(a_scalar[i]).all() or not np.isfinite(a_free[i]).all(): raise ValueError('AIM scalar values invalid')
            valid[i]=True; failures.append({'cid':cid,'feature_valid':True,'failure_code':'','reason':''})
        except Exception as exc:
            failures.append({'cid':cid,'feature_valid':False,'failure_code':type(exc).__name__,'reason':str(exc)})
    output=Path(output); ledger=Path(ledger); output.parent.mkdir(parents=True,exist_ok=True); ledger.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,cids=np.asarray(cids,dtype='U32'),boltz_atoms=boltz,aim_atoms=aim,aim_atoms_free=free,global_boltz=np.concatenate([x1,x2],axis=1).astype(np.float32),atom_mask=mask,allowed=allowed,simple_scalars=simple,aim_scalars=a_scalar,aim_scalars_free=a_free,feature_valid=valid)
    with ledger.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['cid','feature_valid','failure_code','reason']); w.writeheader(); w.writerows(failures)
    schema={'schema':'qgcf.aim_features.v1','row_order':'records_csv','keys':{'cids':'65 unicode IDs','boltz_atoms':'(65,max_atoms,128)','aim_atoms':'bound AIM (65,max_atoms,256)','aim_atoms_free':'free AIM control (65,max_atoms,256)','global_boltz':'(65,768), x1 then x2','atom_mask':'(65,max_atoms)','allowed':'(65,max_atoms,max_atoms), adjacency plus self','simple_scalars':SIMPLE_NAMES,'aim_scalars':AIM_NAMES,'aim_scalars_free':AIM_FREE_NAMES,'feature_valid':'(65,) bool'},'scalar_units':{'MMFF94s':'kcal/mol','AIM energy gap':'eV','AIM force statistics':'eV/angstrom','AIM charges':'calculator native dimensionless charges over all prepared atoms including H'},'scalar_statistics':'Force norms use heavy atoms only. AIM energies use bound-minus-free gap; no absolute AIM energy enters primary scalars.','missing':'NaN numerical rows; false validity; all 65 failure ledger rows retained','input_hashes':{'records_sha256':digest(records_csv),'exact_inputs_sha256':digest(exact_inputs),'chemistry_sha256':digest(PACKAGE/'arm2_chemistry'/'per_record_coverage.csv')},'output_hashes':{'npz_sha256':digest(output),'ledger_sha256':digest(ledger)},'valid_rows':int(valid.sum()),'denominator':65}
    output.with_suffix('.schema.json').write_text(json.dumps(schema,indent=2)+'\n',encoding='utf-8')
    return schema

def main():
    p=argparse.ArgumentParser(); mode=p.add_mutually_exclusive_group(required=True); mode.add_argument('--dry-run',action='store_true'); mode.add_argument('--assemble',action='store_true')
    p.add_argument('--records',type=Path,required=True); p.add_argument('--exact-inputs',type=Path,required=True); p.add_argument('--aim-dir',type=Path,required=True); p.add_argument('--manifest-dir',type=Path,required=True); p.add_argument('--smoke-receipt',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--ledger',type=Path,required=True); 
    a=p.parse_args()
    if not a.assemble:
        for x in (a.records,a.exact_inputs,a.manifest_dir):
            if not x.exists(): raise FileNotFoundError(x)
        print(json.dumps({'status':'PASS','dry_run_only':True,'records':65,'required_arrays':['cids','boltz_atoms','aim_atoms','aim_atoms_free','global_boltz','atom_mask','allowed','simple_scalars','aim_scalars','aim_scalars_free','feature_valid'],'aim_requires_receipt_hashes_maps_coords_forces':True,'model_calls':0,'future_output':str(a.output),'future_ledger':str(a.ledger)},indent=2)); return
    result=assemble(a.records,a.exact_inputs,a.aim_dir,a.output,a.ledger,a.manifest_dir,a.smoke_receipt)
    print(json.dumps({'status':'PASS' if result['valid_rows']>=59 else 'BLOCKED','valid_rows':result['valid_rows'],'denominator':65,'npz':str(a.output),'ledger':str(a.ledger),'schema':str(a.output.with_suffix('.schema.json'))},indent=2))
    if result['valid_rows'] < 59:
        raise SystemExit(1)

if __name__=='__main__': main()
