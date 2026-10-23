"""Test the bundle contract with synthetic AIM outputs. Do not call a model."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('assembler', ROOT/'arm5_integration/assemble_feature_bundle.py')
assembler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assembler)
OUT = ROOT/'orchestrator/synthetic_assembler_test'/sys.argv[1]
OUT.mkdir(parents=True,exist_ok=False)
MAN = OUT/'synthetic_manifests'; AIM = OUT/'synthetic_AIM_outputs'
MAN.mkdir(); AIM.mkdir()
SOURCE = Path('C:/Users/Saxen/ppi_IG/results/representation_diagnosis/20261015/attempt02/gcp_a100_retry_01/collected')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')

def main():
    rng = np.random.default_rng(1729)
    ids = sorted(p.stem for p in (ROOT/'arm1_atoms/safe_exports').glob('*.npz'))
    for cid in ids:
        path = ROOT/'arm1_atoms/safe_exports'/f'{cid}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        with np.load(path,allow_pickle=False) as a:
            source = a['source_atom_indices'].copy()
            xyz = a['x_pred_heavy_coords'].astype(float)
        mol = Chem.MolFromSmiles(meta['source_smiles'])
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i,point in enumerate(xyz): conf.SetAtomPosition(i,point.tolist())
        mol.AddConformer(conf)
        hydrogenated = Chem.AddHs(mol,addCoords=True)
        fullz = np.array([a.GetAtomicNum() for a in hydrogenated.GetAtoms()],dtype=int)
        fullxyz = np.asarray(hydrogenated.GetConformer().GetPositions())
        heavy = np.flatnonzero(fullz>1)
        charge = Chem.GetFormalCharge(mol)
        prep = {'synthetic_only':True,'status':'PASS','formal_charge':charge,'free_conformer_pending':False,
                'simple_scalar_names':assembler.SIMPLE_NAMES,'simple_scalars':[0.0]*len(assembler.SIMPLE_NAMES)}
        for state in ['bound','free']:
            coords = fullxyz.copy() if state=='bound' else fullxyz+2.0
            manifest = {'synthetic_only':True,'id':f'{cid}_{state}','parent_cid':cid,'state':state,
                        'formal_charge':charge,'atomic_numbers':fullz.tolist(),'coord_angstrom':coords.tolist(),
                        'heavy_atom_indices':heavy.tolist(),'source_atom_index':source.tolist(),
                        'coordinate_unit':'angstrom','multiplicity':1}
            mp = MAN/f'{cid}_{state}.json'
            write(mp,manifest)
            prep[state] = {'synthetic_only':True,'status':'PASS','mmff_status':0,'mmff_energy_kcal_mol':0.0,
                           'manifest_sha256':sha(mp)}
            output = AIM/f'{cid}_{state}'/'features.npz'
            output.parent.mkdir()
            np.savez_compressed(output,features=rng.normal(size=(len(source),256)).astype('f4'),
                source_atom_index=source,heavy_atom_indices=heavy,atomic_numbers=fullz,coord_angstrom=coords,
                energy_native=np.array([float(state=='bound')]),charges=np.zeros(len(fullz)),
                forces_native=np.ones((len(fullz),3),dtype='f4'))
            receipt = {'synthetic_only':True,'status':'passed','model':assembler.AIM_MEMBER,
                'source_commit':assembler.AIM_COMMIT,'weight_sha256':assembler.WEIGHT_SHA,'weight_bytes':8836941,
                'input_sha256':sha(mp),'feature_channels':256,'features_npz_sha256':sha(output)}
            write(output.parent/'receipt.json',receipt)
        write(MAN/f'{cid}_preparation_receipt.json',prep)
    smoke = OUT/'synthetic_smoke_receipt.json'
    write(smoke,{'synthetic_only':True,'status':'passed','weight_sha256':assembler.WEIGHT_SHA,
        'molecule_evaluations':18,'api_calls':14,'metadata_parity':'passed',
        'input_ids':['57341_bound','68261_bound','570059_bound']})
    result = assembler.assemble(SOURCE/'records.csv',SOURCE/'exact_inputs.npz',AIM,OUT/'bundle.npz',
        OUT/'ledger.csv',MAN,smoke)
    assert result['valid_rows']==62, result
    with np.load(OUT/'bundle.npz',allow_pickle=False) as a:
        assert len(a['cids'])==65 and a['feature_valid'].sum()==62
        assert a['simple_scalars'].shape==(65,18) and a['aim_scalars'].shape==(65,19)
        assert np.isfinite(a['aim_scalars'][a['feature_valid']]).all()
        held = set(a['cids'][~a['feature_valid']].tolist())
        assert held=={'660560','3243170','6375556'}
    ledger = list(csv.DictReader((OUT/'ledger.csv').open()))
    assert len(ledger)==65
    # Reject an altered receipt without another assembly pass.
    first = ids[0]
    path = AIM/f'{first}_bound'/'features.npz'
    receipt_path = path.parent/'receipt.json'
    good = json.loads(receipt_path.read_text())
    write(OUT/'preserved_receipt_before_negative_test.json',good)
    with np.load(ROOT/'arm1_atoms/safe_exports'/f'{first}.npz',allow_pickle=False) as a:
        src,z,xyz = a['source_atom_indices'],a['atomic_numbers'],a['x_pred_heavy_coords']
    checks = []
    for field, value in [('status','failed'),('features_npz_sha256','0'*64)]:
        bad = dict(good); bad[field]=value
        write(receipt_path,bad)
        try:
            assembler.read_aim(path,first,'bound',src,z,xyz,0,MAN,AIM)
        except ValueError:
            checks.append(field)
            write(OUT/f'negative_{field}.json',bad)
        else:
            raise AssertionError(f'The assembler accepted an altered {field}.')
    write(receipt_path,good)
    with np.load(path,allow_pickle=False) as a: arrays={k:a[k].copy() for k in a.files}
    arrays['coord_angstrom'][0,0] += 1.0
    np.savez_compressed(path,**arrays)
    bad = dict(good); bad['features_npz_sha256']=sha(path); write(receipt_path,bad)
    try:
        assembler.read_aim(path,first,'bound',src,z,xyz,0,MAN,AIM)
    except ValueError:
        checks.append('coordinate_change')
    else:
        raise AssertionError('The assembler accepted altered coordinates.')
    report={'status':'PASS','synthetic_only':True,'valid_rows':62,'ledger_rows':65,
        'held_cids':sorted(held),'simple_scalars':18,'aim_scalars':19,'negative_checks':checks,
        'pretrained_model_calls':0,'fits':0,'MMFF_minimizations':0}
    write(OUT/'result.json',report)
    write(ROOT/'orchestrator/assembler_synthetic_test.json',report)
    print(json.dumps(report))

try:
    main()
except Exception as exc:
    write(OUT/'FAILED_test_receipt.json',{'status':'BLOCKED','synthetic_only':True,'error':str(exc),
                                        'pretrained_model_calls':0,'fits':0})
    raise
