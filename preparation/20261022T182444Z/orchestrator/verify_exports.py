"""Check candidate exports against primary blobs without model calls."""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from rdkit import Chem

HERE = Path(__file__).resolve().parent
PACK = HERE.parent
BUNDLE = Path('C:/Users/Saxen/ppi_IG/results/qhts_feasibility/20261012/independent_data_arm/capture_capacity_retries_20261012/request_0002/collected/bundle')
DTYPES = {'torch.float32':'<f4','torch.float64':'<f8','torch.int64':'<i8','torch.int32':'<i4','torch.int16':'<i2','torch.int8':'i1','torch.uint8':'u1','torch.bool':'?'}

def fields(node):
    return {k['value']:v for k,v in node['items']}

def tensor(node):
    assert node['metadata']['kind'] == 'torch'
    payload = gzip.decompress((BUNDLE / 'captures' / node['file']).read_bytes())
    return np.frombuffer(payload, dtype=DTYPES[node['metadata']['dtype']]).reshape(node['metadata']['shape'])

def main():
    reports = []
    failed = []
    for file in sorted((PACK/'arm1_atoms/safe_exports').glob('*.npz')):
        cid = file.stem
        saved = json.loads(file.with_suffix('.json').read_text())
        capture = json.loads((BUNDLE/'captures/compounds'/f'{cid}.json').read_text())
        kwargs = fields(capture['calls'][0]['kwargs'])
        feats = fields(kwargs['feats'])
        source = Chem.RemoveHs(Chem.MolFromSmiles(capture['source_record']['smiles']))
        n = source.GetNumAtoms()
        with np.load(file, allow_pickle=False) as archive:
            a = {key:archive[key] for key in archive.files}
        xyz = tensor(kwargs['x_pred'])[0,0,a['feature_atom_rows'],:]
        z = tensor(kwargs['z'])
        receptor = np.flatnonzero(tensor(feats['token_pad_mask'])[0].astype(bool) & (tensor(feats['mol_type'])[0] == 0))
        expected_context = z[0,a['token_positions'][:,None],receptor[None,:],:].mean(axis=1).astype(np.float32)
        adj = np.zeros((n,n), dtype=np.float32)
        for bond in source.GetBonds():
            i,j = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            adj[i,j] = adj[j,i] = bond.GetBondTypeAsDouble()
        checks = {'npz_sha256': hashlib.sha256(file.read_bytes()).hexdigest() == saved['npz_sha256'],
                  'source_atom_bijection': np.array_equal(a['source_atom_indices'],np.arange(n)),
                  'unique_atom_rows': len(set(a['feature_atom_rows'].tolist())) == n,
                  'unique_token_positions': len(set(a['token_positions'].tolist())) == n,
                  'exact_scored_coordinates': np.array_equal(xyz,a['x_pred_heavy_coords']),
                  'exact_receptor_context': np.array_equal(expected_context,a['ligand_to_receptor_z_mean']),
                  'finite_context': bool(np.isfinite(expected_context).all()),
                  'source_elements': np.array_equal(a['atomic_numbers'],[x.GetAtomicNum() for x in source.GetAtoms()]),
                  'source_bonds': np.array_equal(a['graph_adjacency'],adj)}
        specified = {int(i):label for i,label in Chem.FindMolChiralCenters(source,includeUnassigned=False)}
        actual_stereo = {}
        if specified:
            posed = Chem.Mol(source)
            Chem.RemoveStereochemistry(posed)
            conf = Chem.Conformer(n)
            for i,point in enumerate(xyz):
                conf.SetAtomPosition(i,point.astype(float).tolist())
            posed.AddConformer(conf)
            Chem.AssignStereochemistryFrom3D(posed, replaceExistingTags=True)
            actual_stereo = {int(i):label for i,label in Chem.FindMolChiralCenters(posed,includeUnassigned=True)}
            checks['source_specified_atom_stereo_in_scored_geometry'] = all(actual_stereo.get(i) == label for i,label in specified.items())
        lengths = [float(np.linalg.norm(xyz[b.GetBeginAtomIdx()]-xyz[b.GetEndAtomIdx()])) for b in source.GetBonds()]
        report = {'cid':cid,'checks':checks,'source_specified_atom_stereo':specified,'scored_atom_stereo':actual_stereo,
                  'bond_length_min_angstrom':min(lengths),'bond_length_max_angstrom':max(lengths)}
        reports.append(report)
        if not all(checks.values()):
            failed.append(cid)
    result = {'status':'PASS' if len(reports)==62 and not failed else 'BLOCKED', 'candidate_exports':len(reports),
              'failed_cids':failed,'records':reports,'model_calls':0,
              'geometry_limit':'Bond lengths are descriptive. No new geometry acceptance threshold applies.'}
    (HERE/'export_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['status','candidate_exports','failed_cids']}))
    return 0 if result['status']=='PASS' else 1

if __name__=='__main__':
    raise SystemExit(main())
