"""Prepare fixed-identity ligand geometries. Do not call a neural model."""
import hashlib
import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

RDKIT_VERSION = '2026.03.6'
SEED = 1729
MAX_STEPS = 200
N_CONFORMERS = 8

def source_graph(smiles):
    if rdBase.rdkitVersion != RDKIT_VERSION:
        raise ValueError('Use RDKit 2026.03.6.')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('The source graph does not parse.')
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError('Hold disconnected components for a scope decision.')
    if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
        raise ValueError('Hold radicals for a scope decision.')
    if any(s.specified == Chem.StereoSpecified.Unspecified for s in Chem.FindPotentialStereo(mol)):
        raise ValueError('Resolve unspecified stereochemistry first.')
    return mol

def graph_receipt(mol):
    text = Chem.MolToSmiles(mol, isomericSmiles=True)
    return {'isomeric_smiles': text, 'graph_sha256': hashlib.sha256(text.encode()).hexdigest(),
            'formal_charge': Chem.GetFormalCharge(mol), 'multiplicity': 1,
            'rdkit': rdBase.rdkitVersion, 'seed': SEED, 'max_steps': MAX_STEPS}

def bound_geometry(smiles, source_atom_indices, heavy_coordinates):
    mol = source_graph(smiles)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    indices = list(map(int, source_atom_indices))
    xyz = np.asarray(heavy_coordinates, dtype=np.float64)
    if len(indices) != len(set(indices)) or set(indices) != set(heavy):
        raise ValueError('Require an explicit heavy-atom bijection.')
    if xyz.shape != (len(heavy), 3) or not np.isfinite(xyz).all():
        raise ValueError('Require finite scored heavy-atom coordinates.')
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for index, position in zip(indices, xyz):
        conformer.SetAtomPosition(index, position.tolist())
    mol.AddConformer(conformer)
    hydrogenated = Chem.AddHs(mol, addCoords=True)
    props = AllChem.MMFFGetMoleculeProperties(hydrogenated, mmffVariant='MMFF94s')
    if props is None:
        raise ValueError('MMFF94s parameters are missing.')
    force_field = AllChem.MMFFGetMoleculeForceField(hydrogenated, props)
    for index in heavy:
        force_field.AddFixedPoint(index)
    status = force_field.Minimize(maxIts=MAX_STEPS)
    after = np.asarray(hydrogenated.GetConformer().GetPositions())
    if not np.array_equal(after[indices], xyz):
        raise ValueError('Fixed heavy-atom coordinates changed.')
    receipt = graph_receipt(mol)
    receipt.update(status='PASS' if status == 0 else 'BLOCKED', mmff_status=int(status),
                   mmff_energy_kcal_mol=float(force_field.CalcEnergy()),
                   source_atom_indices=indices, heavy_atom_indices=heavy,
                   coordinates_angstrom=after, atomic_numbers=np.array([a.GetAtomicNum() for a in hydrogenated.GetAtoms()]))
    return receipt

def free_geometry(smiles):
    mol = source_graph(smiles)
    hydrogenated = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = SEED
    params.numThreads = 1
    params.enforceChirality = True
    ids = list(AllChem.EmbedMultipleConfs(hydrogenated, numConfs=N_CONFORMERS, params=params))
    if len(ids) != N_CONFORMERS:
        raise ValueError('Embedding did not produce 8 conformers.')
    props = AllChem.MMFFGetMoleculeProperties(hydrogenated, mmffVariant='MMFF94s')
    if props is None:
        raise ValueError('MMFF94s parameters are missing.')
    trials = []
    for cid in ids:
        ff = AllChem.MMFFGetMoleculeForceField(hydrogenated, props, confId=cid)
        status = int(ff.Minimize(maxIts=MAX_STEPS))
        trials.append({'conformer_index': int(cid), 'mmff_status': status, 'energy_kcal_mol': float(ff.CalcEnergy())})
    valid = [x for x in trials if x['mmff_status'] == 0 and np.isfinite(x['energy_kcal_mol'])]
    receipt = graph_receipt(mol)
    receipt.update(status='PASS' if valid else 'BLOCKED', trials=trials)
    if valid:
        best = min(valid, key=lambda x: (x['energy_kcal_mol'], x['conformer_index']))
        receipt.update(selected_conformer=best['conformer_index'], mmff_energy_kcal_mol=best['energy_kcal_mol'],
                       coordinates_angstrom=np.asarray(hydrogenated.GetConformer(best['conformer_index']).GetPositions()),
                       atomic_numbers=np.array([a.GetAtomicNum() for a in hydrogenated.GetAtoms()]),
                       heavy_atom_indices=[a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1])
    return receipt
