"""Preserve and correct preparation specification drift."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent.parent
path = root / 'arm2_chemistry/chemistry_decision.json'
data = json.loads(path.read_text())
before = data['preparation_specification'].copy()
spec = data['preparation_specification']
spec['free_conformer'] = 'Use ETKDGv3 with seed 1729. Generate 8 conformers. Minimize each with MMFF94s for at most 200 steps. Select the lowest converged energy. Break equal-energy ties by conformer index. Keep all identities fixed.'
spec['bound_hydrogens'] = 'Add hydrogens with RDKit AddHs and coordinates. Map exact captured heavy atoms explicitly. Fix every heavy atom. Minimize hydrogen positions with MMFF94s for at most 200 steps. Preserve heavy-atom coordinates exactly.'
spec['failure_handling'] = 'Admit only MMFF status 0 as converged. Record missing parameters, embedding failure, and nonconvergence. Do not substitute a method. Stop a dependent extraction when preparation fails.'
spec['determinism'] = 'Pin RDKit 2026.03.6, ETKDGv3, seed 1729, MMFF94s, and 200-step limits. Use conformer index to break energy ties. Save every setting and outcome.'
path.write_text(json.dumps(data, indent=2) + '\n')
receipt = {'status': 'PASS', 'cause': 'The arm specification used an undefined CID-derived seed. It omitted optimization limits.',
           'authority': 'FROZEN_EXPERIMENT_PROPOSAL_v1_20261022.json feature_boundary',
           'previous_specification': before, 'corrected_specification': spec,
           'other_correction': 'The arm first reported MMFF94s parameter count 0 from an untested field. It then checked MMFFGetMoleculeProperties with MMFF94s. All 65 pass parameter availability. Optimization remains untested.'}
(root / 'orchestrator/chemistry_correction_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print('PASS: Fixed seed and step limits. Preserved the previous specification.')
