"""Check synthetic geometry preparation only."""
import json
from pathlib import Path
import numpy as np
from chemistry_prepare import source_graph, bound_geometry, free_geometry

def main():
    # These synthetic coordinates have no relation to retained ligand captures.
    xyz = np.array([[0.,0.,0.], [1.5,0.,0.], [2.,1.3,0.]])
    bound = bound_geometry('CCO', [0,1,2], xyz)
    assert np.array_equal(bound['coordinates_angstrom'][:3], xyz)
    assert bound['status'] == 'PASS'
    free = free_geometry('CCO')
    assert free['status'] == 'PASS' and len(free['trials']) == 8
    repeated = free_geometry('CCO')
    assert np.array_equal(free['coordinates_angstrom'], repeated['coordinates_angstrom'])
    for invalid in ['CC(O)C(=O)O', '[CH3]']:
        try:
            source_graph(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('The invalid synthetic scope passed.')
    try:
        bound_geometry('CCO', [0,0,2], xyz)
    except ValueError:
        pass
    else:
        raise AssertionError('A duplicate atom map passed.')
    report = {'status':'PASS', 'synthetic_only':True, 'checks': ['fixed heavy atoms', '8 free conformers',
              'repeat determinism', 'unspecified stereo rejection', 'radical rejection', 'duplicate map rejection'],
              'model_calls':0, 'retained_ligand_minimizations':0}
    Path(__file__).with_name('chemistry_synthetic_test.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))

if __name__ == '__main__':
    main()
