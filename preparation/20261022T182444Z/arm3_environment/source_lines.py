from pathlib import Path
for name in ['source/aimnet/calculators/calculator.py','source/aimnet/models/base.py']:
 s=Path(name).read_text().splitlines()
 for i,line in enumerate(s):
  if any(term in line for term in ['def eval(', 'def _split_batch_dim', 'def prepare_input', 'def calc_masks', 'def validate_mode2_input']):
   print('\n',name,'line',i+1)
   print('\n'.join(f'{j+1}: {s[j]}' for j in range(i,min(len(s),i+38))))
