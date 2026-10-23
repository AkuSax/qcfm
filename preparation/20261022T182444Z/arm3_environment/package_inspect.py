from pathlib import Path
import inspect
from aimnet.calculators.calculator import AIMNet2Calculator
print('Calculator',[(n,str(inspect.signature(getattr(AIMNet2Calculator,n)))) for n in ['__init__','calculate','forward','__call__','process_input','set_grad_tensors'] if hasattr(AIMNet2Calculator,n)])
from pathlib import Path
s=Path('source/aimnet/calculators/resolve.py').read_text()
i=s.index('def resolve_model(')
print(s[i:i+2300])
p=Path('source/aimnet/calculators/calculator.py')
s=p.read_text()
for q in ['class Calculator','def forward','def calculate','self.model(data)','def calculate(', 'def process_output']:
 i=s.find(q)
 if i>=0: print('\n---',q,'line',s.count('\n',0,i)+1,'---\n',s[i:i+1800])
p=Path('source/aimnet/models')
for f in p.rglob('*.py'):
 s=f.read_text(errors='replace')
 if 'aim' in s:
  for q in ['"aim"','aim =','aim:','return data']:
   i=s.find(q)
   if i>=0: print('\n---',f,q,'line',s.count('\n',0,i)+1,'---\n',s[max(0,i-250):i+550])
