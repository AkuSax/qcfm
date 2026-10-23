import json
import subprocess
from pathlib import Path
root = Path(__file__).resolve().parent
for name in ['dependency_receipt.json','weight_stage_receipt.json','import_receipt.json','synthetic_test_receipt.json','setup_failure.json']:
 p=root/name
 print(name, json.loads(p.read_text(encoding='utf-8')))
print('lock_lines',len((root/'requirements.lock').read_text(encoding='utf-8').splitlines()))
print('source_commit',subprocess.check_output(['git','-C',str(root/'source'),'rev-parse','HEAD'],text=True).strip())
print('source_clean',not subprocess.check_output(['git','-C',str(root/'source'),'status','--porcelain'],text=True).strip())
print('weight_bytes',(root/'weights'/'aimnet2_wb97m_d3_0.pt').stat().st_size)
