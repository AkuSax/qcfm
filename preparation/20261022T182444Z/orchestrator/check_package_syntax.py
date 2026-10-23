"""Parse prepared code and JSON. Do not import models or run training."""
import ast
import json
import os
from pathlib import Path

root = Path(__file__).resolve().parent.parent
checks = []
for folder, dirs, files in os.walk(root):
    dirs[:] = [d for d in dirs if d not in {'venv','source','.git','__pycache__','weights'}]
    for name in files:
        path = Path(folder)/name
        if path.suffix not in {'.py','.json'}:
            continue
        try:
            data = path.read_text(encoding='utf-8-sig')
            ast.parse(data, filename=str(path)) if path.suffix == '.py' else json.loads(data)
            checks.append({'path':str(path.relative_to(root)),'status':'PASS'})
        except Exception as exc:
            checks.append({'path':str(path.relative_to(root)),'status':'BLOCKED','error':str(exc)})
result = {'status':'PASS' if all(x['status']=='PASS' for x in checks) else 'BLOCKED',
          'files':len(checks),'checks':checks,'pretrained_model_calls':0,'fits':0}
(root/'orchestrator/package_syntax.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','files']}))
raise SystemExit(0 if result['status']=='PASS' else 1)
