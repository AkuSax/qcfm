import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
names = ["receipts.py", "extract_aim_features.py", "smoke_three.py", "stage_weight.py"]
rows = []
for name in names:
    ast.parse((root / name).read_text(encoding="utf-8"), filename=name)
    rows.append({"file": name, "parsed": True})
(root / "python_ast_parse_receipt.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
print("AST parse passed for 4 Python files")
