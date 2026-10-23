import importlib.metadata as md
import json
import sys
from pathlib import Path

import numpy
import rdkit
import torch
import aimnet
from extract_aim_features import AIMFeaturesCalculator

Calc = AIMFeaturesCalculator.calculator_class()
assert "aim" in Calc.keys_out
assert "aim" in Calc.atom_feature_keys
assert torch.cuda.is_available() is False
receipt = {
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "rdkit": rdkit.__version__,
    "aimnet": md.version("aimnet"),
    "warp-lang": md.version("warp-lang"),
    "nvalchemi-toolkit-ops": md.version("nvalchemi-toolkit-ops"),
    "aim_export_key": "aim",
    "aim_in_keys_out": True,
    "aim_in_atom_feature_keys": True,
    "cuda_available": torch.cuda.is_available(),
    "model_initialized": False,
    "weights_loaded": False,
}
out = Path(__file__).resolve().parent / "import_receipt.json"
out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
print(json.dumps(receipt, indent=2))
