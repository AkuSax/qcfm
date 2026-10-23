from pathlib import Path
import runpy
import sys
import torch

package = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(package / "arm4_protocol"))

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
print({"torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads()})
runpy.run_path(str(Path(__file__).with_name("guarded_fit_entrypoint.py")), run_name="__main__")
