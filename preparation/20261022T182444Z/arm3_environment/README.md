# AIMNet2 extraction environment

The isolated environment uses Python 3.11.1 and PyTorch 2.10.0+cpu.
It lives in `venv/` under this folder.
The environment does not change a global Python installation.

The source checkout uses commit `d14692839f4e5544f4eb38c15cf01298c27960f8`.
The selected model is `aimnet2-wb97m-d3_0`.
Its weight file has 8,836,941 bytes.
Its SHA-256 is `f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28`.
The staged weight is in `weights/aimnet2_wb97m_d3_0.pt`.
No code has loaded or run the weight.

`requirements.in` pins direct dependencies.
`requirements.lock` pins the direct and transitive packages for setup.
Setup reads `requirements.lock` and does not replace it.
`observed_freeze.txt` records the installed package versions after setup.
`setup.ps1` creates the local environment and installs the locked packages.
It stores pip caches and temporary files in this folder.
`checkout_source.ps1` retrieves the pinned source when it is absent.
`stage_weight.py` downloads the pinned weight only when its file is absent.
It verifies the file size and SHA-256 before it saves the file.
It rejects a mismatched existing file.

The JSON input manifest needs these fields:

- `id`: molecule identifier.
- `formal_charge`: integer net charge.
- `atomic_numbers`: all atom rows, including added hydrogen atoms.
- `coord_angstrom`: coordinates for all atom rows.
- `heavy_atom_indices`: input row indices for every atom with atomic number above 1.
- `source_atom_index`: source graph indices in the same order as `heavy_atom_indices`.

`extract_aim_features.py --self-test` runs synthetic schema and mapping checks.
`import_check.py` checks imports and the `aim` feature registration.
`import_receipt.json` and `synthetic_test_receipt.json` record these checks.

The future 3-ligand smoke command is:

```powershell
.\run_three_ligand_smoke.ps1 -Execute `
  -Ligand1 .\inputs\ligand1.json `
  -Ligand2 .\inputs\ligand2.json `
  -Ligand3 .\inputs\ligand3.json `
  -Output .\smoke_output -Device cpu
```

The command requires `-Execute` before it can load the model.
It stops after the first failed check.
It has a limit of 18 molecule evaluations in 14 calculator calls.
It checks energy parity, atom features, padded rows, batch output, rotations, permutations, and forces.
It writes a global receipt before it builds either calculator.
It marks incomplete receipts as failed when a check raises an error.

The local CPU runtime passed synthetic handling and import checks.
The pretrained model smoke remains untested.
The first dependency install hit a transient Windows file lock.
The recorded retry succeeded.
See `install_transfer_accounting.json` for the download-byte limit status.
