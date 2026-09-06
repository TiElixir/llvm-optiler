# AI Compiler Optimization Explorer

[![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-torch--mlir-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/UI-React-61DAFB?logo=react&logoColor=111827)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/Frontend-TypeScript-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![MLIR](https://img.shields.io/badge/IR-MLIR-555555?logo=llvm&logoColor=white)](https://mlir.llvm.org/)
[![Status](https://img.shields.io/badge/status-active%20development-orange)](PLAN.md)

**AI Compiler Optimization Explorer** is an interactive PyTorch compiler visualization tool for inspecting how machine-learning programs lower through **Torch-MLIR**, **MLIR**, and **Linalg** intermediate representations.

Paste or select a PyTorch model, compile it through the local Torch-MLIR pipeline, inspect optimization snapshots, and explore tensor dataflow in a React Flow graph. The project is designed for compiler engineers, ML systems researchers, students, and developers learning how PyTorch programs become lower-level IR.

> Explore PyTorch compilation, MLIR lowering, Linalg operations, compiler pass traces, and tensor dataflow in one browser-based developer tool.

## Highlights

- **PyTorch-to-MLIR exploration** — compile small PyTorch models to Linalg-on-tensors IR.
- **Optimization timeline** — browse compiler pass snapshots and generated MLIR files.
- **Interactive dataflow graph** — inspect operations, operands, block arguments, yields, and returns.
- **PyTorch Names mode** — map SSA values and region arguments back to readable Python names.
- **Trace and selection tools** — follow downstream dataflow and focus on related graph nodes.
- **Template library** — keep reusable `.py` models in `templates/` and load them from the sidebar.
- **Dark mode and fullscreen graph view** — use the explorer comfortably during debugging or teaching.
- **FastAPI backend** — expose compilation, run artifacts, MLIR snapshots, and templates through a small local API.

## Sample
### Code
```py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x, w, b):
        y = torch.matmul(x, w)
        return torch.relu(y + b)

# You must define 'model' and 'inputs'
model = MyModel()
inputs = (torch.randn(4, 4), torch.randn(4, 4), torch.randn(4))
```
### Visual

<img width="1252" height="792" alt="image" src="https://github.com/user-attachments/assets/73e44786-d0e2-477e-a63c-fedd2f3224e2" />


## Architecture

```text
PyTorch model
     |
     v
Torch-MLIR compilation
     |
     v
MLIR / Linalg-on-tensors snapshots
     |
     +--> FastAPI artifact and template API
     |
     v
React + TypeScript + React Flow explorer
```

The backend compiles the submitted model, captures pass output, stores run artifacts under `runs/`, and serves metadata and snapshots. The frontend loads the timeline, displays MLIR source, and renders a navigable graph of the parsed IR.

## Requirements

- Python **3.11 or 3.12** recommended for the pinned Torch and Torch-MLIR packages.
- Node.js and npm.
- A CPU-compatible PyTorch/Torch-MLIR environment.
- Linux, macOS, or Windows with a supported Python toolchain.

The pinned dependencies currently target an older Torch-MLIR nightly build. Package availability may vary by Python version and operating system.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/TiElixir/xComp.git
cd xComp
```

### 2. Create a Python environment

Using `uv`:

```bash
uv venv .venv --python 3.11
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install backend and compiler dependencies:

```bash
python -m pip install -r requirements.txt
```

### 3. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## Run Locally

Start the FastAPI backend from the repository root:

```bash
.venv/bin/uvicorn backend_api:app --host 0.0.0.0 --port 8001
```

In a second terminal, start the Vite development server:

```bash
cd frontend
npm run dev -- --port 3000
```

Open [http://localhost:3000](http://localhost:3000) in your browser.

The frontend checks ports `8001` and `8000` for the backend API. If you use another backend port, update the frontend configuration or start the server on one of those ports.

## Using the Explorer

1. Open the PyTorch source editor.
2. Paste a model that defines both `model` and `inputs`.
3. Click **Compile**.
4. Select a generated optimization snapshot from the sidebar.
5. Switch between **PyTorch Source** and **MLIR Trace**.
6. Enable **Show PyTorch Names** to replace SSA labels with source-level names where mappings are available.
7. Click graph nodes to focus related operations or use the context menu's **Trace** action to follow dataflow.

A minimal compilable model looks like this:

```python
import torch
import torch.nn as nn


class AddModel(nn.Module):
    def forward(self, x, bias):
        return x + bias


model = AddModel()
inputs = (torch.randn(4, 4), torch.randn(4, 4))
```

## PyTorch Templates

Place reusable PyTorch programs in the repository's `templates/` directory:

```text
templates/
├── matmult.py
├── sample2.py
└── your_model.py
```

Each `.py` file appears in the frontend sidebar under **PyTorch Templates**. Selecting a template loads it into the editor, where it can be compiled like pasted code.

Templates should define:

- A `model` object or callable.
- An `inputs` tuple or compatible input structure.

## Project Layout

```text
.
├── backend_api.py             # FastAPI service and compilation endpoint
├── experiment2.py             # Standalone Torch-MLIR experiment
├── minimal_experiment.py      # Minimal compilation experiment
├── parser.py                  # Trace parsing utilities
├── requirements.txt           # Python dependencies
├── templates/                 # Loadable PyTorch model templates
├── runs/                      # Generated compilation artifacts (ignored)
└── frontend/
    ├── src/                   # React graph explorer and MLIR parser
    ├── package.json           # Frontend scripts and dependencies
    └── vite.config.ts         # Vite configuration
```

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/api/compile` | Compile submitted PyTorch source and create a run |
| `GET` | `/api/runs` | List available compilation runs |
| `GET` | `/api/runs/{run_id}` | Load run events and PyTorch metadata |
| `GET` | `/api/runs/{run_id}/file` | Read a generated snapshot file |
| `GET` | `/api/templates` | List Python files in `templates/` |
| `GET` | `/api/templates/{name}` | Read one validated Python template |

FastAPI's interactive API documentation is available at:

- [http://localhost:8001/docs](http://localhost:8001/docs)
- [http://localhost:8001/redoc](http://localhost:8001/redoc)

## Development Commands

Frontend build:

```bash
cd frontend
npm run build
```

Frontend lint:

```bash
cd frontend
npm run lint
```

Frontend tests:

```bash
cd frontend
npx vitest run
```

Backend syntax check:

```bash
python -m py_compile backend_api.py
```

## Generated Files and Git Hygiene

Compilation output is written to `runs/` and includes MLIR snapshots, event logs, wrappers, and source metadata. These generated artifacts are ignored by Git. Python caches, frontend dependencies, build output, logs, and local environment files are also excluded through `.gitignore`.

## Troubleshooting

### Torch or Torch-MLIR cannot be installed

Use Python 3.11 or 3.12 and install the pinned requirements inside a fresh virtual environment. Torch-MLIR nightly wheels are version-sensitive and may not support the newest Python releases.

### The frontend shows no runs

Confirm that the backend is running on port `8001` or `8000`, then reload the frontend. Check the browser console and backend terminal for compilation or CORS errors.

### Compilation fails

Confirm that the submitted source defines `model` and `inputs`. Start with one of the files in `templates/`, then simplify the model until the Torch-MLIR operation is supported.

## Roadmap

- Broader Torch-MLIR dialect and pass support.
- Richer operation provenance across compilation stages.
- IR diffing between before/after pass snapshots.
- LLVM IR and native-code exploration.
- Reproducible run metadata and benchmark comparisons.
- Automated CI for frontend and backend validation.

## Contributing

Contributions are welcome. For focused changes:

1. Create a feature branch.
2. Make the smallest coherent change.
3. Run the relevant frontend and backend checks.
4. Update the documentation when behavior or setup changes.
5. Open a pull request with a clear description and validation notes.

## License

No license file is currently included. Add an explicit `LICENSE` file before distributing this project for reuse.

## Keywords

`PyTorch compiler` · `Torch-MLIR` · `MLIR` · `Linalg` · `LLVM` · `compiler optimization` · `intermediate representation` · `IR visualization` · `dataflow graph` · `machine learning systems` · `React Flow` · `FastAPI` · `PyTorch graph explorer`
