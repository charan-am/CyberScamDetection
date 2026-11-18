# Phase 0 — Environment and Project Skeleton: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a reproducible Python 3.12.16 environment, a pinned dependency set, an importable `scamdetector` package, and a pytest suite that exits 0 with genuine smoke tests — the foundation every later phase depends on.

**Architecture:** `src/` layout package installed editable into a uv-managed 3.12.16 virtual environment; dependency versions captured from the resolver, never typed by hand; a two-test environment contract proves the interpreter and package wiring before any ML work begins.

**Tech Stack:** uv 0.13.8, Python 3.12.16, pandas, numpy, scikit-learn, joblib, streamlit; pytest for verification.

**Spec:** `docs/superpowers/specs/2026-11-18-cyber-scam-detection-design.md` (Revision 3, Phase 0)

## Global Constraints

- Interpreter: **Python 3.12.16 only**. Python 3.14 is present on this machine and must not be used.
- `pip.exe` is blocked by an Application Control policy. Use `uv pip`; never bare `pip`. `uv venv` does not install `pip`, so `python -m pip` only works if a venv was created with `--seed`.
- Repository root: `C:\Users\Charan A M\CyberScamDetection`. Existing files and Git history must be preserved.
- No destructive operations. The only deletion permitted in this plan is `.venv-verify`, created by Task 2 for a clean-install check.
- Dependency versions must come from the resolver output, never from memory.
- Package version is `0.1.0` in exactly one authoritative place; tests detect drift.
- Working branch: `master`, repository currently has no commits. First commit establishes history.
- Commands below are Windows **CMD**. In this harness's PowerShell shell, use `.venv\Scripts\python.exe` directly instead of activation.

## Review Focus

- **Wrong interpreter silently active** — the user runs `python` and gets 3.14 with no error until something breaks inside scikit-learn. Expected: the suite fails immediately and loudly with the actual version named.
- **Imports depend on the current directory** — an editable install means `import scamdetector` works from any cwd; running from elsewhere must still succeed. Expected: import from a temporary directory passes.
- **Hand-typed dependency versions** — a version that does not exist for 3.12 breaks a fresh install. Expected: a clean venv installs from the committed manifests without edits.
- **Virtual environment or caches committed to Git** — repo bloat and a broken future clone. Expected: `git ls-files` contains no `.venv`, `__pycache__`, or `*.pyc`.
- **Duplicate version strings between `pyproject.toml` and `__init__.py`** — drift makes `0.1.0` a lie. Expected: a test compares both and fails on mismatch.

---

### Task 1: Virtual environment on Python 3.12.16

**Files:**
- Create: `.python-version`

**Interfaces:**
- Produces: `.venv\Scripts\python.exe` reporting exactly `3.12.16`

- [ ] **Step 1: Confirm uv is available and record its version**

Run:
```cmd
uv --version
```
Expected: `uv 0.13.8` or newer. If the command is not found, stop and report it; do not fall back to the system `pip`.

- [ ] **Step 2: Create the virtual environment from the 3.12.16 build**

Run:
```cmd
cd /d "C:\Users\Charan A M\CyberScamDetection"
uv venv --python 3.12.16 .venv
```
Expected: success message naming the interpreter path under `%APPDATA%\uv\python\cpython-3.12.16-windows-x86_64-none\`.

If uv cannot resolve `3.12.16` by number, run instead:
```cmd
uv venv --python "%APPDATA%\uv\python\cpython-3.12.16-windows-x86_64-none\python.exe" .venv
```

- [ ] **Step 3: Prove the interpreter version**

Run:
```cmd
.venv\Scripts\python -V
```
Expected: `Python 3.12.16`. If it prints 3.14.x, delete nothing; stop and report the mismatch.

- [ ] **Step 4: Record the interpreter contract in `.python-version`**

Create `.python-version` containing the single line:
```
3.12.16
```

- [ ] **Step 5: Commit**

```cmd
git add .python-version
git commit -m "chore: pin python 3.12.16 interpreter contract"
```

---

### Task 2: Pinned dependency manifests

**Files:**
- Create: `requirements.txt`
- Create: `requirements-dev.txt`

**Interfaces:**
- Consumes: the activated `.venv` from Task 1
- Produces: two manifests that recreate the environment exactly; direct versions recorded in `requirements.txt`

- [ ] **Step 1: Install the runtime dependencies unpinned**

Run:
```cmd
uv pip list
uv pip install pandas numpy scikit-learn joblib streamlit
```
Expected: the first command reports the venv is empty; the install selects 3.12-compatible wheels with no build-from-source warnings. Note: `uv venv` does **not** install `pip`, so `python -m pip` is unavailable unless the venv was created with `--seed`. `uv pip` is the supported path.

- [ ] **Step 2: Capture the resolved runtime environment**

Run:
```cmd
uv pip freeze > requirements.txt
```
Expected: one pinned `name==version` line per installed distribution (runtime only; pytest is not installed yet). Do not edit these versions by hand.

- [ ] **Step 3: Install and pin the test dependency**

Run:
```cmd
uv pip install pytest
uv pip freeze | findstr /i "^pytest==" > requirements-dev.txt
```
Expected: `requirements-dev.txt` contains a single `pytest==<resolved version>` line.

- [ ] **Step 4: Verify the pins install in a scratch environment**

Run:
```cmd
uv venv --python 3.12.16 .venv-verify
uv pip install --python .venv-verify\Scripts\python.exe -r requirements.txt -r requirements-dev.txt
.venv-verify\Scripts\python -c "import pandas, numpy, sklearn, joblib, streamlit, pytest; print('ok')"
```
Expected: `ok`. Then remove only this scratch environment:
```cmd
rmdir /s /q .venv-verify
```

- [ ] **Step 5: Commit**

```cmd
git add requirements.txt requirements-dev.txt
git commit -m "chore: pin resolved runtime and test dependencies"
```

---

### Task 3: Package skeleton and genuine smoke tests

**Files:**
- Create: `pyproject.toml`
- Create: `src/scamdetector/__init__.py`
- Create: `tests/test_environment.py`

**Interfaces:**
- Produces: `scamdetector.__version__: str == "0.1.0"`; installed distribution name `scamdetector`; a pytest suite importable from any working directory

- [ ] **Step 1: Write the failing environment tests**

Create `tests/test_environment.py`:
```python
import subprocess
import sys
from importlib.metadata import version

import scamdetector


def test_interpreter_is_python_312():
    assert sys.version_info[:2] == (3, 12), f"Expected 3.12, got {sys.version.split()[0]}"


def test_package_version_matches_declared_version():
    assert scamdetector.__version__ == "0.1.0"
    assert version("scamdetector") == scamdetector.__version__


def test_package_imports_from_another_directory(tmp_path):
    result = subprocess.run(
        [sys.executable, "-c", "import scamdetector; print(scamdetector.__version__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"
```

- [ ] **Step 2: Run the tests to verify they fail for the right reason**

Run:
```cmd
.venv\Scripts\python -m pytest tests\test_environment.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'scamdetector'` (the package does not exist yet). A failure for any other reason means fix the environment before continuing.

- [ ] **Step 3: Create the package and its metadata**

Create `src/scamdetector/__init__.py`:
```python
"""Cyber scam detection package."""

__version__ = "0.1.0"
```

Create `pyproject.toml` with: `[build-system]` using `setuptools>=68` and `setuptools.build_meta`; `[project]` with `name = "scamdetector"`, `version = "0.1.0"`, `requires-python = ">=3.12,<3.13"`, `description`, and no runtime dependencies listed (they live in `requirements.txt`); `[tool.setuptools.packages.find]` with `where = ["src"]`.

- [ ] **Step 4: Install editable and confirm the import contract**

Run:
```cmd
uv pip install -e .
.venv\Scripts\python -c "import scamdetector; print(scamdetector.__version__)"
```
Expected: `0.1.0`.

- [ ] **Step 5: Run the tests to verify they pass**

Run:
```cmd
.venv\Scripts\python -m pytest tests\test_environment.py -v
```
Expected: 3 passed.

- [ ] **Step 6: Commit**

```cmd
git add pyproject.toml src/scamdetector/__init__.py tests/test_environment.py
git commit -m "feat: add importable scamdetector package with environment smoke tests"
```

---

### Task 4: Pytest configuration and suite gate

**Files:**
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: marker vocabulary (`unit`, `integration`, `app`, `slow`) and `testpaths = ["tests"]` used by every later phase

- [ ] **Step 1: Add the pytest section**

Append to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
markers = [
    "unit: pure-function tests",
    "integration: cross-module tests",
    "app: Streamlit AppTest tests",
    "slow: tests that train a model or read the full dataset",
]
```

- [ ] **Step 2: Verify the suite gate**

Run:
```cmd
.venv\Scripts\python -m pytest
```
Expected: `3 passed`, exit code 0. Confirm the exit code explicitly:
```cmd
echo %ERRORLEVEL%
```
Expected: `0`.

- [ ] **Step 3: Verify the environment test actually guards the interpreter**

Run:
```cmd
"C:\Program Files\Python314\python.exe" -m pytest tests\test_environment.py::test_interpreter_is_python_312 -v
```
Expected: FAIL with `Expected 3.12, got 3.14.3`, proving the guard works. This command is diagnostic only and is not part of the passing suite.

- [ ] **Step 4: Commit**

```cmd
git add pyproject.toml
git commit -m "chore: configure pytest paths and phase markers"
```

---

### Task 5: Repository hygiene, licensing and README setup

**Files:**
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `README.md`

**Interfaces:**
- Produces: a repository whose tracked files exclude environments, caches and generated data; MIT code license kept distinct from dataset terms

- [ ] **Step 1: Write `.gitignore`**

Include at minimum: `.venv/`, `.venv-verify/`, `__pycache__/`, `*.py[cod]`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `data/processed/`, `.streamlit/secrets.toml`, `*.log`, `Thumbs.db`, `.DS_Store`.

Do **not** ignore `models/*.joblib` (decision 2) or `reports/*.json` (evidence).

- [ ] **Step 2: Write `LICENSE` (MIT) and `THIRD_PARTY_NOTICES.md`**

`LICENSE`: standard MIT text, copyright line `Copyright (c) 2026 Charan A M` — confirm the name before committing.
`THIRD_PARTY_NOTICES.md`: state that the UCI SMS Spam Collection is used under **CC BY 4.0**, name the citation (Almeida, Hidalgo & Yamakami, 2011, *Contributions to the Study of SMS Spam Filtering*), and state plainly that the MIT license does not cover the dataset. Record the exact dataset URL and license link in Phase 1 when the file is downloaded; do not invent them now.

- [ ] **Step 3: Write the README setup section**

Include: project one-liner, status (`Phase 0 complete`), prerequisites (uv, Python 3.12.16), and the setup block:
```cmd
cd /d "C:\Users\Charan A M\CyberScamDetection"
uv venv --python 3.12.16 .venv
.venv\Scripts\activate
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install -e .
python -m pytest
streamlit run app\streamlit_app.py
```
Mark the last line as "available from Phase 3". Every command above must be one that actually ran in Tasks 1–4.

- [ ] **Step 4: Verify repository hygiene**

Run:
```cmd
git add -A
git status --short
git ls-files | findstr /i /c:".venv" /c:"__pycache__" /c:".pyc"
```
Expected: `git status` shows only intended files; the `git ls-files` search prints nothing.

- [ ] **Step 5: Commit**

```cmd
git commit -m "docs: add MIT license, dataset attribution notice and setup instructions"
```

---

### Task 6: Phase 0 exit verification

**Files:**
- Create: `docs/phase0-verification.md`

**Interfaces:**
- Consumes: everything produced in Tasks 1–5
- Produces: recorded evidence that Phase 0 acceptance criteria are met

- [ ] **Step 1: Run the full acceptance sequence**

Run and record the exact output of each:
```cmd
.venv\Scripts\python -V
.venv\Scripts\python -m pytest
echo %ERRORLEVEL%
.venv\Scripts\python -c "import scamdetector; print(scamdetector.__version__)"
git status --short
```

Expected: `Python 3.12.16`; `3 passed`; `0`; `0.1.0`; empty status.

- [ ] **Step 2: Write `docs/phase0-verification.md`**

Record the date, the five commands, their verbatim output, and the resolved versions of pandas, numpy, scikit-learn, joblib, streamlit and pytest. Note that Python 3.14 was deliberately not used.

- [ ] **Step 3: Commit**

```cmd
git add docs/phase0-verification.md
git commit -m "docs: record phase 0 verification evidence"
```

- [ ] **Step 4: Stop**

Do not begin Phase 1. Report the verification output and wait for approval.

---

## Phase 0 Acceptance Criteria (mapped to the spec)

| Criterion | Evidence |
|---|---|
| `.venv\Scripts\python -V` reports 3.12.16 | Task 1 Step 3, Task 6 Step 1 |
| Interpreter guard passes; 3.14 fails it | Task 3 tests, Task 4 Step 3 |
| Dependencies pinned from resolver output | Task 2 Steps 2–3 |
| Clean-venv install from manifests succeeds | Task 2 Step 4 |
| `python -m pytest` exits 0 with ≥2 genuine tests | Task 3 Step 5, Task 4 Step 2 |
| `scamdetector.__version__` is `0.1.0` and matches distribution metadata | Task 3 Step 4 and its test |
| No venv/caches tracked by Git | Task 5 Step 4 |
| MIT for code; dataset terms separate | Task 5 Step 2 |
| README setup commands are the ones actually run | Task 5 Step 3 |
| Evidence committed | Task 6 |

## Out of Scope for This Plan

Phase 1 (dataset) begins only after Phase 0 approval. No dataset download, no model code, no Streamlit app, no Docker in this plan.
