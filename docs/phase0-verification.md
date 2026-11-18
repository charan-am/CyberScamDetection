# Phase 0 Verification Evidence

**Date:** 2026-11-18
**Interpreter:** CPython 3.12.16 (uv-managed build)
**Platform:** Windows
**Python 3.14 status:** deliberately not used; `C:\Program Files\Python314\python.exe` was only invoked to prove the interpreter guard rejects it.

## Acceptance sequence

### 1. Interpreter version

```cmd
.venv\Scripts\python -V
```

```text
Python 3.12.16
```

### 2. Test suite

```cmd
.venv\Scripts\python -m pytest
```

```text
tests\test_environment.py ...                                            [100%]

============================== 3 passed in 0.13s ==============================
```

### 3. Exit code

```cmd
echo %ERRORLEVEL%
```

```text
0
```

### 4. Package import contract

```cmd
.venv\Scripts\python -c "import scamdetector; print(scamdetector.__version__)"
```

```text
0.1.0
```

### 5. Working tree

```cmd
git status --short
```

```text
(no output)
```

## Interpreter guard proof

The suite's assertion was run under the system Python 3.14.3 interpreter:

```text
AssertionError: Expected 3.12, got 3.14.3
```

pytest itself is not installed on the 3.14 interpreter, and installing it there would
add an unnecessary dependency, so the guard was proven at the assertion level.

## Clean-environment install proof

`.venv-verify` was created from the committed manifests, installed all 44 pinned
distributions, and reported:

```text
import pandas, numpy, sklearn, joblib, streamlit, pytest -> ok
```

The scratch environment was then deleted. It was created by this task only.

## Resolved dependency versions

| Package | Version |
|---|---|
| pandas | 3.1.1 |
| numpy | 2.5.5 |
| scikit-learn | 1.9.1 |
| joblib | 1.6.0 |
| streamlit | 1.67.0 |
| pytest | 9.2.0 |

`requirements.txt` pins the full resolved runtime set (39 distributions);
`requirements-dev.txt` pins pytest. Versions were captured from `uv pip freeze`,
not typed by hand.

## Commits produced

```text
241b989 chore: pin python 3.12.16 interpreter contract
f6a0d00 chore: pin resolved runtime and test dependencies
c6684c2 feat: add importable scamdetector package with environment smoke tests
aa536ca chore: configure pytest paths and phase markers
73ad30b docs: add approved design and phase 0 implementation plan
255f22a docs: add MIT license, dataset attribution notice and setup instructions
```

## Phase 1

Not started. Phase 1 begins only after explicit approval.
