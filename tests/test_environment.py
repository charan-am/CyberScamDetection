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
