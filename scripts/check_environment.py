"""Check the FEROS/CERES development Python environment.

This script validates dependency imports and the local CERES extension modules
without starting a reduction. It is intentionally small so it can be run after
each environment rebuild with:

    uv run python scripts/check_environment.py
"""

from __future__ import annotations

import importlib
import contextlib
import io
import os
import pathlib
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))


def add_local_import_paths() -> None:
    for relative_path in (
        "feros",
        "utils/Correlation",
        "utils/GLOBALutils",
        "utils/OptExtract",
        "utils/SSEphem",
        "utils/CCF",
    ):
        sys.path.insert(0, str(REPO_ROOT / relative_path))


def check_module(module_name: str) -> tuple[str, str]:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            importlib.import_module(module_name)
    except Exception as exc:  # noqa: BLE001 - report exact legacy import failure.
        return module_name, f"FAIL: {exc.__class__.__name__}: {exc}"
    return module_name, "OK"


def main() -> int:
    add_local_import_paths()
    modules = [
        "numpy",
        "scipy",
        "astropy",
        "matplotlib",
        "statsmodels",
        "ephem",
        "pandas",
        "pycurl",
        "PyAstronomy",
        "ferosutils",
        "correlation",
        "GLOBALutils",
        "Marsh",
        "CCF",
    ]
    results = [check_module(module_name) for module_name in modules]
    width = max(len(module_name) for module_name, _ in results)
    for module_name, status in results:
        print(f"{module_name:<{width}}  {status}")
    return 0 if all(status == "OK" for _, status in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
