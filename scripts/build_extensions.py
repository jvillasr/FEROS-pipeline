"""Build local CERES extension modules for the active Python environment."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD_TMP = REPO_ROOT / ".build-tmp"


def run_command(command: list[str], cwd: pathlib.Path) -> None:
    env = os.environ.copy()
    env["TMPDIR"] = str(BUILD_TMP)
    env.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".matplotlib-cache"))
    BUILD_TMP.mkdir(exist_ok=True)
    shown_cwd = cwd.relative_to(REPO_ROOT)
    print(f"+ {' '.join(command)}  # cwd={shown_cwd}", flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def clean_transient_build_dirs() -> None:
    for path in (
        REPO_ROOT / "utils" / "OptExtract" / "build",
        BUILD_TMP,
    ):
        if path.exists():
            shutil.rmtree(path)


def main() -> int:
    try:
        run_command(
            [sys.executable, "setup.py", "build_ext", "--inplace"],
            REPO_ROOT / "utils" / "OptExtract",
        )
        run_command(
            [
                sys.executable,
                "-m",
                "numpy.f2py",
                "-c",
                "--fcompiler=gnu95",
                "--f77flags=-ffixed-line-length-none",
                "CCF.pyf",
                "CCF.f",
            ],
            REPO_ROOT / "utils" / "CCF",
        )
    finally:
        clean_transient_build_dirs()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
