"""Build CERES extensions while keeping build work outside the source tree."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def build_root() -> pathlib.Path:
    """Use an explicit build root, configured temporary storage or the user cache."""
    configured = os.environ.get("FEROS_BUILD_ROOT") or os.environ.get("TMPDIR")
    cache = pathlib.Path(os.environ.get("XDG_CACHE_HOME", str(pathlib.Path.home() / ".cache")))
    return pathlib.Path(configured).expanduser().resolve() if configured else cache / "feros-pipeline" / "build"


def run_command(command: list[str], cwd: pathlib.Path, scratch: pathlib.Path) -> None:
    env = os.environ.copy()
    env["TMPDIR"] = str(scratch)
    cache = pathlib.Path(env.get("XDG_CACHE_HOME", str(pathlib.Path.home() / ".cache")))
    env.setdefault("MPLCONFIGDIR", str(cache / "feros-pipeline" / "matplotlib"))
    shown_cwd = cwd.relative_to(REPO_ROOT)
    print(f"+ {' '.join(command)}  # cwd={shown_cwd}", flush=True)
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def main() -> int:
    destination = build_root()
    destination.mkdir(parents=True, exist_ok=True)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="feros-build-", dir=str(destination)))
    print(f"Build work directory: {scratch}", flush=True)
    try:
        run_command(
            [sys.executable, "setup.py", "build_ext",
             "--build-temp", str(scratch / "marsh" / "temp"),
             "--build-lib", str(scratch / "marsh" / "lib"), "--inplace"],
            REPO_ROOT / "utils" / "OptExtract",
            scratch,
        )
        run_command(
            [sys.executable, "-m", "numpy.f2py", "-c", "--fcompiler=gnu95",
             "--f77flags=-ffixed-line-length-none", "--build-dir", str(scratch / "ccf"),
             "CCF.pyf", "CCF.f"],
            REPO_ROOT / "utils" / "CCF",
            scratch,
        )
    finally:
        print(f"Build work retained for inspection: {scratch}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
