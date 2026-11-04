from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

from astropy.io import fits

import reduce_targets_parallel as parallel


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_output_root(raw_dir: Path, root: Path) -> Path:
    name = raw_dir.resolve().name.rstrip(os.sep)
    suffix = "_red" if not name.endswith("_red") else ""
    return root / "outputs" / f"{name}{suffix}"


def count_science_frames(raw_dir: Path) -> int:
    count = 0
    for path in sorted(raw_dir.glob("*.fits")):
        try:
            header = fits.getheader(path)
        except Exception:
            continue
        dpr_type = str(header.get("HIERARCH ESO DPR TYPE", "")).strip()
        if dpr_type in parallel.SCIENCE_DPR_TYPES:
            count += 1
    return count


def node_suggestion(target_count: int) -> dict[str, Any]:
    cpu_count = os.cpu_count() or 1
    try:
        load_1m = os.getloadavg()[0]
    except OSError:
        load_1m = 0.0
    free_cpus = max(1.0, float(cpu_count) - load_1m)
    suggested_slots = max(1, min(24, int(free_cpus * 0.25)))
    target_npools = 2 if suggested_slots >= 4 else 1
    target_workers = max(1, min(target_count, 8, max(1, suggested_slots // target_npools)))
    calibration_npools = max(1, min(24, suggested_slots))
    return {
        "cpu_count": cpu_count,
        "load_1m": round(load_1m, 2),
        "free_cpus_estimate": round(free_cpus, 2),
        "calibration_npools": calibration_npools,
        "target_workers": target_workers,
        "target_npools": target_npools,
        "estimated_target_worker_pool_slots": target_workers * target_npools,
    }


def prompt_text(label: str, default: str) -> str:
    value = input(f"{label} [{default}]: ").strip()
    return value or default


def prompt_int(label: str, default: int) -> int:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            parsed = int(value)
        except ValueError:
            print("Enter an integer.")
            continue
        if parsed < 1:
            print("Enter a positive integer.")
            continue
        return parsed


def should_prompt(args: argparse.Namespace) -> bool:
    return not args.yes and not args.dry_run and sys.stdin.isatty() and sys.stdout.isatty()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reduce a FEROS night with the validated target-parallel calibration-cache workflow."
    )
    parser.add_argument("raw_dir", type=Path, help="Raw FEROS 4k input directory.")
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        help="Output root under this repo's outputs/ directory. Defaults to outputs/<raw-dir-name>_red.",
    )
    parser.add_argument("--targets", nargs="+", help="Advanced: reduce only these OBJECT names.")
    parser.add_argument("--target-workers", type=int, help="Advanced: simultaneous target workers.")
    parser.add_argument("--target-npools", type=int, help="Advanced: FEROS -npools value for each target worker.")
    parser.add_argument("--calibration-npools", type=int, help="Advanced: FEROS -npools value for the calibration seed.")
    parser.add_argument("--yes", "-y", action="store_true", help="Accept suggested settings without interactive prompts.")
    parser.add_argument("--dry-run", action="store_true", help="Print the planned run without creating outputs or reducing data.")
    args = parser.parse_args()

    root = repo_root()
    raw_dir = args.raw_dir.resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"raw directory does not exist: {raw_dir}")

    discovered = parallel.discover_targets(raw_dir)
    if not discovered:
        raise SystemExit(f"no FEROS science targets found in {raw_dir}")
    target_count = len(args.targets) if args.targets else len(discovered)
    science_frame_count = count_science_frames(raw_dir)
    suggestion = node_suggestion(target_count)

    output_root = args.output_root.resolve() if args.output_root and args.output_root.is_absolute() else args.output_root
    if output_root is None:
        output_root = default_output_root(raw_dir, root)
    calibration_npools = args.calibration_npools or suggestion["calibration_npools"]
    target_workers = args.target_workers or suggestion["target_workers"]
    target_npools = args.target_npools or suggestion["target_npools"]

    print("FEROS target-parallel reduction plan")
    print(f"  raw_dir: {raw_dir}")
    print(f"  output_root: {output_root}")
    print(f"  target names: {target_count}")
    print(f"  science frames: {science_frame_count}")
    print(f"  node CPUs: {suggestion['cpu_count']}, 1-minute load: {suggestion['load_1m']}")
    print(f"  calibration npools: {calibration_npools}")
    print(f"  target workers: {target_workers}")
    print(f"  target npools: {target_npools}")
    print(f"  estimated target pool slots: {target_workers * target_npools}")
    if not args.dry_run and "TMUX" not in os.environ:
        print("  note: this is a long reduction; run inside tmux if the session must survive disconnects.")

    if should_prompt(args):
        output_root = Path(prompt_text("Output root", str(output_root)))
        calibration_npools = prompt_int("Calibration npools", calibration_npools)
        target_workers = prompt_int("Target workers", target_workers)
        target_npools = prompt_int("Target npools", target_npools)
        proceed = input("Start reduction? [Y/n]: ").strip().lower()
        if proceed not in {"", "y", "yes"}:
            print("Cancelled.")
            return 1
    elif not args.yes and not args.dry_run:
        print("Non-interactive shell detected; using suggested settings. Pass --dry-run to inspect only.")

    return parallel.run_parallel_reduction(
        raw_dir,
        output_root,
        targets_requested=args.targets,
        max_targets=None,
        all_targets=True,
        target_workers=target_workers,
        calibration_npools=calibration_npools,
        target_npools_arg=target_npools,
        calibration_access_mode="calibdir",
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
