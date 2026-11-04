from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from astropy.io import fits

from feros_diagnostics_common import ensure_output_dir, sha256_file, write_json

SCIENCE_DPR_TYPES = {"OBJECT,WAVE", "VELOC,WAVE", "OBJECT,SKY", "VELOC,SKY"}
CALIBRATION_PATTERNS = [
    "MasterBias.fits",
    "Flat.fits",
    "trace.pkl",
    "P*.fits",
    "S_flat*.fits",
    "FEROS.*spec*.fits.S",
    "FEROS.*wavsolpars.pkl",
    "FEROS.*_sp_ob.fits",
    "FEROS.*_sp_co.fits",
    "shifts.pkl",
    "ThAr_Ne_shifts.dat",
    "thar_shifts.pdf",
]
CALIBRATION_TRANSFER_MODES = ("copy", "hardlink", "symlink")
CALIBRATION_ACCESS_MODES = ("calibdir", *CALIBRATION_TRANSFER_MODES)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_value(args: list[str], cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()
    except Exception:  # noqa: BLE001 - provenance should not block planning/dry-runs
        return ""


def safe_target_name(target: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", target.strip())
    return safe.strip("._") or "unnamed_target"


def discover_targets(raw_dir: Path) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(raw_dir.glob("*.fits")):
        try:
            header = fits.getheader(path)
        except Exception:
            continue
        dpr_type = str(header.get("HIERARCH ESO DPR TYPE", "")).strip()
        if dpr_type not in SCIENCE_DPR_TYPES:
            continue
        target = str(header.get("OBJECT", "")).strip()
        if not target or target.lower() in seen:
            continue
        seen.add(target.lower())
        targets.append(
            {
                "target": target,
                "safe_target": safe_target_name(target),
                "first_file": str(path),
                "dpr_type": dpr_type,
                "mjd_obs": header.get("MJD-OBS", header.get("MJD")),
                "date_obs": header.get("DATE-OBS"),
            }
        )
    return targets


def select_targets(discovered: list[dict[str, Any]], requested: list[str] | None, max_targets: int | None, all_targets: bool) -> list[dict[str, Any]]:
    if requested:
        by_name = {row["target"].lower(): row for row in discovered}
        missing = [target for target in requested if target.lower() not in by_name]
        if missing:
            raise SystemExit(f"requested targets not found in raw directory: {', '.join(missing)}")
        return [by_name[target.lower()] for target in requested]
    if all_targets:
        return discovered
    return discovered[: max_targets or 2]


def output_root_from_arg(path: Path, root: Path) -> Path:
    output_root = path if path.is_absolute() else root / path
    output_root = output_root.resolve()
    outputs_root = (root / "outputs").resolve()
    if outputs_root not in [output_root, *output_root.parents]:
        raise SystemExit(f"output root must be inside private outputs/: {output_root}")
    return output_root


def dirout_arg(path: Path) -> str:
    return str(path.resolve()) + os.sep


def feros_command(root: Path, raw_dir: Path, dirout: Path, target: str, npools: int, calibdir: Path | None = None) -> list[str]:
    command = [
        "uv",
        "--cache-dir",
        ".uv-cache",
        "run",
        "python",
        str(root / "feros" / "ferospipe.py"),
        str(raw_dir),
        "-dirout",
        dirout_arg(dirout),
        "-o2do",
        target,
        "-npools",
        str(npools),
        "-avoid_plot",
    ]
    if calibdir is not None:
        command.extend(["-calibdir", dirout_arg(calibdir)])
    return command


def calibration_compatibility_command(root: Path, raw_dir: Path, calibration_dir: Path, output_dir: Path, label: str) -> list[str]:
    return [
        "uv",
        "--cache-dir",
        ".uv-cache",
        "run",
        "python",
        str(root / "scripts" / "check_calibration_compatibility.py"),
        str(raw_dir),
        str(calibration_dir),
        "--output-dir",
        str(output_dir),
        "--label",
        label,
    ]


def run_logged(command: list[str], cwd: Path, log_path: Path) -> dict[str, Any]:
    start = time.monotonic()
    started_at = utc_now()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    with log_path.open("w") as log_handle:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=log_handle, stderr=subprocess.STDOUT, text=True)
    ended_at = utc_now()
    return {
        "command": command,
        "log_path": str(log_path),
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": round(time.monotonic() - start, 3),
        "returncode": process.returncode,
    }


def run_calibration_compatibility_check(root: Path, raw_dir: Path, calibration_dir: Path, output_dir: Path) -> dict[str, Any]:
    diagnostics_dir = ensure_output_dir(output_dir)
    command = calibration_compatibility_command(root, raw_dir, calibration_dir, diagnostics_dir, "calibration_seed")
    result = run_logged(command, root, diagnostics_dir / "run.log")
    summary_path = diagnostics_dir / "calibration_compatibility_summary.json"
    result["output_dir"] = str(diagnostics_dir)
    result["summary_path"] = str(summary_path)
    if summary_path.exists():
        try:
            result["summary"] = json.loads(summary_path.read_text())
        except Exception as exc:  # noqa: BLE001 - preserve diagnostics failure context in manifest
            result["summary_error"] = repr(exc)
    else:
        result["summary_error"] = "summary JSON was not written"
    return result


def calibration_files(calibration_dir: Path) -> list[Path]:
    files: set[Path] = set()
    for pattern in CALIBRATION_PATTERNS:
        files.update(path for path in calibration_dir.glob(pattern) if path.is_file())
    return sorted(files)


def transfer_calibration_file(source: Path, destination: Path, mode: str) -> None:
    source = source.resolve()
    destination = destination.resolve()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"calibration destination already exists: {destination}")
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "hardlink":
        os.link(source, destination)
    elif mode == "symlink":
        link_target = os.path.relpath(source, start=destination.parent)
        os.symlink(link_target, destination)
    else:
        raise ValueError(f"unsupported calibration transfer mode: {mode}")


def calibration_transfer_metadata(source: Path, destination: Path, mode: str) -> dict[str, Any]:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    source_stat = source_resolved.stat()
    destination_stat = destination_resolved.stat()
    row = {
        "file": source.name,
        "mode": mode,
        "source": str(source_resolved),
        "destination": str(destination),
        "destination_resolved": str(destination_resolved),
        "size_bytes": destination_stat.st_size,
        "sha256": sha256_file(destination),
        "same_device": source_stat.st_dev == destination_stat.st_dev,
        "same_inode": source_stat.st_dev == destination_stat.st_dev and source_stat.st_ino == destination_stat.st_ino,
        "destination_is_symlink": destination.is_symlink(),
        "destination_link_count": destination_stat.st_nlink,
    }
    if destination.is_symlink():
        row["symlink_target"] = os.readlink(destination)
    return row


def transfer_calibration_files(calibration_dir: Path, target_dir: Path, mode: str) -> list[dict[str, Any]]:
    transferred: list[dict[str, Any]] = []
    for source in calibration_files(calibration_dir):
        destination = target_dir / source.name
        transfer_calibration_file(source, destination, mode)
        transferred.append(calibration_transfer_metadata(source, destination, mode))
    return transferred


def copy_calibration_files(calibration_dir: Path, target_dir: Path) -> list[dict[str, Any]]:
    return transfer_calibration_files(calibration_dir, target_dir, "copy")


def copy_proc_product(product: Path, collected_proc: Path, target: str, kind: str) -> dict[str, Any]:
    destination = collected_proc / product.name
    shutil.copy2(product, destination)
    return {
        "target": target,
        "kind": kind,
        "source": str(product),
        "destination": str(destination),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def count_result_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.strip())


def collect_target_products(output_root: Path, target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    collected_proc = ensure_output_dir(output_root / "collected" / "proc")
    collected: list[dict[str, Any]] = []
    aggregate_results = collected_proc / "results.txt"
    with aggregate_results.open("w") as aggregate:
        for target_row in target_rows:
            target_dir = output_root / "targets" / target_row["safe_target"]
            proc_dir = target_dir / "proc"
            for product in sorted(proc_dir.glob("*_sp.fits")) + sorted(proc_dir.glob("*_XCs_*.pdf")):
                collected.append(copy_proc_product(product, collected_proc, target_row["target"], "proc_product"))
            for product in sorted(target_dir.glob("*_XC_*.pkl")):
                collected.append(copy_proc_product(product, collected_proc, target_row["target"], "ccf_pickle"))
            results_path = proc_dir / "results.txt"
            if results_path.exists():
                named_results = collected_proc / f"{target_row['safe_target']}__results.txt"
                shutil.copy2(results_path, named_results)
                aggregate.write(results_path.read_text())
                collected.append(
                    {
                        "target": target_row["target"],
                        "kind": "results",
                        "source": str(results_path),
                        "destination": str(named_results),
                        "result_rows": count_result_rows(named_results),
                        "size_bytes": named_results.stat().st_size,
                        "sha256": sha256_file(named_results),
                    }
                )
    collected.append(
        {
            "target": "all",
            "kind": "aggregate_results",
            "source": "",
            "destination": str(aggregate_results),
            "result_rows": count_result_rows(aggregate_results),
            "size_bytes": aggregate_results.stat().st_size,
            "sha256": sha256_file(aggregate_results),
        }
    )
    return collected


def collection_summary(output_root: Path, target_rows: list[dict[str, Any]], collected_products: list[dict[str, Any]]) -> dict[str, Any]:
    collected_proc = output_root / "collected" / "proc"
    kinds = Counter(str(row.get("kind", "")) for row in collected_products)
    target_summaries: list[dict[str, Any]] = []
    missing_targets: list[str] = []
    for target_row in target_rows:
        target = target_row["target"]
        target_products = [row for row in collected_products if row.get("target") == target]
        target_kind_counts = Counter(str(row.get("kind", "")) for row in target_products)
        result_rows = sum(int(row.get("result_rows") or 0) for row in target_products if row.get("kind") == "results")
        summary = {
            "target": target,
            "safe_target": target_row["safe_target"],
            "collected_product_count": len(target_products),
            "kind_counts": dict(sorted(target_kind_counts.items())),
            "result_rows": result_rows,
        }
        if target_kind_counts.get("proc_product", 0) == 0 or result_rows == 0:
            missing_targets.append(target)
        target_summaries.append(summary)
    return {
        "collected_proc": str(collected_proc),
        "collected_product_count": len(collected_products),
        "kind_counts": dict(sorted(kinds.items())),
        "aggregate_result_rows": count_result_rows(collected_proc / "results.txt"),
        "target_count": len(target_rows),
        "targets_with_missing_products": missing_targets,
        "target_summaries": target_summaries,
    }


def run_parallel_reduction(
    raw_dir: Path,
    output_root_arg: Path,
    *,
    targets_requested: list[str] | None = None,
    max_targets: int | None = 2,
    all_targets: bool = False,
    target_workers: int = 2,
    calibration_npools: int = 24,
    target_npools_arg: int | None = None,
    legacy_npools_arg: int | None = None,
    calibration_access_mode: str = "calibdir",
    dry_run: bool = False,
) -> int:
    root = repo_root()
    raw_dir = raw_dir.resolve()
    if not raw_dir.is_dir():
        raise SystemExit(f"raw directory does not exist: {raw_dir}")
    if calibration_access_mode not in CALIBRATION_ACCESS_MODES:
        raise SystemExit(f"unsupported calibration access mode: {calibration_access_mode}")
    target_npools = target_npools_arg if target_npools_arg is not None else (legacy_npools_arg if legacy_npools_arg is not None else 2)
    if target_workers < 1 or calibration_npools < 1 or target_npools < 1:
        raise SystemExit("--target-workers, --calibration-npools, and --target-npools must be positive")

    output_root = output_root_from_arg(output_root_arg, root)
    discovered = discover_targets(raw_dir)
    targets = select_targets(discovered, targets_requested, max_targets, all_targets)
    if not targets:
        raise SystemExit(f"no science targets found in {raw_dir}")

    calibration_dir = output_root / "calibration"
    compatibility_dir = output_root / "diagnostics" / "calibration_compatibility"
    manifest: dict[str, Any] = {
        "status": "planned" if dry_run else "started",
        "created_at": utc_now(),
        "repo_root": str(root),
        "git_commit": git_value(["rev-parse", "HEAD"], root),
        "git_status_short": git_value(["status", "--short"], root),
        "raw_dir": str(raw_dir),
        "output_root": str(output_root),
        "target_workers": target_workers,
        "calibration_npools": calibration_npools,
        "target_npools": target_npools,
        "legacy_npools_argument": legacy_npools_arg,
        "estimated_target_worker_pool_slots": target_workers * target_npools,
        "calibration_access_mode": calibration_access_mode,
        "discovered_target_count": len(discovered),
        "targets": targets,
        "calibration_dir": str(calibration_dir),
        "calibration_seed_target": targets[0]["target"],
        "calibration_patterns": CALIBRATION_PATTERNS,
        "calibration_compatibility_dir": str(compatibility_dir),
        "calibration_compatibility_command": calibration_compatibility_command(
            root,
            raw_dir,
            calibration_dir,
            compatibility_dir,
            "calibration_seed",
        ),
    }

    if dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output root already exists and is not empty: {output_root}")
    ensure_output_dir(output_root)
    write_json(output_root / "run_manifest.json", manifest)

    ensure_output_dir(calibration_dir)
    calibration_command = feros_command(root, raw_dir, calibration_dir, targets[0]["target"], calibration_npools)
    calibration_result = run_logged(calibration_command, root, calibration_dir / "run.log")
    manifest["calibration_result"] = calibration_result
    manifest["calibration_files"] = [
        {"file": path.name, "path": str(path), "sha256": sha256_file(path)} for path in calibration_files(calibration_dir)
    ]
    write_json(output_root / "run_manifest.json", manifest)
    if calibration_result["returncode"] != 0:
        manifest["status"] = "failed_calibration"
        write_json(output_root / "run_manifest.json", manifest)
        return int(calibration_result["returncode"])

    compatibility_result = run_calibration_compatibility_check(root, raw_dir, calibration_dir, compatibility_dir)
    manifest["calibration_compatibility_result"] = compatibility_result
    write_json(output_root / "run_manifest.json", manifest)
    compatibility_status = compatibility_result.get("summary", {}).get("status")
    if compatibility_result["returncode"] != 0 or compatibility_status != "compatible":
        manifest["status"] = "failed_calibration_compatibility"
        write_json(output_root / "run_manifest.json", manifest)
        print(f"Calibration compatibility check failed; see {compatibility_dir}")
        return int(compatibility_result["returncode"] or 1)

    for target_row in targets:
        target_dir = ensure_output_dir(output_root / "targets" / target_row["safe_target"])
        if calibration_access_mode == "calibdir":
            target_row["calibration_access"] = {
                "mode": "calibdir",
                "calibration_dir": str(calibration_dir),
            }
            target_row["command"] = feros_command(root, raw_dir, target_dir, target_row["target"], target_npools, calibdir=calibration_dir)
        else:
            target_row["calibration_transfers"] = transfer_calibration_files(calibration_dir, target_dir, calibration_access_mode)
            target_row["command"] = feros_command(root, raw_dir, target_dir, target_row["target"], target_npools)
        target_row["log_path"] = str(target_dir / "run.log")
    write_json(output_root / "run_manifest.json", manifest)

    def run_target(target_row: dict[str, Any]) -> dict[str, Any]:
        target_dir = output_root / "targets" / target_row["safe_target"]
        result = run_logged(target_row["command"], root, target_dir / "run.log")
        return {"target": target_row["target"], "safe_target": target_row["safe_target"], **result}

    with concurrent.futures.ThreadPoolExecutor(max_workers=target_workers) as executor:
        target_results = list(executor.map(run_target, targets))

    manifest["target_results"] = target_results
    manifest["collected_products"] = collect_target_products(output_root, targets)
    manifest["collection_summary"] = collection_summary(output_root, targets, manifest["collected_products"])
    manifest["finished_at"] = utc_now()
    manifest["status"] = "complete" if all(result["returncode"] == 0 for result in target_results) else "failed_targets"
    write_json(output_root / "run_manifest.json", manifest)

    print(f"Wrote parallel reduction manifest to {output_root / 'run_manifest.json'}")
    print(f"status: {manifest['status']}")
    return 0 if manifest["status"] == "complete" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Experimental isolated target-level FEROS reduction wrapper.")
    parser.add_argument("raw_dir", type=Path, help="Raw FEROS 4k input directory.")
    parser.add_argument("output_root", type=Path, help="Output root under this repo's outputs/ directory.")
    parser.add_argument("--targets", nargs="+", help="Specific OBJECT names to reduce.")
    parser.add_argument("--max-targets", type=int, default=2, help="Number of discovered targets to run when --targets/--all-targets is not set.")
    parser.add_argument("--all-targets", action="store_true", help="Run every discovered science target.")
    parser.add_argument("--target-workers", type=int, default=2, help="Number of simultaneous target workers.")
    parser.add_argument("--calibration-npools", type=int, default=24, help="FEROS -npools value for the single calibration-seed run.")
    parser.add_argument("--target-npools", type=int, default=None, help="FEROS -npools value for each target worker. Defaults to --npools or 2.")
    parser.add_argument("--npools", type=int, default=None, help="Backward-compatible alias for --target-npools.")
    parser.add_argument(
        "--calibration-access-mode",
        "--calibration-transfer-mode",
        dest="calibration_access_mode",
        choices=CALIBRATION_ACCESS_MODES,
        default="calibdir",
        help="How target workers access calibration products. Use calibdir for the explicit read-only calibration cache.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the planned run without creating outputs or running reductions.")
    args = parser.parse_args()
    return run_parallel_reduction(
        args.raw_dir,
        args.output_root,
        targets_requested=args.targets,
        max_targets=args.max_targets,
        all_targets=args.all_targets,
        target_workers=args.target_workers,
        calibration_npools=args.calibration_npools,
        target_npools_arg=args.target_npools,
        legacy_npools_arg=args.npools,
        calibration_access_mode=args.calibration_access_mode,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
