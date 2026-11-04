from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from feros_diagnostics_common import ensure_output_dir, write_csv, write_json


LOG_MARKERS = {
    "done_with_extraction2": re.compile(r"done with extraction2"),
    "working_on_thar": re.compile(r"Working on ThAr\+Ne file"),
    "computing_wavelength_solution": re.compile(r"Computing wavelength solution of ThAr file"),
    "using_cached_wavelength_solution": re.compile(r"Using previously computed wavelength solution"),
    "start_global_culling": re.compile(r"Start Global culling"),
    "maxfev_warning": re.compile(r"maxfev"),
    "visible_deprecation_warning": re.compile(r"VisibleDeprecationWarning"),
    "selected_thar_reference": re.compile(r"^This:"),
    "target_listing": re.compile(r"The following targets will be processed"),
}


def parse_iso(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    normalised = value.replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(normalised)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def iso_from_timestamp(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).replace(microsecond=0).isoformat()


def seconds_between(start: dt.datetime | None, end: dt.datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return round((end - start).total_seconds(), 3)


def load_manifest(run_root: Path) -> dict[str, Any]:
    path = run_root / "run_manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def infer_npools(command: list[Any] | None) -> int | None:
    if not command:
        return None
    values = [str(value) for value in command]
    for index, value in enumerate(values):
        if value == "-npools" and index + 1 < len(values):
            try:
                return int(values[index + 1])
            except ValueError:
                return None
    return None


def outer_log_path(run_root: Path) -> Path:
    return run_root.parent / f"{run_root.name}_outer.log"


def parse_outer_log(run_root: Path) -> dict[str, Any]:
    path = outer_log_path(run_root)
    result: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if not path.exists():
        return result
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) != 2:
            continue
        key, value = parts
        if key in {"START_UTC", "END_UTC"}:
            result[key.lower()] = value
        elif key in {"DURATION_SECONDS", "EXIT_STATUS"}:
            try:
                result[key.lower()] = int(value)
            except ValueError:
                result[key.lower()] = value
    return result


def categorise_file(path: Path) -> str:
    name = path.name
    if name == "run_manifest.json":
        return "manifest"
    if name.endswith(".log") or name.startswith("reduced_data_"):
        return "log"
    if name in {"MasterBias.fits", "Flat.fits", "trace.pkl", "P.fits", "P_ob.fits", "P_co.fits"}:
        return "master_calibration"
    if name.startswith("S_flat") and name.endswith(".fits"):
        return "flat_extraction"
    if name.endswith("spec.ob.fits.S") or name.endswith("spec.co.fits.S"):
        return "optimal_extraction_product"
    if name.endswith("spec.simple.ob.fits.S") or name.endswith("spec.simple.co.fits.S"):
        return "simple_extraction_product"
    if name.endswith(".wavsolpars.pkl"):
        return "wavelength_solution_pickle"
    if name.endswith("_sp_ob.fits") or name.endswith("_sp_co.fits"):
        return "wavelength_solution_spectrum"
    if name in {"shifts.pkl", "ThAr_Ne_shifts.dat", "thar_shifts.pdf"}:
        return "thar_shift_selection"
    if "_XC_" in name and name.endswith(".pkl"):
        return "ccf_pickle"
    if name.endswith("_sp.fits"):
        return "science_spectrum"
    if name == "results.txt":
        return "results_table"
    if name.startswith("BAC_") and name.endswith(".fits"):
        return "science_intermediate"
    if name.endswith(".fits"):
        return "fits_other"
    if name.endswith(".pkl"):
        return "pickle_other"
    return "other"


def phase_entries(run_root: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    calibration = manifest.get("calibration_result") or {}
    if calibration:
        log_path = Path(str(calibration.get("log_path", "")))
        entries.append(
            {
                "phase": "calibration",
                "target": "calibration",
                "phase_dir": run_root / "calibration",
                "log_path": log_path if str(log_path) else run_root / "calibration" / "run.log",
                "started_at": calibration.get("started_at"),
                "ended_at": calibration.get("ended_at"),
                "duration_seconds": calibration.get("duration_seconds"),
                "returncode": calibration.get("returncode"),
                "npools": infer_npools(calibration.get("command")),
            }
        )
    for target_result in manifest.get("target_results") or []:
        log_path = Path(str(target_result.get("log_path", "")))
        target = str(target_result.get("target") or target_result.get("safe_target") or "target")
        safe_target = str(target_result.get("safe_target") or target)
        entries.append(
            {
                "phase": f"target:{safe_target}",
                "target": target,
                "phase_dir": log_path.parent if str(log_path) else run_root / "targets" / safe_target,
                "log_path": log_path,
                "started_at": target_result.get("started_at"),
                "ended_at": target_result.get("ended_at"),
                "duration_seconds": target_result.get("duration_seconds"),
                "returncode": target_result.get("returncode"),
                "npools": infer_npools(target_result.get("command")),
            }
        )
    if not entries:
        entries.append(
            {
                "phase": "run",
                "target": "",
                "phase_dir": run_root,
                "log_path": run_root / "run.log",
                "started_at": None,
                "ended_at": None,
                "duration_seconds": None,
                "returncode": None,
                "npools": None,
            }
        )
    return entries


def iter_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: (path.stat().st_mtime, str(path)))


def relative_to(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def file_rows_for_phase(label: str, run_root: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    phase_dir = Path(entry["phase_dir"])
    phase_start = parse_iso(entry.get("started_at"))
    phase_end = parse_iso(entry.get("ended_at"))
    files = iter_files(phase_dir)
    first_mtime = files[0].stat().st_mtime if files else None
    rows: list[dict[str, Any]] = []
    previous_row: dict[str, Any] | None = None
    for path in files:
        stat = path.stat()
        mtime = dt.datetime.fromtimestamp(stat.st_mtime, tz=dt.timezone.utc)
        seconds_since_phase_start = seconds_between(phase_start, mtime)
        seconds_since_first_file = round(stat.st_mtime - first_mtime, 3) if first_mtime is not None else None
        inside_phase_window = phase_start is None or phase_end is None or (phase_start <= mtime <= phase_end)
        row = {
            "label": label,
            "run_root": str(run_root),
            "phase": entry["phase"],
            "target": entry.get("target") or "",
            "relative_path": relative_to(path, run_root),
            "category": categorise_file(path),
            "size_bytes": stat.st_size,
            "mtime_utc": iso_from_timestamp(stat.st_mtime),
            "inside_phase_window": "yes" if inside_phase_window else "no",
            "seconds_since_phase_start": seconds_since_phase_start,
            "seconds_since_first_file": seconds_since_first_file,
            "seconds_since_previous_file": "",
            "previous_relative_path": "",
            "previous_category": "",
        }
        if previous_row is not None:
            previous_mtime = parse_iso(str(previous_row["mtime_utc"]))
            row["seconds_since_previous_file"] = seconds_between(previous_mtime, mtime)
            row["previous_relative_path"] = previous_row["relative_path"]
            row["previous_category"] = previous_row["category"]
        rows.append(row)
        previous_row = row
    return rows


def phase_summary(label: str, run_root: Path, entry: dict[str, Any], file_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mtimes = [parse_iso(str(row["mtime_utc"])) for row in file_rows]
    mtimes = [value for value in mtimes if value is not None]
    active_rows = [row for row in file_rows if row.get("inside_phase_window") == "yes"]
    active_mtimes = [parse_iso(str(row["mtime_utc"])) for row in active_rows]
    active_mtimes = [value for value in active_mtimes if value is not None]
    category_counts = Counter(str(row["category"]) for row in file_rows)
    active_category_counts = Counter(str(row["category"]) for row in active_rows)
    first_mtime = min(mtimes) if mtimes else None
    last_mtime = max(mtimes) if mtimes else None
    first_active_mtime = min(active_mtimes) if active_mtimes else None
    last_active_mtime = max(active_mtimes) if active_mtimes else None
    return {
        "label": label,
        "run_root": str(run_root),
        "phase": entry["phase"],
        "target": entry.get("target") or "",
        "npools": entry.get("npools"),
        "returncode": entry.get("returncode"),
        "started_at": entry.get("started_at") or "",
        "ended_at": entry.get("ended_at") or "",
        "duration_seconds": entry.get("duration_seconds"),
        "log_path": str(entry.get("log_path") or ""),
        "phase_dir": str(entry.get("phase_dir") or ""),
        "file_count": len(file_rows),
        "first_file_mtime": first_mtime.isoformat() if first_mtime else "",
        "last_file_mtime": last_mtime.isoformat() if last_mtime else "",
        "file_span_seconds": seconds_between(first_mtime, last_mtime),
        "active_file_count": len(active_rows),
        "first_active_file_mtime": first_active_mtime.isoformat() if first_active_mtime else "",
        "last_active_file_mtime": last_active_mtime.isoformat() if last_active_mtime else "",
        "active_file_span_seconds": seconds_between(first_active_mtime, last_active_mtime),
        "category_counts_json": json.dumps(dict(sorted(category_counts.items())), sort_keys=True),
        "active_category_counts_json": json.dumps(dict(sorted(active_category_counts.items())), sort_keys=True),
    }


def gap_rows(file_rows: list[dict[str, Any]], top_n: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in file_rows:
        if row.get("inside_phase_window") != "yes":
            continue
        grouped.setdefault((str(row["label"]), str(row["run_root"]), str(row["phase"])), []).append(row)
    result: list[dict[str, Any]] = []
    for rows in grouped.values():
        rows.sort(key=lambda row: str(row["mtime_utc"]))
        previous: dict[str, Any] | None = None
        for row in rows:
            if previous is None:
                previous = row
                continue
            previous_mtime = parse_iso(str(previous["mtime_utc"]))
            current_mtime = parse_iso(str(row["mtime_utc"]))
            gap_seconds = seconds_between(previous_mtime, current_mtime)
            if gap_seconds is not None:
                result.append(
                    {
                        "label": row["label"],
                        "run_root": row["run_root"],
                        "phase": row["phase"],
                        "target": row["target"],
                        "gap_seconds": gap_seconds,
                        "previous_relative_path": previous["relative_path"],
                        "previous_category": previous["category"],
                        "current_relative_path": row["relative_path"],
                        "current_category": row["category"],
                        "current_mtime_utc": row["mtime_utc"],
                    }
                )
            previous = row
    result.sort(key=lambda row: float(row["gap_seconds"]), reverse=True)
    return result[:top_n]


def log_marker_rows(label: str, run_root: Path, entry: dict[str, Any]) -> list[dict[str, Any]]:
    path = Path(str(entry.get("log_path") or ""))
    rows = {
        marker: {
            "label": label,
            "run_root": str(run_root),
            "phase": entry["phase"],
            "target": entry.get("target") or "",
            "log_path": str(path),
            "marker": marker,
            "count": 0,
            "first_line": "",
            "last_line": "",
        }
        for marker in LOG_MARKERS
    }
    if not path.exists():
        return list(rows.values())
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        for marker, pattern in LOG_MARKERS.items():
            if pattern.search(line):
                row = rows[marker]
                row["count"] = int(row["count"]) + 1
                if not row["first_line"]:
                    row["first_line"] = line_number
                row["last_line"] = line_number
    return list(rows.values())


def analyse_run(run_root: Path, label: str, top_gaps: int) -> dict[str, Any]:
    manifest = load_manifest(run_root)
    entries = phase_entries(run_root, manifest)
    all_file_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    for entry in entries:
        rows = file_rows_for_phase(label, run_root, entry)
        all_file_rows.extend(rows)
        phase_rows.append(phase_summary(label, run_root, entry, rows))
        marker_rows.extend(log_marker_rows(label, run_root, entry))
    return {
        "label": label,
        "run_root": str(run_root),
        "manifest_status": manifest.get("status", ""),
        "calibration_npools": manifest.get("calibration_npools") or infer_npools((manifest.get("calibration_result") or {}).get("command")),
        "target_npools": manifest.get("target_npools"),
        "target_workers": manifest.get("target_workers"),
        "outer_log": parse_outer_log(run_root),
        "phase_rows": phase_rows,
        "file_rows": all_file_rows,
        "gap_rows": gap_rows(all_file_rows, top_gaps),
        "log_marker_rows": marker_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise FEROS/CERES run timing from manifests, logs, and file mtimes.")
    parser.add_argument("run_roots", nargs="+", type=Path, help="Run roots, usually outputs/<run-name> directories.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Diagnostics output directory.")
    parser.add_argument("--label-prefix", default="", help="Optional prefix for run labels in output tables.")
    parser.add_argument("--top-gaps", type=int, default=12, help="Number of largest per-file gaps to report.")
    args = parser.parse_args()

    output_dir = ensure_output_dir(args.output_dir.resolve())
    analyses = []
    for run_root_arg in args.run_roots:
        run_root = run_root_arg.resolve()
        if not run_root.is_dir():
            raise SystemExit(f"run root does not exist: {run_root}")
        label = f"{args.label_prefix}{run_root.name}" if args.label_prefix else run_root.name
        analyses.append(analyse_run(run_root, label, args.top_gaps))

    phase_rows = [row for analysis in analyses for row in analysis["phase_rows"]]
    file_rows = [row for analysis in analyses for row in analysis["file_rows"]]
    gap_rows_all = [row for analysis in analyses for row in analysis["gap_rows"]]
    marker_rows = [row for analysis in analyses for row in analysis["log_marker_rows"]]

    write_csv(
        output_dir / "phase_timing.csv",
        phase_rows,
        [
            "label",
            "run_root",
            "phase",
            "target",
            "npools",
            "returncode",
            "started_at",
            "ended_at",
            "duration_seconds",
            "log_path",
            "phase_dir",
            "file_count",
            "first_file_mtime",
            "last_file_mtime",
            "file_span_seconds",
            "active_file_count",
            "first_active_file_mtime",
            "last_active_file_mtime",
            "active_file_span_seconds",
            "category_counts_json",
            "active_category_counts_json",
        ],
    )
    write_csv(
        output_dir / "file_timeline.csv",
        file_rows,
        [
            "label",
            "run_root",
            "phase",
            "target",
            "relative_path",
            "category",
            "size_bytes",
            "mtime_utc",
            "inside_phase_window",
            "seconds_since_phase_start",
            "seconds_since_first_file",
            "seconds_since_previous_file",
            "previous_relative_path",
            "previous_category",
        ],
    )
    write_csv(
        output_dir / "largest_file_gaps.csv",
        gap_rows_all,
        [
            "label",
            "run_root",
            "phase",
            "target",
            "gap_seconds",
            "previous_relative_path",
            "previous_category",
            "current_relative_path",
            "current_category",
            "current_mtime_utc",
        ],
    )
    write_csv(
        output_dir / "log_markers.csv",
        marker_rows,
        ["label", "run_root", "phase", "target", "log_path", "marker", "count", "first_line", "last_line"],
    )
    summary = {
        "output_dir": str(output_dir),
        "runs": [
            {
                "label": analysis["label"],
                "run_root": analysis["run_root"],
                "manifest_status": analysis["manifest_status"],
                "calibration_npools": analysis["calibration_npools"],
                "target_npools": analysis["target_npools"],
                "target_workers": analysis["target_workers"],
                "outer_log": analysis["outer_log"],
                "phases": analysis["phase_rows"],
                "largest_gaps": analysis["gap_rows"],
            }
            for analysis in analyses
        ],
    }
    write_json(output_dir / "timing_summary.json", summary)

    print(f"Wrote timing diagnostics to {output_dir}")
    for analysis in analyses:
        calibration = next((row for row in analysis["phase_rows"] if row["phase"] == "calibration"), None)
        if calibration:
            print(
                analysis["label"],
                "calibration:",
                calibration.get("duration_seconds"),
                "s",
                "npools:",
                calibration.get("npools"),
                "file span:",
                calibration.get("file_span_seconds"),
                "s",
            )
        if analysis["gap_rows"]:
            biggest = analysis["gap_rows"][0]
            print(
                analysis["label"],
                "largest file gap:",
                biggest["gap_seconds"],
                "s",
                biggest["previous_category"],
                "->",
                biggest["current_category"],
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
