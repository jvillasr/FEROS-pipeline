from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from feros_diagnostics_common import (
    NUMERIC_RESULT_COLUMNS,
    ensure_output_dir,
    finite_stats,
    list_product_files,
    parse_results,
    sha256_file,
    write_csv,
    write_json,
)

REQUIRED_HDUS = [
    "WAVELENGTH",
    "EXTRACTED FLUX",
    "BLAZE CORRECTED FLUX",
    "CONTINUUM NORMALIZED FLUX",
    "WAVELENGTH GRID",
    "MERGED SPECTRUM",
]


def summarise_inventory(proc_dir: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind, paths in list_product_files(proc_dir).items():
        for path in paths:
            rows.append(
                {
                    "label": label,
                    "kind": kind,
                    "file": path.name,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path) if path.is_file() and path.stat().st_size < 1024 * 1024 * 1024 else "",
                }
            )
    return rows


def summarise_fits(proc_dir: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(proc_dir.glob("*_sp.fits")):
        try:
            with fits.open(path, memmap=False) as hdul:
                names = [hdu.name.upper() for hdu in hdul]
                missing = [name for name in REQUIRED_HDUS if name not in names]
                rows.append(
                    {
                        "label": label,
                        "file": path.name,
                        "hdu": "PRIMARY",
                        "readable": True,
                        "missing_required_hdus": ";".join(missing),
                        "shape": "",
                        "n_total": "",
                        "n_finite": "",
                        "finite_fraction": "",
                        "min": "",
                        "max": "",
                        "median": "",
                    }
                )
                for hdu in hdul[1:]:
                    data = hdu.data
                    if data is None:
                        continue
                    stats = finite_stats(np.asarray(data))
                    rows.append(
                        {
                            "label": label,
                            "file": path.name,
                            "hdu": hdu.name,
                            "readable": True,
                            "missing_required_hdus": "",
                            "shape": "x".join(str(part) for part in data.shape),
                            **stats,
                        }
                    )
        except Exception as exc:  # noqa: BLE001 - diagnostics should record all FITS failures
            rows.append(
                {
                    "label": label,
                    "file": path.name,
                    "hdu": "",
                    "readable": False,
                    "missing_required_hdus": "",
                    "shape": "",
                    "n_total": "",
                    "n_finite": "",
                    "finite_fraction": "",
                    "min": "",
                    "max": "",
                    "median": "",
                    "error": repr(exc),
                }
            )
    return rows


def comparison_status(missing: list[str], extra: list[str], allow_private_subset: bool) -> str:
    if not missing and not extra:
        return "match"
    if allow_private_subset and missing and not extra:
        return "subset"
    return "difference"


def compare_file_sets(private_proc: Path, reference_proc: Path, allow_private_subset: bool) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    private_products = list_product_files(private_proc)
    reference_products = list_product_files(reference_proc)
    for kind in sorted(private_products):
        private_names = {path.name for path in private_products[kind]}
        reference_names = {path.name for path in reference_products[kind]}
        missing = sorted(reference_names - private_names)
        extra = sorted(private_names - reference_names)
        checks.append(
            {
                "check": f"{kind}_filenames",
                "private_count": len(private_names),
                "reference_count": len(reference_names),
                "status": comparison_status(missing, extra, allow_private_subset),
                "missing_in_private": ";".join(missing),
                "extra_in_private": ";".join(extra),
            }
        )
    return checks


def compare_results(private_rows: list[dict[str, Any]], reference_rows: list[dict[str, Any]], allow_private_subset: bool) -> list[dict[str, Any]]:
    private_by_key = {row.get("ccf_pdf_basename") or f"row-{row['row_index']}": row for row in private_rows}
    reference_by_key = {row.get("ccf_pdf_basename") or f"row-{row['row_index']}": row for row in reference_rows}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(private_by_key) | set(reference_by_key)):
        private_row = private_by_key.get(key)
        reference_row = reference_by_key.get(key)
        if private_row is None or reference_row is None:
            rows.append(
                {
                    "result_key": key,
                    "column": "",
                    "status": (
                        "missing_in_private_expected"
                        if private_row is None and allow_private_subset
                        else ("missing_in_private" if private_row is None else "extra_in_private")
                    ),
                    "private": "",
                    "reference": "",
                    "abs_difference": "",
                }
            )
            continue
        for column in NUMERIC_RESULT_COLUMNS:
            private_value = private_row.get(column)
            reference_value = reference_row.get(column)
            if private_value is None or reference_value is None:
                status = "missing_numeric"
                difference = ""
            else:
                difference = abs(float(private_value) - float(reference_value))
                status = "match" if difference == 0 else "difference"
            rows.append(
                {
                    "result_key": key,
                    "target": private_row.get("target", ""),
                    "column": column,
                    "status": status,
                    "private": private_value,
                    "reference": reference_value,
                    "abs_difference": difference,
                }
            )
    return rows


def row_count_status(private_count: int, reference_count: int, allow_private_subset: bool) -> str:
    if private_count == reference_count:
        return "match"
    if allow_private_subset and private_count < reference_count:
        return "subset"
    return "difference"


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a private FEROS reduction against a reference reduction.")
    parser.add_argument("--private-proc", required=True, type=Path, help="Private reduction proc directory.")
    parser.add_argument("--reference-proc", required=True, type=Path, help="Reference/canonical proc directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Private diagnostics output directory.")
    parser.add_argument("--allow-private-subset", action="store_true", help="Treat products missing from the private proc as expected when the private run is a planned subset of the reference.")
    args = parser.parse_args()

    private_proc = args.private_proc.resolve()
    reference_proc = args.reference_proc.resolve()
    output_dir = ensure_output_dir(args.output_dir.resolve())
    if not private_proc.is_dir():
        raise SystemExit(f"private proc directory does not exist: {private_proc}")
    if not reference_proc.is_dir():
        raise SystemExit(f"reference proc directory does not exist: {reference_proc}")

    inventory_rows = summarise_inventory(private_proc, "private") + summarise_inventory(reference_proc, "reference")
    fits_rows = summarise_fits(private_proc, "private") + summarise_fits(reference_proc, "reference")
    private_results = parse_results(private_proc / "results.txt", "private")
    reference_results = parse_results(reference_proc / "results.txt", "reference")
    result_diffs = compare_results(private_results, reference_results, args.allow_private_subset)
    checks = compare_file_sets(private_proc, reference_proc, args.allow_private_subset)
    checks.append(
        {
            "check": "results_row_count",
            "private_count": len(private_results),
            "reference_count": len(reference_results),
            "status": row_count_status(len(private_results), len(reference_results), args.allow_private_subset),
            "missing_in_private": "",
            "extra_in_private": "",
        }
    )
    result_status_counts = Counter(str(row.get("status", "")) for row in result_diffs)

    write_csv(
        output_dir / "product_inventory.csv",
        inventory_rows,
        ["label", "kind", "file", "path", "size_bytes", "sha256"],
    )
    write_csv(
        output_dir / "fits_summary.csv",
        fits_rows,
        [
            "label",
            "file",
            "hdu",
            "readable",
            "missing_required_hdus",
            "shape",
            "n_total",
            "n_finite",
            "finite_fraction",
            "min",
            "max",
            "median",
            "error",
        ],
    )
    result_fieldnames = ["label", "row_index", "parse_error", *NUMERIC_RESULT_COLUMNS, "target", "instrument", "pipeline", "ccf_pdf_basename", "ccf_pdf", "raw_line"]
    write_csv(output_dir / "results_rows.csv", private_results + reference_results, result_fieldnames)
    write_csv(
        output_dir / "results_numeric_diffs.csv",
        result_diffs,
        ["result_key", "target", "column", "status", "private", "reference", "abs_difference"],
    )
    write_csv(
        output_dir / "comparison_checks.csv",
        checks,
        ["check", "status", "private_count", "reference_count", "missing_in_private", "extra_in_private"],
    )

    summary = {
        "private_proc": str(private_proc),
        "reference_proc": str(reference_proc),
        "output_dir": str(output_dir),
        "allow_private_subset": args.allow_private_subset,
        "checks": checks,
        "private_results_rows": len(private_results),
        "reference_results_rows": len(reference_results),
        "shared_result_rows_compared": sum(
            1 for row in result_diffs if row.get("status") in {"match", "difference", "missing_numeric"}
        ),
        "result_diff_status_counts": dict(sorted(result_status_counts.items())),
        "fits_read_errors": [row for row in fits_rows if row.get("readable") is False],
    }
    write_json(output_dir / "summary.json", summary)
    print(f"Wrote comparison diagnostics to {output_dir}")
    for check in checks:
        print(f"{check['check']}: {check['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
