from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from feros_diagnostics_common import ensure_output_dir, hdu_by_name, wavelength_stats, write_csv, write_json


def object_name_from_header(header: fits.Header, fallback: str) -> str:
    return str(header.get("OBJECT") or header.get("HIERARCH TARGET NAME") or fallback)


def coverage_rows(proc_dir: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(proc_dir.glob("*_sp.fits")):
        try:
            with fits.open(path, memmap=False) as hdul:
                object_name = object_name_from_header(hdul[0].header, path.stem)
                wavelength = np.asarray(hdu_by_name(hdul, "WAVELENGTH").data)
                for order_index in range(wavelength.shape[0]):
                    stats = wavelength_stats(wavelength[order_index])
                    rows.append(
                        {
                            "label": label,
                            "file": path.name,
                            "object": object_name,
                            "hdu": "WAVELENGTH",
                            "order": order_index,
                            "status": "ok",
                            **stats,
                        }
                    )
                merged_wavelength = np.asarray(hdu_by_name(hdul, "WAVELENGTH GRID").data)
                stats = wavelength_stats(merged_wavelength)
                rows.append(
                    {
                        "label": label,
                        "file": path.name,
                        "object": object_name,
                        "hdu": "WAVELENGTH GRID",
                        "order": "merged",
                        "status": "ok",
                        **stats,
                    }
                )
        except Exception as exc:  # noqa: BLE001 - diagnostics should record bad products
            rows.append(
                {
                    "label": label,
                    "file": path.name,
                    "object": path.stem,
                    "hdu": "",
                    "order": "",
                    "status": f"error: {exc!r}",
                    "n_total": "",
                    "n_finite": "",
                    "finite_fraction": "",
                    "min": "",
                    "max": "",
                    "median": "",
                    "n_positive": "",
                    "wavelength_min": "",
                    "wavelength_max": "",
                    "median_step": "",
                }
            )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    good_rows = [row for row in rows if row.get("status") == "ok"]
    per_order = [row for row in good_rows if row.get("hdu") == "WAVELENGTH"]
    merged = [row for row in good_rows if row.get("hdu") == "WAVELENGTH GRID"]
    return {
        "n_files": len({row["file"] for row in good_rows}),
        "n_per_order_rows": len(per_order),
        "n_merged_rows": len(merged),
        "per_order_wavelength_min": min((row["wavelength_min"] for row in per_order if row["wavelength_min"] is not None), default=None),
        "per_order_wavelength_max": max((row["wavelength_max"] for row in per_order if row["wavelength_max"] is not None), default=None),
        "merged_wavelength_min": min((row["wavelength_min"] for row in merged if row["wavelength_min"] is not None), default=None),
        "merged_wavelength_max": max((row["wavelength_max"] for row in merged if row["wavelength_max"] is not None), default=None),
        "error_rows": [row for row in rows if row.get("status") != "ok"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise FEROS wavelength coverage by order and merged spectrum.")
    parser.add_argument("--proc-dir", required=True, type=Path, help="Reduction proc directory containing *_sp.fits files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Private diagnostics output directory.")
    parser.add_argument("--label", default="reduction", help="Label to store in diagnostic rows.")
    args = parser.parse_args()

    proc_dir = args.proc_dir.resolve()
    output_dir = ensure_output_dir(args.output_dir.resolve())
    if not proc_dir.is_dir():
        raise SystemExit(f"proc directory does not exist: {proc_dir}")

    rows = coverage_rows(proc_dir, args.label)
    fieldnames = [
        "label",
        "file",
        "object",
        "hdu",
        "order",
        "status",
        "n_total",
        "n_finite",
        "finite_fraction",
        "min",
        "max",
        "median",
        "n_positive",
        "wavelength_min",
        "wavelength_max",
        "median_step",
    ]
    write_csv(output_dir / "wavelength_coverage.csv", rows, fieldnames)
    summary = {
        "proc_dir": str(proc_dir),
        "output_dir": str(output_dir),
        "label": args.label,
        **summarise(rows),
    }
    write_json(output_dir / "wavelength_summary.json", summary)
    print(f"Wrote wavelength diagnostics to {output_dir}")
    print(
        "per-order range:",
        summary["per_order_wavelength_min"],
        summary["per_order_wavelength_max"],
        "merged range:",
        summary["merged_wavelength_min"],
        summary["merged_wavelength_max"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
