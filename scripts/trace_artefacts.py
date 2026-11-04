from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from feros_diagnostics_common import ensure_output_dir, hdu_by_name, write_csv, write_json

ORDER_STAGES = [
    ("extracted_flux", "EXTRACTED FLUX"),
    ("blaze_corrected_flux", "BLAZE CORRECTED FLUX"),
    ("continuum_normalised_flux", "CONTINUUM NORMALIZED FLUX"),
    (
        "derivative_weighted_flux",
        "CONTINUMM NORMALIZED FLUX MULTIPLIED BY THE DERIVATIVE OF THE WAVELENGTH WITH RESPECT TO THE PIXELS",
    ),
]

MERGED_STAGES = [
    ("merged_spectrum", "MERGED SPECTRUM"),
]


def object_name_from_header(header: fits.Header, fallback: str) -> str:
    return str(header.get("OBJECT") or header.get("HIERARCH TARGET NAME") or fallback)


def window_stats(wavelengths: np.ndarray, values: np.ndarray, centre: float, half_width: float) -> dict[str, Any]:
    wavelengths = np.asarray(wavelengths)
    values = np.asarray(values)
    mask = np.isfinite(wavelengths) & (wavelengths >= centre - half_width) & (wavelengths <= centre + half_width)
    n_pixels = int(mask.sum())
    if n_pixels == 0:
        return {
            "n_pixels": 0,
            "n_finite": 0,
            "wavelength_min": None,
            "wavelength_max": None,
            "value_min": None,
            "value_max": None,
            "value_median": None,
            "wavelength_at_min": None,
            "pixel_at_min": None,
        }
    selected_wavelengths = wavelengths[mask]
    selected_values = values[mask]
    finite = np.isfinite(selected_values)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {
            "n_pixels": n_pixels,
            "n_finite": 0,
            "wavelength_min": float(np.nanmin(selected_wavelengths)),
            "wavelength_max": float(np.nanmax(selected_wavelengths)),
            "value_min": None,
            "value_max": None,
            "value_median": None,
            "wavelength_at_min": None,
            "pixel_at_min": None,
        }
    finite_values = selected_values[finite]
    finite_wavelengths = selected_wavelengths[finite]
    local_min_index = int(np.nanargmin(finite_values))
    original_indices = np.where(mask)[0][finite]
    return {
        "n_pixels": n_pixels,
        "n_finite": n_finite,
        "wavelength_min": float(np.nanmin(selected_wavelengths)),
        "wavelength_max": float(np.nanmax(selected_wavelengths)),
        "value_min": float(np.nanmin(finite_values)),
        "value_max": float(np.nanmax(finite_values)),
        "value_median": float(np.nanmedian(finite_values)),
        "wavelength_at_min": float(finite_wavelengths[local_min_index]),
        "pixel_at_min": int(original_indices[local_min_index]),
    }


def trace_file(path: Path, label: str, centres: list[float], half_width: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with fits.open(path, memmap=False) as hdul:
            object_name = object_name_from_header(hdul[0].header, path.stem)
            order_wavelengths = np.asarray(hdu_by_name(hdul, "WAVELENGTH").data)
            for stage_name, hdu_name in ORDER_STAGES:
                stage_data = np.asarray(hdu_by_name(hdul, hdu_name).data)
                for centre in centres:
                    for order_index in range(order_wavelengths.shape[0]):
                        stats = window_stats(order_wavelengths[order_index], stage_data[order_index], centre, half_width)
                        rows.append(
                            {
                                "label": label,
                                "file": path.name,
                                "object": object_name,
                                "centre": centre,
                                "half_width": half_width,
                                "stage": stage_name,
                                "hdu": hdu_name,
                                "order": order_index,
                                "status": "ok",
                                **stats,
                            }
                        )
            merged_wavelengths = np.asarray(hdu_by_name(hdul, "WAVELENGTH GRID").data)
            for stage_name, hdu_name in MERGED_STAGES:
                stage_data = np.asarray(hdu_by_name(hdul, hdu_name).data)
                for centre in centres:
                    stats = window_stats(merged_wavelengths, stage_data, centre, half_width)
                    rows.append(
                        {
                            "label": label,
                            "file": path.name,
                            "object": object_name,
                            "centre": centre,
                            "half_width": half_width,
                            "stage": stage_name,
                            "hdu": hdu_name,
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
                "centre": "",
                "half_width": half_width,
                "stage": "",
                "hdu": "",
                "order": "",
                "status": f"error: {exc!r}",
                "n_pixels": "",
                "n_finite": "",
                "wavelength_min": "",
                "wavelength_max": "",
                "value_min": "",
                "value_max": "",
                "value_median": "",
                "wavelength_at_min": "",
                "pixel_at_min": "",
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Trace fixed-wavelength FEROS artefact windows through reduced FITS stages.")
    parser.add_argument("--proc-dir", required=True, type=Path, help="Reduction proc directory containing *_sp.fits files.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Private diagnostics output directory.")
    parser.add_argument("--label", default="reduction", help="Label to store in diagnostic rows.")
    parser.add_argument("--wavelengths", nargs="+", type=float, default=[4200.0, 4280.0, 4300.0])
    parser.add_argument("--half-width", type=float, default=2.0, help="Half-width in Angstrom around each centre.")
    args = parser.parse_args()

    proc_dir = args.proc_dir.resolve()
    output_dir = ensure_output_dir(args.output_dir.resolve())
    if not proc_dir.is_dir():
        raise SystemExit(f"proc directory does not exist: {proc_dir}")

    rows: list[dict[str, Any]] = []
    for path in sorted(proc_dir.glob("*_sp.fits")):
        rows.extend(trace_file(path, args.label, args.wavelengths, args.half_width))

    fieldnames = [
        "label",
        "file",
        "object",
        "centre",
        "half_width",
        "stage",
        "hdu",
        "order",
        "status",
        "n_pixels",
        "n_finite",
        "wavelength_min",
        "wavelength_max",
        "value_min",
        "value_max",
        "value_median",
        "wavelength_at_min",
        "pixel_at_min",
    ]
    write_csv(output_dir / "artefact_trace.csv", rows, fieldnames)
    summary = {
        "proc_dir": str(proc_dir),
        "output_dir": str(output_dir),
        "label": args.label,
        "wavelengths": args.wavelengths,
        "half_width": args.half_width,
        "n_rows": len(rows),
        "n_files": len({row["file"] for row in rows if row.get("status") == "ok"}),
        "error_rows": [row for row in rows if row.get("status") != "ok"],
    }
    write_json(output_dir / "artefact_trace_summary.json", summary)
    print(f"Wrote artefact trace diagnostics to {output_dir}")
    print(f"rows: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
