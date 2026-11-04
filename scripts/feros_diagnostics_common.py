from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from astropy.io import fits


RESULT_COLUMNS = [
    "target",
    "bjd",
    "rv",
    "rv_err",
    "bisector_span",
    "bisector_span_err",
    "instrument",
    "pipeline",
    "resolution",
    "teff",
    "logg",
    "metallicity",
    "vsini",
    "xc_min",
    "dispersion",
    "exptime",
    "snr_5130",
    "ccf_pdf",
]

NUMERIC_RESULT_COLUMNS = [
    "bjd",
    "rv",
    "rv_err",
    "bisector_span",
    "bisector_span_err",
    "resolution",
    "teff",
    "logg",
    "metallicity",
    "vsini",
    "xc_min",
    "dispersion",
    "exptime",
    "snr_5130",
]


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> int:
    count = 0
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def parse_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_results(path: Path, label: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for row_index, line in enumerate(path.read_text().splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < len(RESULT_COLUMNS):
            rows.append(
                {
                    "label": label,
                    "row_index": row_index,
                    "parse_error": f"expected {len(RESULT_COLUMNS)} columns, found {len(parts)}",
                    "raw_line": stripped,
                }
            )
            continue
        values = parts[: len(RESULT_COLUMNS) - 1] + [" ".join(parts[len(RESULT_COLUMNS) - 1 :])]
        row = dict(zip(RESULT_COLUMNS, values))
        row["label"] = label
        row["row_index"] = row_index
        row["ccf_pdf_basename"] = Path(str(row["ccf_pdf"])).name
        row["parse_error"] = ""
        row["raw_line"] = stripped
        for column in NUMERIC_RESULT_COLUMNS:
            row[column] = parse_float(str(row[column]))
        rows.append(row)
    return rows


def hdu_by_name(hdul: fits.HDUList, name: str):
    wanted = name.upper()
    for hdu in hdul:
        if hdu.name.upper() == wanted:
            return hdu
    raise KeyError(name)


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values)
    finite = np.isfinite(array)
    n_total = int(array.size)
    n_finite = int(finite.sum())
    if n_finite == 0:
        return {
            "n_total": n_total,
            "n_finite": 0,
            "finite_fraction": 0.0 if n_total else None,
            "min": None,
            "max": None,
            "median": None,
        }
    finite_values = array[finite]
    return {
        "n_total": n_total,
        "n_finite": n_finite,
        "finite_fraction": n_finite / n_total if n_total else None,
        "min": float(np.nanmin(finite_values)),
        "max": float(np.nanmax(finite_values)),
        "median": float(np.nanmedian(finite_values)),
    }


def wavelength_stats(wavelengths: np.ndarray) -> dict[str, Any]:
    stats = finite_stats(wavelengths)
    array = np.asarray(wavelengths)
    finite = np.isfinite(array) & (array > 0)
    if int(finite.sum()) == 0:
        stats.update({"n_positive": 0, "wavelength_min": None, "wavelength_max": None, "median_step": None})
        return stats
    finite_wavelengths = array[finite]
    diffs = np.diff(finite_wavelengths)
    diffs = diffs[np.isfinite(diffs)]
    stats.update(
        {
            "n_positive": int(finite.sum()),
            "wavelength_min": float(np.nanmin(finite_wavelengths)),
            "wavelength_max": float(np.nanmax(finite_wavelengths)),
            "median_step": float(np.nanmedian(diffs)) if diffs.size else None,
        }
    )
    return stats


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_product_files(proc_dir: Path) -> dict[str, list[Path]]:
    return {
        "sp_fits": sorted(proc_dir.glob("*_sp.fits")),
        "ccf_pdf": sorted(proc_dir.glob("*_XCs_*.pdf")),
        "results": [proc_dir / "results.txt"] if (proc_dir / "results.txt").exists() else [],
    }
