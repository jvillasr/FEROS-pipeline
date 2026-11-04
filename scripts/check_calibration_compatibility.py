from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from astropy.io import fits

from feros_diagnostics_common import ensure_output_dir, write_csv, write_json

SCIENCE_DPR_TYPES = {"OBJECT,WAVE", "VELOC,WAVE", "OBJECT,SKY", "VELOC,SKY"}
REQUIRED_SINGLETONS = [
    "MasterBias.fits",
    "Flat.fits",
    "trace.pkl",
    "P.fits",
    "P_ob.fits",
    "P_co.fits",
    "S_flat_ob.fits",
    "S_flat_co.fits",
    "shifts.pkl",
]
OPTIONAL_SINGLETONS = ["thar_shifts.pdf", "ThAr_Ne_shifts.dat"]
WAVSOL_REQUIRED_KEYS = {
    "p1",
    "p1_co",
    "mjd",
    "All_Pixel_Centers",
    "All_Wavelengths",
    "All_Orders",
    "All_Pixel_Centers_co",
    "All_Wavelengths_co",
    "All_Orders_co",
}
TRACE_REQUIRED_KEYS = {"c_ob", "c_co", "c_all", "nord_ob", "nord_co", "nord_all", "GA_flat", "RO_flat"}


def read_bad_files(raw_dir: Path) -> set[str]:
    path = raw_dir / "bad_files.txt"
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def raw_inventory(raw_dir: Path, lamp: str) -> list[dict[str, Any]]:
    bad_names = read_bad_files(raw_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.fits")):
        if path.name in bad_names:
            continue
        try:
            header = fits.getheader(path)
            status = "ok"
        except Exception as exc:  # noqa: BLE001 - diagnostics should record bad inputs
            rows.append(
                {
                    "raw_file": str(path),
                    "basename": path.name,
                    "status": f"error: {exc!r}",
                    "dpr_type": "",
                    "lamp_id": "",
                    "object": "",
                    "date_obs": "",
                    "mjd_obs": "",
                    "selected_calibration": "no",
                    "selected_science": "no",
                }
            )
            continue
        dpr_type = str(header.get("HIERARCH ESO DPR TYPE", "")).strip()
        lamp_id = str(header.get("HIERARCH ESO INS CALMIRR2 ID", "")).strip()
        rows.append(
            {
                "raw_file": str(path),
                "basename": path.name,
                "status": status,
                "dpr_type": dpr_type,
                "lamp_id": lamp_id,
                "object": str(header.get("OBJECT", "")).strip(),
                "date_obs": str(header.get("DATE-OBS", "")).strip(),
                "mjd_obs": header.get("MJD-OBS", header.get("MJD", "")),
                "selected_calibration": "yes" if dpr_type == "WAVE" and lamp_id == lamp else "no",
                "selected_science": "yes" if dpr_type in SCIENCE_DPR_TYPES else "no",
            }
        )
    return rows


def expected_thar_products(raw_path: Path) -> list[tuple[str, str]]:
    stem = raw_path.stem
    return [
        ("optimal_extraction_object", f"{stem}.spec.ob.fits.S"),
        ("optimal_extraction_comparison", f"{stem}.spec.co.fits.S"),
        ("simple_extraction_object", f"{stem}.spec.simple.ob.fits.S"),
        ("simple_extraction_comparison", f"{stem}.spec.simple.co.fits.S"),
        ("wavelength_solution_pickle", f"{stem}.wavsolpars.pkl"),
        ("wavelength_solution_object", f"{stem}_sp_ob.fits"),
        ("wavelength_solution_comparison", f"{stem}_sp_co.fits"),
    ]


def check_fits(path: Path) -> str:
    try:
        fits.getheader(path)
    except Exception as exc:  # noqa: BLE001 - diagnostics should record bad products
        return f"unreadable: {exc!r}"
    return "readable"


def check_pickle(path: Path) -> tuple[str, Any | None]:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - diagnostics should record bad products
        return f"unreadable: {exc!r}", None
    return "readable", payload


def product_row(category: str, name: str, path: Path, required: bool, detail: str = "") -> dict[str, Any]:
    exists = path.exists()
    readable = ""
    status = "ok"
    if not exists:
        status = "missing_required" if required else "missing_optional"
    elif path.suffix.lower() == ".fits" or ".fits." in path.name:
        readable = check_fits(path)
        if readable != "readable" and required:
            status = "unreadable_required"
    return {
        "category": category,
        "product": name,
        "path": str(path),
        "required": "yes" if required else "no",
        "exists": "yes" if exists else "no",
        "readable": readable,
        "status": status,
        "size_bytes": path.stat().st_size if exists else "",
        "detail": detail,
    }


def check_singletons(calibration_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for name in REQUIRED_SINGLETONS:
        path = calibration_dir / name
        row = product_row("singleton", name, path, required=True)
        if path.exists() and name.endswith(".pkl"):
            readable, payload = check_pickle(path)
            row["readable"] = readable
            if readable != "readable":
                row["status"] = "unreadable_required"
            elif name == "trace.pkl":
                keys = set(payload.keys()) if isinstance(payload, dict) else set()
                missing = sorted(TRACE_REQUIRED_KEYS - keys)
                row["detail"] = f"missing_keys={missing}" if missing else "keys_ok"
                if missing:
                    row["status"] = "invalid_required"
            elif name == "shifts.pkl":
                if not isinstance(payload, dict):
                    row["status"] = "invalid_required"
                    row["detail"] = "payload is not a dict"
                else:
                    details["shifts_payload"] = payload
                    row["detail"] = f"keys={sorted(payload.keys())}"
        rows.append(row)
    for name in OPTIONAL_SINGLETONS:
        rows.append(product_row("optional_singleton", name, calibration_dir / name, required=False))
    return rows, details


def check_thar_products(calibration_dir: Path, thar_raw_files: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in thar_raw_files:
        for category, filename in expected_thar_products(raw_path):
            path = calibration_dir / filename
            row = product_row(category, filename, path, required=True, detail=f"raw={raw_path}")
            if path.exists() and category == "wavelength_solution_pickle":
                readable, payload = check_pickle(path)
                row["readable"] = readable
                if readable != "readable":
                    row["status"] = "unreadable_required"
                elif not isinstance(payload, dict):
                    row["status"] = "invalid_required"
                    row["detail"] = f"raw={raw_path}; payload is not a dict"
                else:
                    keys = set(payload.keys())
                    missing = sorted(WAVSOL_REQUIRED_KEYS - keys)
                    row["detail"] = f"raw={raw_path}; missing_keys={missing}" if missing else f"raw={raw_path}; keys_ok"
                    if missing:
                        row["status"] = "invalid_required"
            rows.append(row)
    return rows


def shifts_summary(shifts_payload: Any | None, thar_raw_files: list[Path]) -> dict[str, Any]:
    expected_exact = [str(path) for path in thar_raw_files]
    expected_basenames = [path.name for path in thar_raw_files]
    if not isinstance(shifts_payload, dict):
        return {
            "status": "missing_or_invalid",
            "names_count": 0,
            "vals_count": 0,
            "missing_exact_count": len(expected_exact),
            "missing_basename_count": len(expected_basenames),
            "missing_exact": expected_exact,
            "missing_basenames": expected_basenames,
            "selected_reference": "",
            "selected_reference_basename": "",
        }
    names = [str(value) for value in shifts_payload.get("names", [])]
    vals = list(shifts_payload.get("vals", []))
    names_set = set(names)
    basename_set = {Path(name).name for name in names}
    missing_exact = [name for name in expected_exact if name not in names_set]
    missing_basenames = [name for name in expected_basenames if name not in basename_set]
    selected_reference = ""
    if vals and names and len(vals) == len(names):
        selected_index = min(range(len(vals)), key=lambda index: vals[index])
        selected_reference = names[selected_index]
    status = "ok" if not missing_exact and len(vals) == len(names) and len(names) >= len(expected_exact) else "incompatible"
    return {
        "status": status,
        "names_count": len(names),
        "vals_count": len(vals),
        "missing_exact_count": len(missing_exact),
        "missing_basename_count": len(missing_basenames),
        "missing_exact": missing_exact,
        "missing_basenames": missing_basenames,
        "selected_reference": selected_reference,
        "selected_reference_basename": Path(selected_reference).name if selected_reference else "",
    }


def summarise(raw_rows: list[dict[str, Any]], product_rows: list[dict[str, Any]], shift_info: dict[str, Any]) -> dict[str, Any]:
    raw_counts = Counter(str(row["dpr_type"]) for row in raw_rows if row.get("status") == "ok")
    selected_calibration_count = sum(1 for row in raw_rows if row.get("selected_calibration") == "yes")
    selected_science_count = sum(1 for row in raw_rows if row.get("selected_science") == "yes")
    product_status_counts = Counter(str(row["status"]) for row in product_rows)
    required_bad = [row for row in product_rows if row["required"] == "yes" and row["status"] != "ok"]
    compatible = not required_bad and shift_info.get("status") == "ok" and selected_calibration_count > 0
    return {
        "status": "compatible" if compatible else "incompatible",
        "raw_dpr_type_counts": dict(sorted(raw_counts.items())),
        "selected_calibration_count": selected_calibration_count,
        "selected_science_count": selected_science_count,
        "product_status_counts": dict(sorted(product_status_counts.items())),
        "bad_required_product_count": len(required_bad),
        "bad_required_products": required_bad[:50],
        "shift_compatibility": shift_info,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether FEROS calibration products are complete and compatible with a raw night.")
    parser.add_argument("raw_dir", type=Path, help="Raw FEROS 4k input directory.")
    parser.add_argument("calibration_dir", type=Path, help="Calibration-product directory to check.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Diagnostics output directory.")
    parser.add_argument("--lamp", default="LAMP3", help="Calibration lamp ID to match CERES FileClassify behaviour.")
    parser.add_argument("--label", default="calibration", help="Label stored in the summary JSON.")
    args = parser.parse_args()

    raw_dir = args.raw_dir.resolve()
    calibration_dir = args.calibration_dir.resolve()
    output_dir = ensure_output_dir(args.output_dir.resolve())
    if not raw_dir.is_dir():
        raise SystemExit(f"raw directory does not exist: {raw_dir}")
    if not calibration_dir.is_dir():
        raise SystemExit(f"calibration directory does not exist: {calibration_dir}")

    raw_rows = raw_inventory(raw_dir, args.lamp)
    thar_raw_files = [Path(str(row["raw_file"])) for row in raw_rows if row.get("selected_calibration") == "yes" and row.get("status") == "ok"]
    singleton_rows, singleton_details = check_singletons(calibration_dir)
    product_rows = singleton_rows + check_thar_products(calibration_dir, thar_raw_files)
    shift_info = shifts_summary(singleton_details.get("shifts_payload"), thar_raw_files)
    summary = {
        "label": args.label,
        "raw_dir": str(raw_dir),
        "calibration_dir": str(calibration_dir),
        "output_dir": str(output_dir),
        "lamp": args.lamp,
        **summarise(raw_rows, product_rows, shift_info),
    }

    write_csv(
        output_dir / "raw_inventory.csv",
        raw_rows,
        [
            "raw_file",
            "basename",
            "status",
            "dpr_type",
            "lamp_id",
            "object",
            "date_obs",
            "mjd_obs",
            "selected_calibration",
            "selected_science",
        ],
    )
    write_csv(
        output_dir / "calibration_product_checks.csv",
        product_rows,
        ["category", "product", "path", "required", "exists", "readable", "status", "size_bytes", "detail"],
    )
    write_json(output_dir / "calibration_compatibility_summary.json", summary)

    print(f"Wrote calibration compatibility diagnostics to {output_dir}")
    print("status:", summary["status"])
    print("selected calibration frames:", summary["selected_calibration_count"])
    print("selected science frames:", summary["selected_science_count"])
    print("bad required products:", summary["bad_required_product_count"])
    print("shift status:", summary["shift_compatibility"]["status"])
    return 0 if summary["status"] == "compatible" else 2


if __name__ == "__main__":
    raise SystemExit(main())
