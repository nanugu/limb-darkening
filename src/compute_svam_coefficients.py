#!/usr/bin/env python3
"""Add SVAM limb-darkening coefficients to the target catalog.

This script uses the target metadata/catalog as input. It keeps all existing
columns unchanged and computes coefficients by fitting analytic laws to
synthetic CHARA-sampled visibilities generated from the full model CLV.

Intensity-domain columns already present in the input remain untouched, e.g.
    p1__power1__stagger__MIRCX
    p1_1p__power1__stagger__MYSTIC

New SVAM columns added by this script include:
    alpha_svam_H_Stagger
    alpha_svam_K_Stagger
    p1_svam_H_Stagger
    alpha1_svam_K_Stagger
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from chara_fit_common import resolve_oifits_path
from svam_recovery import as_float, fit_svam_target, is_finite
from public_schema import base_metadata_from_row, clean_grid

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"

DEFAULT_MERGED_CSV = CSV_DIR / "merged_four_branches_wide.csv"
DEFAULT_OIFITS_DIR = os.environ.get("OIFITS_DIR", str(ROOT / "oifits"))
DEFAULT_DATA_DIR = os.environ.get("EXOTIC_LD_DATA_DIR", str(ROOT / "exotic_ld_data"))
DEFAULT_OUTDIR = CSV_DIR
DEFAULT_GRIDS = ("stagger", "kurucz", "mps1", "mps2")
DEFAULT_LAWS = ("power1", "power2")
OUTPUT_CSV = "compute_svam_coefficients.csv"

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H")


def _set_power1_columns(row: dict, fit: dict, grid: str) -> None:
    grid_name = clean_grid(grid)
    row[f"alpha_svam_H_{grid_name}"] = fit["H_PL_alpha_fit"]
    row[f"alpha_svam_H_{grid_name}_err"] = fit["H_PL_alpha_fit_err"]
    row[f"alpha_svam_K_{grid_name}"] = fit["K_PL_alpha_fit"]
    row[f"alpha_svam_K_{grid_name}_err"] = fit["K_PL_alpha_fit_err"]


def _set_power2_columns(row: dict, fit: dict, grid: str) -> None:
    grid_name = clean_grid(grid)
    row[f"p1_svam_H_{grid_name}"] = fit["H_PL_alpha_fit"]
    row[f"p1_svam_H_{grid_name}_err"] = fit["H_PL_alpha_fit_err"]
    row[f"p1_svam_K_{grid_name}"] = fit["K_PL_alpha_fit"]
    row[f"p1_svam_K_{grid_name}_err"] = fit["K_PL_alpha_fit_err"]
    row[f"alpha1_svam_H_{grid_name}"] = fit["H_PL2_alpha_fit"]
    row[f"alpha1_svam_H_{grid_name}_err"] = fit["H_PL2_alpha_fit_err"]
    row[f"alpha1_svam_K_{grid_name}"] = fit["K_PL2_alpha_fit"]
    row[f"alpha1_svam_K_{grid_name}_err"] = fit["K_PL2_alpha_fit_err"]


def _blank_power1_columns(row: dict, grid: str) -> None:
    grid_name = clean_grid(grid)
    for col in (f"alpha_svam_H_{grid_name}", f"alpha_svam_H_{grid_name}_err", f"alpha_svam_K_{grid_name}", f"alpha_svam_K_{grid_name}_err"):
        row[col] = np.nan


def _blank_power2_columns(row: dict, grid: str) -> None:
    grid_name = clean_grid(grid)
    for col in (
        f"p1_svam_H_{grid_name}", f"p1_svam_H_{grid_name}_err",
        f"p1_svam_K_{grid_name}", f"p1_svam_K_{grid_name}_err",
        f"alpha1_svam_H_{grid_name}", f"alpha1_svam_H_{grid_name}_err",
        f"alpha1_svam_K_{grid_name}", f"alpha1_svam_K_{grid_name}_err",
    ):
        row[col] = np.nan


def build_catalog(args: argparse.Namespace) -> Path:
    merged = pd.read_csv(args.merged_csv)
    out_rows: list[dict] = []

    for idx, src_row in merged.iterrows():
        row = base_metadata_from_row(src_row)
        target_name = str(src_row.get("target") or src_row.get("Target_chara") or src_row.get("Target") or idx)
        mircx_path = resolve_oifits_path(src_row.get("oifits_h") or src_row.get("MIRCX_file_first"), args.oifits_dir, require_exists=True, mirc_aliases=True)
        mystic_path = resolve_oifits_path(src_row.get("oifits_k") or src_row.get("MYSTIC_file_first"), args.oifits_dir, require_exists=True, mirc_aliases=True)
        diam_input = as_float(src_row.get("theta_PL") if "theta_PL" in src_row else src_row.get("PL_diam", src_row.get("theta_init")))
        teff = as_float(src_row.get("teff", src_row.get("Teff")))
        logg = as_float(src_row.get("logg"))
        feh = as_float(src_row.get("feh", src_row.get("FeH")))

        if not (mircx_path and mystic_path and is_finite(diam_input) and is_finite(teff) and is_finite(logg) and is_finite(feh)):
            for grid in args.grids:
                if "power1" in args.laws:
                    _blank_power1_columns(row, grid)
                if "power2" in args.laws:
                    _blank_power2_columns(row, grid)
            out_rows.append(row)
            print(f"[SKIP] {target_name}: missing files or Teff/logg/FeH/PL_diam")
            continue

        for grid in args.grids:
            if "power1" in args.laws:
                try:
                    fit = fit_svam_target(
                        target_name,
                        mircx_path,
                        mystic_path,
                        diam_input=diam_input,
                        teff=teff,
                        logg=logg,
                        feh=feh,
                        grid=grid,
                        law="power1",
                        data_dir=args.exotic_data_dir,
                        min_rel_v2=args.min_rel_v2,
                        maxfev=args.maxfev,
                        fig_dir=None,
                    )
                    _set_power1_columns(row, fit, grid)
                except Exception as exc:
                    _blank_power1_columns(row, grid)
                    print(f"[FAIL] {target_name} power1 {grid}: {type(exc).__name__}: {exc}")

            if "power2" in args.laws:
                try:
                    fit = fit_svam_target(
                        target_name,
                        mircx_path,
                        mystic_path,
                        diam_input=diam_input,
                        teff=teff,
                        logg=logg,
                        feh=feh,
                        grid=grid,
                        law="power2",
                        data_dir=args.exotic_data_dir,
                        min_rel_v2=args.min_rel_v2,
                        maxfev=args.maxfev,
                        fig_dir=None,
                    )
                    _set_power2_columns(row, fit, grid)
                except Exception as exc:
                    _blank_power2_columns(row, grid)
                    print(f"[FAIL] {target_name} power2 {grid}: {type(exc).__name__}: {exc}")

        out_rows.append(row)

    out = pd.DataFrame(out_rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / OUTPUT_CSV
    out.to_csv(out_path, index=False)
    print(f"[saved] {out_path} rows={len(out)}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-csv", default=DEFAULT_MERGED_CSV)
    parser.add_argument("--oifits-dir", default=DEFAULT_OIFITS_DIR)
    parser.add_argument("--exotic-data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--grids", nargs="+", default=list(DEFAULT_GRIDS), choices=list(DEFAULT_GRIDS))
    parser.add_argument("--laws", nargs="+", default=list(DEFAULT_LAWS), choices=list(DEFAULT_LAWS))
    parser.add_argument("--min-rel-v2", type=float, default=0.0)
    parser.add_argument("--maxfev", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_catalog(args)


if __name__ == "__main__":
    main()
