#!/usr/bin/env python3
"""Merge public branch-level CSV products into one wide per-target table."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from chara_fit_common import parse_l2_filename
from public_schema import normalize_target

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"

DEFAULT_METADATA = CSV_DIR / "target_metadata.csv"
DEFAULT_VISIBILITY = CSV_DIR / "fit_visibility_laws.csv"
DEFAULT_EXOTIC = CSV_DIR / "compute_exotic_coefficients.csv"
DEFAULT_SVAM = CSV_DIR / "compute_svam_coefficients.csv"
DEFAULT_DIRECT_CLV = CSV_DIR / "fit_diameters_with_direct_CLV.csv"
DEFAULT_OUTPUT = CSV_DIR / "merged_four_branches_wide.csv"

KEY_COLUMNS = {"target", "target_norm", "hd", "date", "oifits_h", "oifits_k"}


def _read_optional(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        print(f"[merge] {label}: missing {path}; skipping")
        return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[merge] {label}: {path} rows={len(df)} cols={len(df.columns)}")
    return df


def _ensure_key(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "target_norm" not in df.columns:
        parsed_targets = None
        for file_col in ("oifits_h", "MIRCX_file_first", "MIRCX_file"):
            if file_col in df.columns:
                parsed_targets = df[file_col].map(lambda value: (parse_l2_filename(str(value)) or {}).get("target", ""))
                break

        target_col = next((cand for cand in ("target", "Target", "TARGET", "Target_chara") if cand in df.columns), None)
        if parsed_targets is not None and parsed_targets.astype(str).str.strip().ne("").any():
            df["target_norm"] = parsed_targets.map(normalize_target)
        elif target_col is not None:
            df["target_norm"] = df[target_col].map(normalize_target)
        else:
            raise ValueError("Input CSV has no target/target_norm column")
    return df[df["target_norm"].astype(str).ne("")].copy()


def _first_nonnull(series: pd.Series):
    vals = series.dropna()
    if vals.empty:
        return np.nan
    text = vals.astype(str)
    text = text[text.str.strip().ne("")]
    if text.empty:
        return np.nan
    return text.iloc[0]


def _ivw(values: pd.Series, errors: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").to_numpy(float)
    e = pd.to_numeric(errors, errors="coerce").to_numpy(float)
    mask = np.isfinite(v) & np.isfinite(e) & (e > 0)
    if not mask.any():
        finite = v[np.isfinite(v)]
        return (float(np.nanmean(finite)), np.nan) if finite.size else (np.nan, np.nan)
    w = 1.0 / e[mask] ** 2
    return float(np.sum(w * v[mask]) / np.sum(w)), float(np.sqrt(1.0 / np.sum(w)))


def collapse_per_target(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_key(df)
    if df.empty:
        return df

    rows: list[dict] = []
    for target_norm, group in df.groupby("target_norm", dropna=True):
        out = {"target_norm": target_norm}
        for col in group.columns:
            if col == "target_norm":
                continue
            if col in KEY_COLUMNS or group[col].dtype == object:
                out[col] = _first_nonnull(group[col])
                continue
            if col.endswith("_err"):
                continue
            err_col = f"{col}_err"
            if err_col in group.columns:
                out[col], out[err_col] = _ivw(group[col], group[err_col])
            else:
                vals = pd.to_numeric(group[col], errors="coerce")
                out[col] = float(np.nanmean(vals)) if vals.notna().any() else np.nan
        for col in group.columns:
            if col.endswith("_err") and col not in out:
                vals = pd.to_numeric(group[col], errors="coerce")
                out[col] = float(np.nanmean(vals)) if vals.notna().any() else np.nan
        rows.append(out)
    return pd.DataFrame(rows)


def _drop_duplicate_metadata_cols(df: pd.DataFrame, protected: set[str]) -> pd.DataFrame:
    keep = []
    for col in df.columns:
        if col in protected or col not in keep:
            keep.append(col)
    return df.loc[:, keep]


def merge_products(args: argparse.Namespace) -> pd.DataFrame:
    base = _ensure_key(pd.read_csv(args.metadata))
    base = collapse_per_target(base)
    merged = base.copy()

    branches = [
        ("visibility", args.visibility),
        ("exotic", args.exotic),
        ("svam", args.svam),
        ("direct_clv", args.direct_clv),
    ]
    for label, path in branches:
        branch = _read_optional(path, label)
        if branch.empty:
            continue
        branch = collapse_per_target(branch)
        drop_cols = [c for c in ("target", "hd", "date", "oifits_h", "oifits_k", "teff", "logg", "feh", "theta_init", "mass") if c in branch.columns]
        branch = branch.drop(columns=drop_cols, errors="ignore")
        before_cols = len(merged.columns)
        merged = merged.merge(branch, on="target_norm", how="left", suffixes=("", f"_{label}"))
        merged = _drop_duplicate_metadata_cols(merged, {"target_norm"})
        print(f"[merge] joined {label}: +{len(merged.columns) - before_cols} columns")

    preferred = [c for c in ("target", "target_norm", "hd", "date", "oifits_h", "oifits_k", "teff", "logg", "feh", "theta_init", "mass") if c in merged.columns]
    other = [c for c in merged.columns if c not in preferred]
    merged = merged[preferred + other]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"[saved] {args.output} rows={len(merged)} cols={len(merged.columns)}")
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", default=DEFAULT_METADATA)
    parser.add_argument("--visibility", default=DEFAULT_VISIBILITY)
    parser.add_argument("--exotic", default=DEFAULT_EXOTIC)
    parser.add_argument("--svam", default=DEFAULT_SVAM)
    parser.add_argument("--direct-clv", default=DEFAULT_DIRECT_CLV)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    merge_products(parse_args())


if __name__ == "__main__":
    main()
