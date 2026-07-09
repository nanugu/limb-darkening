#!/usr/bin/env python3
"""Merge public branch-level CSV products into one averaged per-target table."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from chara_fit_common import parse_l2_filename
from public_schema import normalize_target

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"

DEFAULT_METADATA = CSV_DIR / "target_metadata.csv"
DEFAULT_VISIBILITY = CSV_DIR / "fit_visibility_laws.csv"
DEFAULT_EXOTIC = CSV_DIR / "compute_exotic_coefficients_LONG.csv"
DEFAULT_SVAM = CSV_DIR / "compute_svam_coefficients.csv"
DEFAULT_DIRECT_CLV = CSV_DIR / "fit_diameters_with_direct_CLV.csv"
DEFAULT_EDR3_DISTANCE = CSV_DIR / "edr3_distance_targets_2026.tsv"
DEFAULT_WIKIPEDIA_DISTANCE = CSV_DIR / 'wikipedia_star_distances.csv'
DEFAULT_OUTPUT = CSV_DIR / "merged_four_branches_wide.csv"

KEY_COLUMNS = {"target", "target_norm", "hd", "date", "oifits_h", "oifits_k"}
_RADIUS_FACTOR_RSUN = 0.10751597682403233
MYSTIC_WL_SCALE = (1.0067, 0.0007)
MIRCX_WL_SCALE_2025 = (0.9990, 0.0010)
MIRCX_WL_SCALE_OTHER = (1.0054, 0.0006)


def _observation_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"(20\d{2})", str(value))
    return int(match.group(1)) if match else None


def _wavelength_scale(band: str, date: object) -> tuple[float, float]:
    if band == "K":
        return MYSTIC_WL_SCALE
    if _observation_year(date) == 2025:
        return MIRCX_WL_SCALE_2025
    return MIRCX_WL_SCALE_OTHER


def _combined_wavelength_scale(date: object) -> tuple[float, float]:
    h_scale, h_err = _wavelength_scale("H", date)
    k_scale, k_err = _wavelength_scale("K", date)
    return 0.5 * (h_scale + k_scale), 0.5 * (h_err + k_err)


def _apply_scale_with_uncertainty(values: pd.Series, errors: pd.Series | None, scale: pd.Series, scale_err: pd.Series) -> tuple[pd.Series, pd.Series | None]:
    vals = pd.to_numeric(values, errors="coerce")
    scale = pd.to_numeric(scale, errors="coerce")
    scale_err = pd.to_numeric(scale_err, errors="coerce")
    corrected = vals / scale
    if errors is None:
        return corrected, None
    errs = pd.to_numeric(errors, errors="coerce")
    statistical = errs / scale
    wavelength = vals.abs() * scale_err / scale**2
    corrected_err = np.sqrt(statistical**2 + wavelength**2)
    return corrected, corrected_err


def _apply_wavelength_scale_corrections(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df

    out = df.copy()
    h_pairs = out["date"].map(lambda value: _wavelength_scale("H", value))
    k_pairs = out["date"].map(lambda value: _wavelength_scale("K", value))
    hk_pairs = out["date"].map(_combined_wavelength_scale)

    h_scale = pd.Series([pair[0] for pair in h_pairs], index=out.index, dtype=float)
    h_scale_err = pd.Series([pair[1] for pair in h_pairs], index=out.index, dtype=float)
    k_scale = pd.Series([pair[0] for pair in k_pairs], index=out.index, dtype=float)
    k_scale_err = pd.Series([pair[1] for pair in k_pairs], index=out.index, dtype=float)
    hk_scale = pd.Series([pair[0] for pair in hk_pairs], index=out.index, dtype=float)
    hk_scale_err = pd.Series([pair[1] for pair in hk_pairs], index=out.index, dtype=float)

    band_specific = {
        "theta_ud_H": (h_scale, h_scale_err),
        "theta_ud_K": (k_scale, k_scale_err),
    }
    for col, (scale, scale_err) in band_specific.items():
        if col not in out.columns:
            continue
        err_col = f"{col}_err" if f"{col}_err" in out.columns else None
        corrected, corrected_err = _apply_scale_with_uncertainty(out[col], out[err_col] if err_col else None, scale, scale_err)
        out[col] = corrected
        if err_col is not None and corrected_err is not None:
            out[err_col] = corrected_err

    skip = {"theta_init", "theta_ud_H", "theta_ud_H_err", "theta_ud_K", "theta_ud_K_err"}
    theta_cols = [col for col in out.columns if col.startswith("theta_") and col not in skip]
    for col in theta_cols:
        err_col = f"{col}_err" if f"{col}_err" in out.columns else None
        corrected, corrected_err = _apply_scale_with_uncertainty(out[col], out[err_col] if err_col else None, hk_scale, hk_scale_err)
        out[col] = corrected
        if err_col is not None and corrected_err is not None:
            out[err_col] = corrected_err

    return out


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


MIN_ERROR_FLOOR = 1e-3

def _ivw(values: pd.Series, errors: pd.Series) -> tuple[float, float]:
    v = pd.to_numeric(values, errors="coerce").to_numpy(float)
    e = pd.to_numeric(errors, errors="coerce").to_numpy(float)
    e = np.where(np.isfinite(e) & (e < MIN_ERROR_FLOOR), MIN_ERROR_FLOOR, e)
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


def _join_keys(df: pd.DataFrame) -> list[str]:
    keys = [c for c in ("target_norm", "oifits_h", "oifits_k") if c in df.columns]
    return keys if keys else ["target_norm"]


def _prepare_target_level(df: pd.DataFrame) -> pd.DataFrame:
    df = _ensure_key(df)
    return collapse_per_target(df)


def _prepare_epoch_level(df: pd.DataFrame) -> pd.DataFrame:
    return _ensure_key(df).copy()


def _drop_duplicate_metadata_cols(df: pd.DataFrame, protected: set[str]) -> pd.DataFrame:
    keep = []
    for col in df.columns:
        if col in protected or col not in keep:
            keep.append(col)
    return df.loc[:, keep]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def _load_gaia_distance_map(path: str | Path) -> tuple[dict[int, float], dict[int, float]]:
    path = Path(path)
    if not path.exists():
        print(f"[merge] edr3_distance: missing {path}; skipping Gaia distance enrichment")
        return {}, {}

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.startswith("_1\t")), None)
    if header_idx is None:
        print(f"[merge] edr3_distance: could not locate header in {path}; skipping Gaia distance enrichment")
        return {}, {}

    ddf = pd.read_csv(path, sep="\t", skiprows=header_idx)
    if len(ddf) >= 2 and str(ddf.iloc[0, 0]).strip() == "":
        ddf = ddf.iloc[2:].copy()

    query_col = _find_col(ddf, ["_1"])
    dist_col = _find_col(ddf, ["rpgeo", "rgeo"])
    low_col = _find_col(ddf, ["b_rpgeo", "b_rgeo"])
    high_col = _find_col(ddf, ["B_rpgeo", "B_rgeo"])
    if query_col is None or dist_col is None:
        print(f"[merge] edr3_distance: required columns not found in {path}; skipping Gaia distance enrichment")
        return {}, {}

    ddf["_query"] = ddf[query_col].astype(str).str.strip()
    ddf["_distance_pc"] = pd.to_numeric(ddf[dist_col], errors="coerce")
    ddf = ddf[ddf["_query"].ne("") & ddf["_distance_pc"].notna()].copy()
    if ddf.empty:
        return {}, {}

    if low_col and high_col:
        ddf["_distance_err_pc"] = 0.5 * (
            pd.to_numeric(ddf[high_col], errors="coerce")
            - pd.to_numeric(ddf[low_col], errors="coerce")
        )
    else:
        ddf["_distance_err_pc"] = np.nan

    ddf["_hd"] = ddf["_query"].str.extract(r"HD\s*([0-9]+)", expand=False)
    ddf["_hd"] = pd.to_numeric(ddf["_hd"], errors="coerce")
    ddf = ddf[ddf["_hd"].notna()].copy()
    if ddf.empty:
        return {}, {}

    ddf = ddf.sort_values("_distance_pc").drop_duplicates("_hd", keep="first")
    dist_by_hd = {int(h): float(d) for h, d in zip(ddf["_hd"], ddf["_distance_pc"])}
    err_by_hd = {
        int(h): float(e) for h, e in zip(ddf["_hd"], ddf["_distance_err_pc"]) if np.isfinite(e)
    }
    print(f"[merge] edr3_distance: matched {len(dist_by_hd)} HD entries from {path}")
    return dist_by_hd, err_by_hd


def _load_wikipedia_distance_map(path: str | Path) -> tuple[dict[int, float], dict[int, float]]:
    path = Path(path)
    if not path.exists():
        print(f"[merge] wikipedia_distance: missing {path}; skipping Wikipedia sanity-check distances")
        return {}, {}

    wdf = pd.read_csv(path)
    hd_col = _find_col(wdf, ["HD", "hd"])
    dist_col = _find_col(wdf, ["distance_pc", "distance"])
    plus_col = _find_col(wdf, ["uncertainty_pc_plus"])
    minus_col = _find_col(wdf, ["uncertainty_pc_minus"])
    if hd_col is None or dist_col is None:
        print(f"[merge] wikipedia_distance: required columns not found in {path}; skipping Wikipedia sanity-check distances")
        return {}, {}

    wdf["_hd"] = wdf[hd_col].astype(str).str.extract(r"HD\s*([0-9]+)", expand=False)
    wdf["_hd"] = pd.to_numeric(wdf["_hd"], errors="coerce")
    wdf["_distance_pc"] = pd.to_numeric(wdf[dist_col], errors="coerce")
    wdf = wdf[wdf["_hd"].notna() & wdf["_distance_pc"].notna()].copy()
    if wdf.empty:
        return {}, {}

    if plus_col and minus_col:
        wdf["_distance_err_pc"] = 0.5 * (
            pd.to_numeric(wdf[plus_col], errors="coerce")
            + pd.to_numeric(wdf[minus_col], errors="coerce")
        )
    elif plus_col:
        wdf["_distance_err_pc"] = pd.to_numeric(wdf[plus_col], errors="coerce")
    else:
        wdf["_distance_err_pc"] = np.nan

    dist_by_hd = {int(h): float(d) for h, d in zip(wdf["_hd"], wdf["_distance_pc"])}
    err_by_hd = {int(h): float(e) for h, e in zip(wdf["_hd"], wdf["_distance_err_pc"]) if np.isfinite(e)}
    print(f"[merge] wikipedia_distance: matched {len(dist_by_hd)} HD entries from {path}")
    return dist_by_hd, err_by_hd


def _compute_radius_rsun(theta_mas: pd.Series, distance_pc: pd.Series) -> pd.Series:
    return _RADIUS_FACTOR_RSUN * pd.to_numeric(theta_mas, errors="coerce") * pd.to_numeric(distance_pc, errors="coerce")


def _compute_radius_rsun_err(
    theta_mas: pd.Series, theta_err_mas: pd.Series, distance_pc: pd.Series, distance_err_pc: pd.Series
) -> pd.Series:
    theta = pd.to_numeric(theta_mas, errors="coerce")
    theta_err = pd.to_numeric(theta_err_mas, errors="coerce")
    dist = pd.to_numeric(distance_pc, errors="coerce")
    dist_err = pd.to_numeric(distance_err_pc, errors="coerce")
    radius = _compute_radius_rsun(theta, dist)
    frac2 = (theta_err / theta) ** 2 + (dist_err / dist) ** 2
    return radius * np.sqrt(frac2)


def _enrich_radius_columns(df: pd.DataFrame, gaia_distance_path: str | Path, wikipedia_distance_path: str | Path | None = None) -> pd.DataFrame:
    theta_col = _find_col(df, ["theta_PL", "PL_diam"])
    theta_err_col = _find_col(df, ["theta_PL_err", "PL_diam_err", "PL_diam_err_ivw"])
    hd_col = _find_col(df, ["hd", "HD"])
    if theta_col is None or hd_col is None:
        return df

    gaia_by_hd, gaia_err_by_hd = _load_gaia_distance_map(gaia_distance_path)
    wiki_by_hd, wiki_err_by_hd = _load_wikipedia_distance_map(wikipedia_distance_path)
    if not gaia_by_hd and not wiki_by_hd:
        return df

    out = df.copy()
    hd_vals = pd.to_numeric(out[hd_col], errors="coerce")
    out["distance_gaia_pc"] = hd_vals.map(gaia_by_hd)
    out["distance_gaia_err_pc"] = hd_vals.map(gaia_err_by_hd)
    out["distance_wikipedia_pc"] = hd_vals.map(wiki_by_hd)
    out["distance_wikipedia_err_pc"] = hd_vals.map(wiki_err_by_hd)

    out["distance_pc"] = out["distance_gaia_pc"]
    out["distance_err_pc"] = out["distance_gaia_err_pc"]
    out["distance_source"] = np.where(out["distance_gaia_pc"].notna(), "Gaia EDR3", pd.NA)

    gaia_missing = out["distance_pc"].isna() & out["distance_wikipedia_pc"].notna()
    if gaia_missing.any():
        out.loc[gaia_missing, "distance_pc"] = out.loc[gaia_missing, "distance_wikipedia_pc"]
        out.loc[gaia_missing, "distance_err_pc"] = out.loc[gaia_missing, "distance_wikipedia_err_pc"]
        out.loc[gaia_missing, "distance_source"] = "Wikipedia"

    both = out["distance_gaia_pc"].notna() & out["distance_wikipedia_pc"].notna()
    if both.any():
        ratio = out.loc[both, "distance_gaia_pc"] / out.loc[both, "distance_wikipedia_pc"]
        bad = both.copy()
        bad.loc[both] = (ratio > 1.5) | (ratio < (1.0 / 1.5))
        if bad.any():
            print(f"[merge] distance sanity-check: using Wikipedia for {int(bad.sum())} inconsistent target(s)")
            out.loc[bad, "distance_pc"] = out.loc[bad, "distance_wikipedia_pc"]
            out.loc[bad, "distance_err_pc"] = out.loc[bad, "distance_wikipedia_err_pc"]
            out.loc[bad, "distance_source"] = "Wikipedia"

    out["Radius_Rsun"] = _compute_radius_rsun(out[theta_col], out["distance_pc"])
    if theta_err_col is not None:
        out["Radius_Rsun_err"] = _compute_radius_rsun_err(
            out[theta_col], out[theta_err_col], out["distance_pc"], out["distance_err_pc"]
        )
    out["Rstar"] = out["Radius_Rsun"]
    if "Radius_Rsun_err" in out.columns:
        out["e_Rstar"] = out["Radius_Rsun_err"]
    return out


def merge_products(args: argparse.Namespace) -> pd.DataFrame:
    visibility = _read_optional(args.visibility, "visibility")
    if visibility.empty:
        raise FileNotFoundError(f"Visibility fit table is required: {args.visibility}")
    visibility = _prepare_epoch_level(visibility)
    visibility = _apply_wavelength_scale_corrections(visibility)
    merged = collapse_per_target(visibility)
    print(f"[merge] base visibility rows={len(merged)}")

    metadata = _read_optional(args.metadata, "metadata")
    if not metadata.empty:
        metadata = _prepare_target_level(metadata)
        metadata_targets = set(metadata["target_norm"].dropna().astype(str))
        extra_targets = sorted(set(merged["target_norm"].dropna().astype(str)) - metadata_targets)
        if extra_targets:
            print(f"[merge] dropping {len(extra_targets)} visibility-only target(s) not in metadata sample: {', '.join(extra_targets)}")
            merged = merged[merged["target_norm"].astype(str).isin(metadata_targets)].copy()
        drop_cols = [c for c in ("target", "date", "oifits_h", "oifits_k", "teff", "logg", "feh", "theta_init") if c in metadata.columns]
        metadata = metadata.drop(columns=drop_cols, errors="ignore")
        before_cols = len(merged.columns)
        merged = merged.merge(metadata, on="target_norm", how="left", suffixes=("", "_metadata"))
        merged = _drop_duplicate_metadata_cols(merged, {"target_norm"})
        print(f"[merge] joined metadata: +{len(merged.columns) - before_cols} columns")

    exotic = _read_optional(args.exotic, "exotic")
    if not exotic.empty:
        exotic = _prepare_target_level(exotic)
        drop_cols = [c for c in ("target", "hd", "date", "oifits_h", "oifits_k", "teff", "logg", "feh", "theta_init", "mass") if c in exotic.columns]
        exotic = exotic.drop(columns=drop_cols, errors="ignore")
        before_cols = len(merged.columns)
        merged = merged.merge(exotic, on="target_norm", how="left", suffixes=("", "_exotic"))
        merged = _drop_duplicate_metadata_cols(merged, {"target_norm"})
        print(f"[merge] joined exotic: +{len(merged.columns) - before_cols} columns")

    for label, path in (("svam", args.svam), ("direct_clv", args.direct_clv)):
        branch = _read_optional(path, label)
        if branch.empty:
            continue
        branch = _prepare_epoch_level(branch)
        if label == "direct_clv":
            branch = _apply_wavelength_scale_corrections(branch)
        branch = collapse_per_target(branch)
        join_cols = ["target_norm"]
        drop_cols = [c for c in ("target", "hd", "date", "teff", "logg", "feh", "theta_init", "mass") if c in branch.columns and c not in join_cols]
        branch = branch.drop(columns=drop_cols, errors="ignore")
        before_cols = len(merged.columns)
        merged = merged.merge(branch, on=join_cols, how="left", suffixes=("", f"_{label}"))
        merged = _drop_duplicate_metadata_cols(merged, set(join_cols))
        print(f"[merge] joined {label} on {join_cols}: +{len(merged.columns) - before_cols} columns")

    # Keep the SATLAS mass-selection metadata near the front of the merged table.
    preferred = [c for c in ("target", "target_norm", "hd", "SpectralType", "date", "oifits_h", "oifits_k", "teff", "logg", "feh", "theta_init", "mass", "cite_key_target_specific", "mass_reference_source") if c in merged.columns]
    other = [c for c in merged.columns if c not in preferred]
    merged = merged[preferred + other]
    merged = _enrich_radius_columns(merged, args.edr3_distance, args.wikipedia_distance)
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
    parser.add_argument("--edr3-distance", default=DEFAULT_EDR3_DISTANCE)
    parser.add_argument("--wikipedia-distance", default=DEFAULT_WIKIPEDIA_DISTANCE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    merge_products(parse_args())


if __name__ == "__main__":
    main()
