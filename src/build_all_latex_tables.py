#!/usr/bin/env python3
"""Build the three LaTeX tables used by the paper."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"
DEFAULT_INPUT = CSV_DIR / "merged_four_branches_wide.csv"
DEFAULT_EDR3 = Path(__file__).with_name("edr3_distance_targets_2026.tsv")
DEFAULT_OUTDIR = ROOT
GRIDS = ("Stagger", "Kurucz", "MPS1", "MPS2")
RADIUS_FACTOR_RSUN = 0.5 * (1e-3 / 206265.0) * (3.085677581e16 / 6.957e8)
MYSTIC_WL_SCALE = (1.0067, 0.0007)
MIRCX_WL_SCALE_2025 = (0.9990, 0.0010)
MIRCX_WL_SCALE_OTHER = (1.0054, 0.0006)


def latex_escape(text: object) -> str:
    if pd.isna(text):
        return ""
    out = str(text)
    for old, new in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        out = out.replace(old, new)
    return out


def greekize_star_name(name: object) -> str:
    if pd.isna(name):
        return ""
    text = str(name).strip()
    greek = {
        "alf": r"\alpha",
        "bet": r"\beta",
        "gam": r"\gamma",
        "del": r"\delta",
        "eps": r"\epsilon",
        "zet": r"\zeta",
        "eta": r"\eta",
        "iot": r"\iota",
        "lam": r"\lambda",
        "mu": r"\mu",
        "rho": r"\rho",
        "sig": r"\sigma",
        "ups": r"\upsilon",
        "ksi": r"\xi",
        "phi": r"\phi",
    }
    m = re.match(r"^phi\s*0?([0-9])\s+(.*)$", text, flags=re.IGNORECASE)
    if m:
        return rf"$\phi^{m.group(1)}$ {latex_escape(m.group(2))}"
    parts = text.split(maxsplit=1)
    key = parts[0].lower().rstrip(".") if parts else ""
    if key in greek:
        suffix = latex_escape(parts[1]) if len(parts) > 1 else ""
        return f"${greek[key]}$ {suffix}".strip()
    return latex_escape(text)


def hd_number(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else None


def hd_text(value: object) -> str:
    hd = hd_number(value)
    return f"HD {hd}" if hd is not None else ""


def number(value: object) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return math.nan
    return val if math.isfinite(val) else math.nan


def fmt(value: object, decimals: int) -> str:
    val = number(value)
    if math.isnan(val):
        return ""
    return f"{val:.{decimals}f}"


def fmt_int(value: object) -> str:
    val = number(value)
    if math.isnan(val):
        return ""
    return f"{val:.0f}"


def fmt_pm(value: object, err: object, decimals: int) -> str:
    val = fmt(value, decimals)
    if not val:
        return ""
    sigma = number(err)
    if math.isnan(sigma):
        return val
    return rf"{val} $\pm$ {sigma:.{decimals}f}"


def fmt_signed(value: float, decimals: int) -> str:
    if math.isnan(value):
        return ""
    return f"{value:+.{decimals}f}"


def require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")


def load_merged(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    require_columns(
        df,
        [
            "target",
            "hd",
            "teff",
            "logg",
            "feh",
            "theta_PL",
            "theta_PL_err",
            "theta_ud_H",
            "theta_ud_H_err",
            "theta_ud_K",
            "theta_ud_K_err",
            "v2_0_PL",
            "alpha_chara_H",
            "alpha_chara_H_err",
            "alpha_chara_K",
            "alpha_chara_K_err",
        ],
    )
    df = df.sort_values(["teff", "target"], ascending=[False, True]).reset_index(drop=True)
    df.insert(0, "ID", range(1, len(df) + 1))
    return df


def existing_target_table_paths(outdir: Path) -> list[Path]:
    conflicted = ROOT / "conflicted" / "target_properties_longtable (Narsireddy Anugu's conflicted copy 2026-05-31).tex"
    paths = []
    if conflicted.exists():
        paths.append(conflicted)
    paths.append(outdir / "target_properties_longtable.tex")
    return paths


def load_existing_spectral_types(paths: list[Path]) -> dict[int, str]:
    """Keep the spectral types already curated in the current paper table."""
    spec_by_hd: dict[int, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "&" not in line or r"\\" not in line:
                continue
            fields = [field.strip() for field in line.split(r"\\", 1)[0].split("&")]
            if len(fields) < 9:
                continue
            hd = hd_number(fields[2])
            if hd is not None and hd not in spec_by_hd:
                spec_by_hd[hd] = fields[3]
    return spec_by_hd


def load_curated_radius_values(paths: list[Path]) -> dict[int, tuple[float, float]]:
    """Fallback radii from the previously curated target table."""
    radius_by_hd: dict[int, tuple[float, float]] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "&" not in line or r"\\" not in line:
                continue
            fields = [field.strip() for field in line.split(r"\\", 1)[0].split("&")]
            if len(fields) < 9:
                continue
            hd = hd_number(fields[2])
            nums = re.findall(r"[-+]?\d+(?:\.\d+)?", fields[8])
            if hd is None or not nums or hd in radius_by_hd:
                continue
            value = float(nums[0])
            err = float(nums[1]) if len(nums) > 1 else math.nan
            radius_by_hd[hd] = (value, err)
    return radius_by_hd


def load_distance_map(path: Path | None) -> tuple[dict[int, float], dict[int, float]]:
    if path is None or not path.exists():
        return {}, {}

    ddf = pd.read_csv(path, sep="\t", comment="#", engine="python", skip_blank_lines=True)
    if "_1" not in ddf.columns:
        return {}, {}

    dist_col = "rgeo" if "rgeo" in ddf.columns else "rpgeo" if "rpgeo" in ddf.columns else None
    if dist_col is None:
        return {}, {}

    ddf["_hd"] = ddf["_1"].apply(hd_number)
    ddf["_distance_pc"] = pd.to_numeric(ddf[dist_col], errors="coerce")
    ddf = ddf.loc[ddf["_hd"].notna() & ddf["_distance_pc"].notna()].copy()

    lower = f"b_{dist_col}"
    upper = f"B_{dist_col}"
    if lower in ddf.columns and upper in ddf.columns:
        ddf["_distance_err_pc"] = 0.5 * (
            pd.to_numeric(ddf[upper], errors="coerce") - pd.to_numeric(ddf[lower], errors="coerce")
        )
    else:
        ddf["_distance_err_pc"] = math.nan

    # The VizieR table is a cone-search export. For each query HD, keep the
    # nearest returned Gaia source, matching the previous table-generation logic.
    ddf = ddf.sort_values("_distance_pc").drop_duplicates("_hd", keep="first")
    dist = {int(row["_hd"]): float(row["_distance_pc"]) for _, row in ddf.iterrows()}
    dist_err = {
        int(row["_hd"]): float(row["_distance_err_pc"])
        for _, row in ddf.iterrows()
        if pd.notna(row["_distance_err_pc"]) and float(row["_distance_err_pc"]) >= 0.0
    }
    return dist, dist_err


def radius_from_theta(theta_mas: object, distance_pc: object) -> float:
    theta = number(theta_mas)
    distance = number(distance_pc)
    if math.isnan(theta) or math.isnan(distance):
        return math.nan
    return RADIUS_FACTOR_RSUN * theta * distance


def radius_err(theta_mas: object, theta_err_mas: object, distance_pc: object, distance_err_pc: object) -> float:
    theta = number(theta_mas)
    theta_err = number(theta_err_mas)
    distance = number(distance_pc)
    distance_err = number(distance_err_pc)
    if any(math.isnan(x) for x in (theta, theta_err, distance, distance_err)):
        return math.nan
    if theta <= 0.0 or distance <= 0.0:
        return math.nan
    radius = radius_from_theta(theta, distance)
    return radius * math.sqrt((theta_err / theta) ** 2 + (distance_err / distance) ** 2)


def observation_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    match = re.search(r"(20\d{2})", str(value))
    return int(match.group(1)) if match else None


def wavelength_scale(band: str, date: object) -> tuple[float, float]:
    if band == "K":
        return MYSTIC_WL_SCALE
    if observation_year(date) == 2025:
        return MIRCX_WL_SCALE_2025
    return MIRCX_WL_SCALE_OTHER


def scaled_theta(value: object, band: str, date: object) -> float:
    val = number(value)
    if math.isnan(val):
        return math.nan
    scale, _ = wavelength_scale(band, date)
    return val / scale


def scaled_theta_err(value: object, err: object, band: str, date: object) -> float:
    val = number(value)
    sigma = number(err)
    if math.isnan(val) or math.isnan(sigma):
        return math.nan
    scale, scale_err = wavelength_scale(band, date)
    statistical = sigma / scale
    wavelength = abs(val) * scale_err / scale**2
    return math.sqrt(statistical**2 + wavelength**2)


def write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {path.resolve()}")


def build_target_properties_table(df: pd.DataFrame, outdir: Path, edr3_path: Path | None) -> None:
    existing_paths = existing_target_table_paths(outdir)
    spec_by_hd = load_existing_spectral_types(existing_paths)
    curated_radius_by_hd = load_curated_radius_values(existing_paths)
    dist_by_hd, dist_err_by_hd = load_distance_map(edr3_path)

    lines = [
        r"\startlongtable",
        r"\begin{deluxetable}{r l l l r r r r r}",
        r"\tablecaption{Target properties with a shared index ($ID$). Targets are sorted by effective temperature (hot to cool). Stellar parameters are compiled from \citet{Baines2025} and \citet{Soubiran2016}. The angular diameter $\theta_{\rm PL}$ and physical radius of stars are computed based on Table~\ref{tab:ld_summary} and Gaia EDR3 distance estimates \citep{BailerJones2021}.}\label{tab:target_properties}",
        r"\tablewidth{0pt}",
        r"\tablehead{",
        r"\colhead{$ID$} & \colhead{Target} & \colhead{HD} & \colhead{Spectral type} & \colhead{$T_{\rm eff}$ (K)} & \colhead{$\log g$} & \colhead{[Fe/H]} & \colhead{$\theta_{\rm PL}$ (mas)} & \colhead{$R/R_\odot$} \\",
        r"}",
        r"\startdata",
    ]

    for _, row in df.iterrows():
        hd = hd_number(row["hd"])
        distance = dist_by_hd.get(hd, math.nan)
        distance_err = dist_err_by_hd.get(hd, math.nan)
        radius = radius_from_theta(row["theta_PL"], distance)
        rad_err = radius_err(row["theta_PL"], row["theta_PL_err"], distance, distance_err)
        curated = curated_radius_by_hd.get(hd)
        if curated is not None:
            curated_radius, curated_err = curated
            use_curated = math.isnan(radius)
            if not use_curated and curated_radius > 0.0:
                ratio = radius / curated_radius
                use_curated = ratio > 1.5 or ratio < (1.0 / 1.5)
            if use_curated:
                radius, rad_err = curated_radius, curated_err
        lines.append(
            " & ".join(
                [
                    str(int(row["ID"])),
                    greekize_star_name(row["target"]),
                    hd_text(row["hd"]),
                    spec_by_hd.get(hd, ""),
                    fmt_int(row["teff"]),
                    fmt(row["logg"], 2),
                    fmt(row["feh"], 2),
                    fmt_pm(row["theta_PL"], row["theta_PL_err"], 3),
                    fmt_pm(radius, rad_err, 2),
                ]
            )
            + r" \\"
        )

    lines.extend([r"\enddata", r"\end{deluxetable}"])
    write_text(outdir / "target_properties_longtable.tex", lines)


def build_ld_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    lines = [
        r"\startlongtable",
        r"\begin{deluxetable}{r l r r r r r r}",
        r"\tablecaption{Uniform-disk ($\theta^{\rm UD}$) angular diameters in the $H$ (MIRC-X) and $K$ (MYSTIC) bands, the combined H+K power-law limb-darkened angular diameter ($\theta_{\rm PL}$), the fitted visibility scale factor ($V^2_0$), and the corresponding empirical CHARA band-dependent power-law coefficients ($\alpha_H^{\rm CHARA}$ and $\alpha_K^{\rm CHARA}$). Same $ID$ and order as Table~\ref{tab:target_properties}.}\label{tab:ld_summary}",
        r"\tablewidth{0pt}",
        r"\tablehead{\colhead{$ID$} & \colhead{Target} & \colhead{$\theta^{\rm UD}_{H}$ (mas)} & \colhead{$\theta^{\rm UD}_{K}$ (mas)} & \colhead{$\theta_{\rm PL}$ (mas)} & \colhead{$V^2_0$} & \colhead{$\alpha_H^{\rm CHARA}$} & \colhead{$\alpha_K^{\rm CHARA}$} \\}",
        r"\startdata",
    ]

    for _, row in df.iterrows():
        lines.append(
            " & ".join(
                [
                    str(int(row["ID"])),
                    greekize_star_name(row["target"]),
                    fmt_pm(
                        scaled_theta(row["theta_ud_H"], "H", row["date"]),
                        scaled_theta_err(row["theta_ud_H"], row["theta_ud_H_err"], "H", row["date"]),
                        3,
                    ),
                    fmt_pm(
                        scaled_theta(row["theta_ud_K"], "K", row["date"]),
                        scaled_theta_err(row["theta_ud_K"], row["theta_ud_K_err"], "K", row["date"]),
                        3,
                    ),
                    fmt_pm(row["theta_PL"], row["theta_PL_err"], 3),
                    fmt(row["v2_0_PL"], 3),
                    fmt_pm(row["alpha_chara_H"], row["alpha_chara_H_err"], 3),
                    fmt_pm(row["alpha_chara_K"], row["alpha_chara_K_err"], 3),
                ]
            )
            + r" \\"
        )

    lines.extend([r"\enddata", r"\end{deluxetable}"])
    write_text(outdir / "ld_summary_deluxetable.tex", lines)


def grid_i_col(grid: str, band: str) -> str:
    instrument = "MIRCX" if band == "H" else "MYSTIC"
    return f"p1__power1__{grid.lower()}__{instrument}"


def grid_svam_col(grid: str, band: str) -> str:
    return f"alpha_svam_{band}_{grid}"


def median_series(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    return float(series.median()) if not series.empty else math.nan


def median_fractional_hk(df: pd.DataFrame, h_col: str, k_col: str) -> float:
    h = pd.to_numeric(df[h_col], errors="coerce")
    k = pd.to_numeric(df[k_col], errors="coerce")
    return median_series(100.0 * (h - k) / h)


def median_percent_offset(df: pd.DataFrame, chara_col: str, model_col: str) -> float:
    chara = pd.to_numeric(df[chara_col], errors="coerce")
    model = pd.to_numeric(df[model_col], errors="coerce")
    return median_series(100.0 * (chara - model) / model)


def build_hk_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    for grid in GRIDS:
        require_columns(df, [grid_i_col(grid, "H"), grid_i_col(grid, "K"), grid_svam_col(grid, "H"), grid_svam_col(grid, "K")])

    chara_hk = median_series(df["alpha_chara_H"] - df["alpha_chara_K"])
    chara_hk_pct = median_fractional_hk(df, "alpha_chara_H", "alpha_chara_K")
    lines = [
        r"\begin{table*}",
        r"\centering",
        r"\caption{Median cross-band coefficient behavior and CHARA--model offsets. The empirical CHARA median is",
        rf"$\alpha_H^{{\rm CHARA}}-\alpha_K^{{\rm CHARA}}={chara_hk:.4f}$, corresponding to",
        rf"$100(\alpha_H^{{\rm CHARA}}-\alpha_K^{{\rm CHARA}})/\alpha_H^{{\rm CHARA}}={chara_hk_pct:.2f}\%$.",
        r"The second and third columns give the model cross-band decrease,",
        r"$100(\alpha_H-\alpha_K)/\alpha_H$, for coefficients fit directly to",
        r"$I(\mu)$ and for SVAM coefficients, respectively. The final two columns give",
        r"the visibility-domain SVAM offsets",
        r"$100(\alpha_b^{\rm CHARA}-\alpha_{{\rm SVAM},b}^g)/\alpha_{{\rm SVAM},b}^g$.}",
        r"\label{tab:hk_model_summary}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Grid & $I(\mu)$ $H-K$ (\%) & SVAM $H-K$ (\%) & SVAM $\Delta_H$ (\%) & SVAM $\Delta_K$ (\%) \\",
        r"\midrule",
    ]

    for grid in GRIDS:
        i_hk = median_fractional_hk(df, grid_i_col(grid, "H"), grid_i_col(grid, "K"))
        svam_hk = median_fractional_hk(df, grid_svam_col(grid, "H"), grid_svam_col(grid, "K"))
        svam_h = median_percent_offset(df, "alpha_chara_H", grid_svam_col(grid, "H"))
        svam_k = median_percent_offset(df, "alpha_chara_K", grid_svam_col(grid, "K"))
        lines.append(f"{grid} & {i_hk:.2f} & {svam_hk:.2f} & {fmt_signed(svam_h, 2)} & {fmt_signed(svam_k, 2)} " + r"\\")

    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    write_text(outdir / "hk_data_vs_model_summary.tex", lines)


def remove_stale_outputs(outdir: Path) -> None:
    for filename in ("calibrators_deluxetable.tex", "target_index_mapping.csv", "final_calibrators_list.csv"):
        path = outdir / filename
        if path.exists():
            path.unlink()
            print(f"Removed stale output: {path.resolve()}")


def build_tables(input_csv: Path, edr3_distance_tsv: Path | None, outdir: Path) -> None:
    df = load_merged(input_csv)
    outdir.mkdir(parents=True, exist_ok=True)
    remove_stale_outputs(outdir)
    build_target_properties_table(df, outdir, edr3_distance_tsv)
    build_ld_summary_table(df, outdir)
    build_hk_summary_table(df, outdir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--edr3-distance-tsv", type=Path, default=DEFAULT_EDR3)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()
    build_tables(args.input_csv, args.edr3_distance_tsv, args.outdir)


if __name__ == "__main__":
    main()
