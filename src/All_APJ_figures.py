#!/usr/bin/env python3
"""Generate the main ApJ limb-darkening figures from the wide merged table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"
DEFAULT_CSV = CSV_DIR / "merged_four_branches_wide.csv"
DEFAULT_OUTDIR = ROOT / "Figures"

GRID_STYLES = {
    "Stagger": {"color": "0.10", "marker": "o"},
    "Kurucz": {"color": "0.30", "marker": "*"},
    "MPS1": {"color": "0.50", "marker": "^"},
    "MPS2": {"color": "0.70", "marker": "D"},
    "SATLASRoss": {"color": "0.20", "marker": "P"},
}
DIAMETER_GRID_STYLES = {
    **GRID_STYLES,
}
GRID_KEYS = {
    "Stagger": "stagger",
    "Kurucz": "kurucz",
    "MPS1": "mps1",
    "MPS2": "mps2",
    "SATLASRoss": "satlasross",
}

LAW_SPECS = {
    "power": {
        "symbol": r"\alpha",
        "empirical_label": "CHARA power-law fit",
        "model_i_label": r"ExoTiC-LD {grid} $I(\mu)$",
        "model_svam_label": r"Fit model $V^2$ ({grid})",
    },
    "linear": {
        "symbol": "a",
        "empirical_label": "CHARA linear-law fit",
        "model_i_label": r"ExoTiC-LD {grid} $I(\mu)$",
        "model_svam_label": r"Fit model $V^2$ ({grid})",
    },
}

OTHER_LAW_DIAMETERS = [
    ("Power-2 law", ["theta_power2", "PL2_diam"], ["theta_power2_err", "PL2_diam_err", "PL2_diam_err_ivw"], "s", "#1B9E77"),
    ("Linear law", ["theta_linear", "LL_diam"], ["theta_linear_err", "LL_diam_err", "LL_diam_err_ivw"], "X", "#D95F02"),
    ("Quadratic law", ["theta_quadratic", "QL_diam"], ["theta_quadratic_err", "QL_diam_err", "QL_diam_err_ivw"], "*", "#7570B3"),
    ("UD H", ["theta_ud_H", "H_UD_diam"], ["theta_ud_H_err", "H_UD_diam_err", "H_UD_diam_err_ivw"], "^", "#1F78B4"),
    ("UD K", ["theta_ud_K", "K_UD_diam"], ["theta_ud_K_err", "K_UD_diam_err", "K_UD_diam_err_ivw"], "v", "#A6761D"),
]

DIRECT_CLV_DIAMETERS = [
    ("SATLAS", ["theta_CLV_SATLAS"], ["theta_CLV_SATLAS_err"], "X", "#7B3294"),
    ("Stagger", ["theta_CLV_Stagger"], ["theta_CLV_Stagger_err"], "o", "#000000"),
    ("Kurucz", ["theta_CLV_Kurucz"], ["theta_CLV_Kurucz_err"], "s", "#7F7F7F"),
    ("MPS1", ["theta_CLV_MPS1"], ["theta_CLV_MPS1_err"], "^", "#999999"),
    ("MPS2", ["theta_CLV_MPS2"], ["theta_CLV_MPS2_err"], "D", "#555555"),
]


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def values(df: pd.DataFrame, candidates: list[str]) -> np.ndarray:
    col = pick_col(df, candidates)
    if col is None:
        return np.full(len(df), np.nan)
    return pd.to_numeric(df[col], errors="coerce").to_numpy(float)


def col_exists(df: pd.DataFrame, candidates: list[str]) -> bool:
    return pick_col(df, candidates) is not None


def short_target_label(value) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    text = text.replace("_", " ")
    greek = {
        "alf": r"$\alpha$",
        "bet": r"$\beta$",
        "gam": r"$\gamma$",
        "del": r"$\delta$",
        "eps": r"$\epsilon$",
        "zet": r"$\zeta$",
        "eta": r"$\eta$",
        "tet": r"$\theta$",
        "iot": r"$\iota$",
        "kap": r"$\kappa$",
        "lam": r"$\lambda$",
        "mu": r"$\mu$",
        "nu": r"$\nu$",
        "ksi": r"$\xi$",
        "pi": r"$\pi$",
        "rho": r"$\rho$",
        "sig": r"$\sigma$",
        "tau": r"$\tau$",
        "ups": r"$\upsilon$",
        "phi": r"$\phi$",
        "chi": r"$\chi$",
        "psi": r"$\psi$",
        "ome": r"$\omega$",
    }
    first = text.split()[0]
    key = first[:3].lower()
    if key in greek:
        suffix = first[3:]
        prefix = greek[key]
        if suffix:
            text = " ".join([f"{prefix}{suffix}"] + text.split()[1:])
        else:
            text = " ".join([prefix] + text.split()[1:])
    return text


def target_labels(df: pd.DataFrame) -> np.ndarray:
    col = pick_col(df, ["target", "Target", "Target_catalog", "target_norm"])
    if col is None:
        return np.full(len(df), "", dtype=object)
    return df[col].to_numpy(dtype=object)


def annotate_points(ax, x: np.ndarray, y: np.ndarray, labels: np.ndarray, *, fontsize: float = 5.8) -> None:
    offsets = [(3, 3), (3, -5), (-8, 3), (-8, -5), (5, 6), (-10, 6)]
    count = 0
    for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        lab = short_target_label(label)
        if not lab:
            continue
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            lab, xy=(xi, yi), xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, color="0.12", alpha=0.90, zorder=20,
            ha="left" if dx >= 0 else "right", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.25),
            arrowprops=dict(
                arrowstyle="-", color="0.25", alpha=0.55, lw=0.35,
                shrinkA=0.0, shrinkB=1.5,
            ),
        )
        count += 1


def grid_i_col(grid: str, band: str, law: str = "power") -> list[str]:
    key = GRID_KEYS.get(grid, str(grid).lower())
    if law == "power":
        return [
            f"alpha_I_{band}_{grid}",
            f"alpha_I_{band}_{key}",
        ]
    if law == "linear":
        return [
            f"a_I_{band}_{grid}",
            f"a_I_{band}_{key}",
        ]
    raise ValueError(f"Unsupported law: {law}")


def grid_svam_col(grid: str, band: str, law: str = "power") -> list[str]:
    key = GRID_KEYS.get(grid, str(grid).lower())
    if law == "power":
        return [
            f"alpha_svam_{band}_{grid}",
            f"alpha_SVAM_{band}_{grid}",
            f"alpha_svam_{band}_{key}",
            f"alpha_SVAM_{band}_{key}",
        ]
    if law == "linear":
        return [
            f"a_svam_{band}_{grid}",
            f"a_SVAM_{band}_{grid}",
            f"a_svam_{band}_{key}",
            f"a_SVAM_{band}_{key}",
        ]
    raise ValueError(f"Unsupported law: {law}")


def coefficient_label(source: str, grid: str | None = None, law: str = "power") -> str:
    spec = LAW_SPECS[law]
    if grid is None:
        return spec["empirical_label"]
    if source == "I":
        if grid == "SATLASRoss":
            return r"SATLAS $I(\mu_{\rm Ross})$"
        return spec["model_i_label"].format(grid=grid)
    if source == "SVAM":
        if grid == "SATLASRoss":
            return r"Fit SATLAS $V^2$ ($\mu_{\rm Ross}$)"
        return spec["model_svam_label"].format(grid=grid)
    return grid


def percent_residual_ylabel(source: str, grid: str, law: str = "power") -> str:
    symbol = LAW_SPECS[law]["symbol"]
    model = rf"{symbol}_{{\rm {grid}}}"
    return rf"$({symbol}_{{\rm CHARA}} - {model})/{model}$ (%)"


def chara_coeff(df: pd.DataFrame, band: str, law: str = "power") -> np.ndarray:
    if law == "power":
        return values(df, [f"alpha_chara_{band}", f"alpha_{band}_CHARA"])
    if law == "linear":
        return values(df, [f"a_chara_{band}", f"{band}_LL_a"])
    raise ValueError(f"Unsupported law: {law}")


def chara_coeff_err(df: pd.DataFrame, band: str, law: str = "power") -> np.ndarray:
    if law == "power":
        return values(df, [f"alpha_chara_{band}_err", f"alpha_{band}_CHARA_err"])
    if law == "linear":
        return values(df, [f"a_chara_{band}_err", f"{band}_LL_a_err_ivw"])
    raise ValueError(f"Unsupported law: {law}")


def median_residuals(df: pd.DataFrame, source: str, *, percent: bool, law: str = "power") -> list[dict[str, float | int | str]]:
    rows = []
    for grid in GRID_STYLES:
        row: dict[str, float | int | str] = {"grid": grid}
        for band in ("H", "K"):
            y_chara = chara_coeff(df, band, law=law)
            y_model = values(df, grid_i_col(grid, band, law=law) if source == "I" else grid_svam_col(grid, band, law=law))
            mask = finite_xy(y_chara, y_model)
            if percent:
                mask &= y_model != 0
            residual = y_chara[mask] - y_model[mask]
            if percent:
                residual = 100.0 * residual / y_model[mask]
            row[f"{band}_median"] = float(np.nanmedian(residual)) if residual.size else np.nan
            row[f"n_{band}"] = int(mask.sum())
        row["H_minus_K"] = float(row["H_median"]) - float(row["K_median"])
        rows.append(row)
    return rows


def median_h_minus_k(values_h: np.ndarray, values_k: np.ndarray) -> tuple[float, int]:
    mask = finite_xy(values_h, values_k)
    delta = values_h[mask] - values_k[mask]
    return (float(np.nanmedian(delta)) if delta.size else np.nan, int(mask.sum()))


def median_h_minus_k_percent(values_h: np.ndarray, values_k: np.ndarray) -> tuple[float, int]:
    mask = finite_xy(values_h, values_k) & (values_h != 0)
    delta_pct = 100.0 * (values_h[mask] - values_k[mask]) / values_h[mask]
    return (float(np.nanmedian(delta_pct)) if delta_pct.size else np.nan, int(mask.sum()))


def print_h_minus_k_table(df: pd.DataFrame) -> None:
    chara_hk, n_chara = median_h_minus_k(chara_coeff(df, "H"), chara_coeff(df, "K"))
    print("H-K coefficient differences:")
    print(f"{'Grid':<10}{'CHARA H-K':>12}{'I H-K':>12}{'SVAM H-K':>12}{'n_CHARA':>10}{'n_I':>7}{'n_SVAM':>9}")
    for grid in GRID_STYLES:
        i_hk, n_i = median_h_minus_k(
            values(df, grid_i_col(grid, "H")),
            values(df, grid_i_col(grid, "K")),
        )
        svam_hk, n_svam = median_h_minus_k(
            values(df, grid_svam_col(grid, "H")),
            values(df, grid_svam_col(grid, "K")),
        )
        print(
            f"{grid:<10}"
            f"{chara_hk:>+12.4f}"
            f"{i_hk:>+12.4f}"
            f"{svam_hk:>+12.4f}"
            f"{n_chara:>10d}"
            f"{n_i:>7d}"
            f"{n_svam:>9d}"
        )
    print()

    chara_hk_pct, n_chara_pct = median_h_minus_k_percent(chara_coeff(df, "H"), chara_coeff(df, "K"))
    print("H-K coefficient differences (%):")
    print(f"{'Grid':<10}{'CHARA H-K (%)':>15}{'I H-K (%)':>13}{'SVAM H-K (%)':>16}{'n_CHARA':>10}{'n_I':>7}{'n_SVAM':>9}")
    for grid in GRID_STYLES:
        i_hk_pct, n_i_pct = median_h_minus_k_percent(
            values(df, grid_i_col(grid, "H")),
            values(df, grid_i_col(grid, "K")),
        )
        svam_hk_pct, n_svam_pct = median_h_minus_k_percent(
            values(df, grid_svam_col(grid, "H")),
            values(df, grid_svam_col(grid, "K")),
        )
        print(
            f"{grid:<10}"
            f"{chara_hk_pct:>+15.2f}"
            f"{i_hk_pct:>+13.2f}"
            f"{svam_hk_pct:>+16.2f}"
            f"{n_chara_pct:>10d}"
            f"{n_i_pct:>7d}"
            f"{n_svam_pct:>9d}"
        )
    print()


def print_percent_residual_tables(df: pd.DataFrame) -> None:
    print_h_minus_k_table(df)

    print("\nCHARA - model:")
    for source, title in (("I", "Direct I(mu) coefficients"), ("SVAM", "SVAM coefficients")):
        print(title)
        print(f"{'Grid':<10}{'H median':>12}{'K median':>13}{'H-K':>10}{'n_H':>7}{'n_K':>6}")
        for row in median_residuals(df, source, percent=False):
            print(
                f"{row['grid']:<10}"
                f"{row['H_median']:>+12.4f}"
                f"{row['K_median']:>+13.4f}"
                f"{row['H_minus_K']:>+10.4f}"
                f"{row['n_H']:>7d}"
                f"{row['n_K']:>6d}"
            )
        print()

    print("100*(CHARA - model) / model:\n")
    for source, title in (("I", "Direct I(mu) coefficients"), ("SVAM", "SVAM coefficients")):
        print(title)
        print(f"{'Grid':<10}{'H median (%)':>14}{'K median (%)':>15}{'H-K (%)':>11}{'n_H':>7}{'n_K':>6}")
        for row in median_residuals(df, source, percent=True):
            print(
                f"{row['grid']:<10}"
                f"{row['H_median']:>+14.2f}"
                f"{row['K_median']:>+15.2f}"
                f"{row['H_minus_K']:>+11.2f}"
                f"{row['n_H']:>7d}"
                f"{row['n_K']:>6d}"
            )
        print()

    print("100*(alpha_I - alpha_SVAM) / alpha_I:\n")
    print(f"{'Grid':<12}{'H median (%)':>14}{'K median (%)':>15}{'H-K (%)':>11}{'n_H':>7}{'n_K':>6}")
    for grid in GRID_STYLES:
        row: dict[str, float | int | str] = {"grid": grid}
        for band in ("H", "K"):
            y_i = values(df, grid_i_col(grid, band))
            y_svam = values(df, grid_svam_col(grid, band))
            mask = finite_xy(y_i, y_svam) & (y_i != 0)
            residual = 100.0 * (y_i[mask] - y_svam[mask]) / y_i[mask]
            row[f"{band}_median"] = float(np.nanmedian(residual)) if residual.size else np.nan
            row[f"n_{band}"] = int(mask.sum())
        row["H_minus_K"] = float(row["H_median"]) - float(row["K_median"])
        print(
            f"{row['grid']:<12}"
            f"{row['H_median']:>+14.2f}"
            f"{row['K_median']:>+15.2f}"
            f"{row['H_minus_K']:>+11.2f}"
            f"{row['n_H']:>7d}"
            f"{row['n_K']:>6d}"
        )
    print()


def teff_values(df: pd.DataFrame) -> np.ndarray:
    return values(df, ["teff", "Teff"])


def finite_xy(x: np.ndarray, y: np.ndarray, *extra: np.ndarray) -> np.ndarray:
    mask = np.isfinite(x) & np.isfinite(y)
    for arr in extra:
        mask &= np.isfinite(arr)
    return mask


def has_any_linear_svam(df: pd.DataFrame) -> bool:
    for grid in GRID_STYLES:
        for band in ("H", "K"):
            if col_exists(df, grid_svam_col(grid, band, law="linear")):
                return True
    return False


def color_norm(teff: np.ndarray, cmap_name: str):
    finite = teff[np.isfinite(teff)]
    cmap = mpl.colormaps.get_cmap(cmap_name)
    if finite.size:
        vmin, vmax = np.nanpercentile(finite, [2, 98])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
            vmin, vmax = np.nanmin(finite), np.nanmax(finite)
    else:
        vmin, vmax = 0.0, 1.0
    return cmap, mpl.colors.Normalize(vmin=float(vmin), vmax=float(vmax))


def set_equal_limits(ax, arrays: list[np.ndarray]) -> None:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        return
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.06 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], "--", color="0.55", lw=1.0, zorder=0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")


def binned_trend(x: np.ndarray, y: np.ndarray, target_bins: int = 7) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    if x.size < 5:
        order = np.argsort(x)
        return x[order], y[order]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    n_bins = int(np.clip(target_bins, 3, max(3, x.size // 3)))
    groups = np.array_split(np.arange(x.size), n_bins)
    return (
        np.array([np.nanmedian(x[idx]) for idx in groups if idx.size], dtype=float),
        np.array([np.nanmedian(y[idx]) for idx in groups if idx.size], dtype=float),
    )


def plot_chara_vs_model_coeff_1to1(
    df: pd.DataFrame,
    *,
    source: str,
    output: Path,
    dpi: int,
    law: str = "power",
) -> None:
    coeff_symbol = LAW_SPECS[law]["symbol"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharex=False, sharey=False)
    panel_specs = [("H", axes[0]), ("K", axes[1])]

    for band, ax in panel_specs:
        y_chara = chara_coeff(df, band, law=law)
        yerr = chara_coeff_err(df, band, law=law)
        all_arrays: list[np.ndarray] = []

        for grid, style in GRID_STYLES.items():
            x_model = values(df, grid_i_col(grid, band, law=law) if source == "I" else grid_svam_col(grid, band, law=law))
            mask = finite_xy(x_model, y_chara)
            if not mask.any():
                continue
            xx = x_model[mask]
            yy = y_chara[mask]
            ye = yerr[mask]
            ax.errorbar(
                xx, yy, yerr=ye, fmt="none",
                ecolor=style["color"], elinewidth=0.8, capsize=1.8, alpha=0.18, zorder=1,
            )
            ax.scatter(
                xx, yy, s=44, marker=style["marker"],
                facecolors="none" if grid != "Stagger" else style["color"],
                edgecolors=style["color"], linewidths=0.9, alpha=0.88,
                label=grid, zorder=3,
            )
            all_arrays.extend([xx, yy])

        if all_arrays:
            set_equal_limits(ax, all_arrays)
            ax.text(0.06, 0.90, "1:1", transform=ax.transAxes, color="0.35", fontsize=10.5)

        ax.set_title(rf"${band}$ band", fontsize=12)
        ax.set_xlabel(rf"Model ${coeff_symbol}_{{{band}}}$")
        ax.set_ylabel(rf"CHARA ${coeff_symbol}_{{{band}}}$")
        ax.grid(True, which="major", alpha=0.24, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.09, linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction="in", which="both", top=True, right=True)

    axes[1].legend(frameon=False, fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def finish_diameter_axes(ax, ax_res, all_values: list[np.ndarray], all_residuals: list[np.ndarray],
                         xlabel: str, ylabel: str, residual_ylabel: str) -> None:
    vals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in all_values])
    vals = vals[np.isfinite(vals)]
    if not vals.size:
        raise SystemExit("No finite diameter pairs found for requested plot.")
    lo, hi = float(vals.min()), float(vals.max())
    pad = 0.05 * (hi - lo if hi > lo else 1.0)
    lo -= pad
    hi += pad
    ax.plot([lo, hi], [lo, hi], "--", lw=1.2, color="0.45", zorder=0)
    ax.text(0.07, 0.92, "1:1", transform=ax.transAxes, color="0.35", fontsize=10.5)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    residuals = np.concatenate([np.asarray(v, dtype=float).ravel() for v in all_residuals])
    residuals = residuals[np.isfinite(residuals)]
    ax_res.axhline(0.0, color="0.35", lw=1.0, zorder=0)
    if residuals.size:
        rlo, rhi = float(residuals.min()), float(residuals.max())
        rpad = max(0.25, 0.10 * (rhi - rlo)) if rhi > rlo else 0.5
        ax_res.set_ylim(rlo - rpad, rhi + rpad)

    ax.set_ylabel(ylabel)
    ax_res.set_xlabel(xlabel)
    ax_res.set_ylabel(residual_ylabel)
    for axis in (ax, ax_res):
        axis.grid(True, which="major", alpha=0.22, linewidth=0.8)
        axis.grid(True, which="minor", alpha=0.08, linewidth=0.5)
        axis.xaxis.set_minor_locator(AutoMinorLocator())
        axis.yaxis.set_minor_locator(AutoMinorLocator())
        axis.tick_params(direction="in", which="both", top=True, right=True)
    ax.text(0.015, 0.975, "(a)", transform=ax.transAxes, va="top", ha="left", fontsize=15.5)
    ax_res.text(0.015, 0.96, "(b)", transform=ax_res.transAxes, va="top", ha="left", fontsize=15.5)


def plot_hk_comparison(
    df: pd.DataFrame,
    *,
    source: str,
    output: Path,
    dpi: int,
    cmap_name: str,
    law: str = "power",
) -> None:
    teff = teff_values(df)
    cmap, norm = color_norm(teff, cmap_name)
    coeff_symbol = LAW_SPECS[law]["symbol"]
    h_chara = chara_coeff(df, "H", law=law)
    k_chara = chara_coeff(df, "K", law=law)
    eh_chara = chara_coeff_err(df, "H", law=law)
    ek_chara = chara_coeff_err(df, "K", law=law)

    fig, ax = plt.subplots(figsize=(5.7, 5.3))
    m = finite_xy(h_chara, k_chara)
    ax.errorbar(
        h_chara[m], k_chara[m], xerr=eh_chara[m], yerr=ek_chara[m],
        fmt="none", ecolor=(0.2, 0.35, 0.55, 0.28), elinewidth=1.0,
        capsize=2, zorder=1,
    )
    ax.scatter(
        h_chara[m], k_chara[m], c=teff[m], cmap=cmap, norm=norm,
        s=52, edgecolors="k", linewidths=0.45, zorder=3,
        label=coefficient_label(source),
    )
    if m.sum() >= 2:
        fit = np.polyfit(h_chara[m], k_chara[m], 1)
        xs = np.linspace(np.nanmin(h_chara[m]), np.nanmax(h_chara[m]), 100)
        ax.plot(xs, fit[0] * xs + fit[1], color="#2f80ed", lw=2.2)

    all_arrays = [h_chara[m], k_chara[m]]
    for grid, style in GRID_STYLES.items():
        h = values(df, grid_i_col(grid, "H") if source == "I" else grid_svam_col(grid, "H"))
        k = values(df, grid_i_col(grid, "K") if source == "I" else grid_svam_col(grid, "K"))
        gm = finite_xy(h, k)
        if not gm.any():
            continue
        ax.scatter(
        h[gm], k[gm], s=38, facecolors="none", edgecolors=style["color"],
            marker=style["marker"], linewidths=0.8, alpha=0.85,
            label=coefficient_label(source, grid, law=law),
        )
        if gm.sum() >= 2:
            fit = np.polyfit(h[gm], k[gm], 1)
            xs = np.linspace(np.nanmin(h[gm]), np.nanmax(h[gm]), 100)
            ax.plot(xs, fit[0] * xs + fit[1], color=style["color"], lw=1.8, alpha=0.8)
        all_arrays.extend([h[gm], k[gm]])

    set_equal_limits(ax, all_arrays)
    ax.set_xlabel(r"$H$-band power-law coefficient")
    ax.set_ylabel(r"$K$-band power-law coefficient")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="best")
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$T_{\rm eff}$ (K)")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def plot_teff_comparison(
    df: pd.DataFrame,
    *,
    source: str,
    output: Path,
    dpi: int,
    cmap_name: str,
    residual_grid: str = "MPS2",
    label_targets: bool = False,
    law: str = "power",
) -> None:
    teff = teff_values(df)
    cmap, norm = color_norm(teff, cmap_name)
    coeff_symbol = LAW_SPECS[law]["symbol"]
    h_chara, k_chara = chara_coeff(df, "H", law=law), chara_coeff(df, "K", law=law)
    eh_chara, ek_chara = chara_coeff_err(df, "H", law=law), chara_coeff_err(df, "K", law=law)
    labels = target_labels(df)

    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 6.05), sharex="col",
        gridspec_kw={"height_ratios": [2.65, 1.0], "hspace": 0.08, "wspace": 0.18},
    )
    panels = [("H", h_chara, eh_chara, axes[0, 0], axes[1, 0]),
              ("K", k_chara, ek_chara, axes[0, 1], axes[1, 1])]

    for band, y_chara, e_chara, ax, rax in panels:
        m = finite_xy(teff, y_chara)
        panel_yvals: list[np.ndarray] = []
        if m.any():
            panel_yvals.append(y_chara[m])
        ax.errorbar(teff[m], y_chara[m], yerr=e_chara[m], fmt="none",
                    ecolor=(0.2, 0.35, 0.55, 0.28), elinewidth=1.0, capsize=2)
        ax.scatter(teff[m], y_chara[m], c=teff[m], cmap=cmap, norm=norm,
                   s=48, edgecolors="k", linewidths=0.4, zorder=3,
                   label=coefficient_label(source, law=law))
        if label_targets:
            annotate_points(ax, teff[m], y_chara[m], labels[m])

        ref = None
        for grid, style in GRID_STYLES.items():
            y_grid = values(df, grid_i_col(grid, band, law=law) if source == "I" else grid_svam_col(grid, band, law=law))
            gm = finite_xy(teff, y_grid)
            if not gm.any():
                continue
            order = np.argsort(teff[gm])
            ax.plot(teff[gm][order], y_grid[gm][order], color=style["color"],
                    lw=1.6, alpha=0.85, marker=style["marker"], mfc="none",
                    mec=style["color"], ms=5,
                    label=coefficient_label(source, grid, law=law))
            panel_yvals.append(y_grid[gm])
            if grid == residual_grid:
                ref = y_grid

        if ref is not None:
            rm = finite_xy(teff, y_chara, ref) & (ref != 0)
            residual_pct = 100.0 * (y_chara[rm] - ref[rm]) / ref[rm]
            residual_err_pct = 100.0 * e_chara[rm] / np.abs(ref[rm])
            rax.axhline(0, color="0.55", ls="--", lw=1.0)
            rax.errorbar(teff[rm], residual_pct, yerr=residual_err_pct,
                         fmt="none", ecolor=(0.2, 0.35, 0.55, 0.35),
                         capsize=2)
            rax.scatter(teff[rm], residual_pct, c=teff[rm],
                        cmap=cmap, norm=norm, s=36, edgecolors="k",
                        linewidths=0.35, zorder=3)

        #ax.set_title(f"{band} band")
        ax.set_ylabel(rf"${band}$-band power-law coefficient")
        ax.grid(True, alpha=0.25)
        rax.set_xlabel(r"$T_{\rm eff}$ (K)")
        if source == "I":
            rax.set_ylabel(
                rf"$(\alpha_{{{band}}}^{{\rm CHARA}}-\alpha_{{I,{band}}}^{{\rm {residual_grid}}})/"
                rf"\alpha_{{I,{band}}}^{{\rm {residual_grid}}}\ (\%)$"
            )
        else:
            rax.set_ylabel(
                rf"$(\alpha_{{{band}}}^{{\rm CHARA}}-\alpha_{{{{\rm SVAM}},{band}}}^{{\rm {residual_grid}}})/"
                rf"\alpha_{{{{\rm SVAM}},{band}}}^{{\rm {residual_grid}}}\ (\%)$"
            )
        rax.grid(True, alpha=0.25)
        rax.set_ylim(-100, 100)
        ax.set_ylim([0.05, 0.36])

    axes[0, 1].legend(frameon=False, fontsize=8, loc="best")
    fig.subplots_adjust(left=0.075, right=0.88, bottom=0.10, top=0.985)
    cax = fig.add_axes([0.895, 0.14, 0.022, 0.72])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.set_label(r"$T_{\rm eff}$ (K)")
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def plot_i_vs_svam_teff_comparison(
    df: pd.DataFrame,
    *,
    output: Path,
    dpi: int,
    law: str = "power",
) -> None:
    teff = teff_values(df)
    fig, axes = plt.subplots(
        2, 2, figsize=(10.5, 6.0), sharex="col",
        gridspec_kw={"height_ratios": [2.65, 1.0], "hspace": 0.08, "wspace": 0.18},
    )
    panels = [
        ("H", axes[0, 0], axes[1, 0]),
        ("K", axes[0, 1], axes[1, 1]),
    ]

    for band, ax, rax in panels:
        top_values: list[np.ndarray] = []
        residual_values: list[np.ndarray] = []
        for grid, style in GRID_STYLES.items():
            y_i = values(df, grid_i_col(grid, band, law=law))
            y_svam = values(df, grid_svam_col(grid, band, law=law))
            m_i = finite_xy(teff, y_i)
            m_svam = finite_xy(teff, y_svam)
            if m_i.any():
                order = np.argsort(teff[m_i])
                ax.plot(
                    teff[m_i][order], y_i[m_i][order],
                    color=style["color"], lw=1.8, alpha=0.90,
                    marker=style["marker"], ms=5, mfc=style["color"],
                    mec=style["color"], label=rf"{grid} $I(\mu)$",
                )
                top_values.append(y_i[m_i])
            if m_svam.any():
                order = np.argsort(teff[m_svam])
                ax.plot(
                    teff[m_svam][order], y_svam[m_svam][order],
                    color=style["color"], lw=1.6, ls="--", alpha=0.90,
                    marker=style["marker"], ms=5, mfc="white",
                    mec=style["color"], label=rf"{grid} SVAM",
                )
                top_values.append(y_svam[m_svam])

            rm = finite_xy(teff, y_i, y_svam) & (y_i != 0)
            if rm.any():
                residual = 100.0 * (y_i[rm] - y_svam[rm]) / y_i[rm]
                order = np.argsort(teff[rm])
                rax.plot(
                    teff[rm][order], residual[order],
                    color=style["color"], lw=1.5, alpha=0.90,
                    marker=style["marker"], ms=4.6, mfc="white",
                    mec=style["color"],
                )
                residual_values.append(residual)

        ax.set_ylabel(rf"${band}$-band power-law coefficient")
        ax.grid(True, alpha=0.25)
        rax.axhline(0.0, color="0.55", ls="--", lw=1.0)
        rax.set_xlabel(r"$T_{\rm eff}$ (K)")
        rax.set_ylabel(rf"${band}$-band SVAM offset (%)")
        rax.grid(True, alpha=0.25)

        if top_values:
            vals = np.concatenate([arr[np.isfinite(arr)] for arr in top_values if arr.size])
            if vals.size:
                lo, hi = float(vals.min()), float(vals.max())
                pad = max(0.01, 0.10 * (hi - lo)) if hi > lo else 0.02
                ax.set_ylim(lo - pad, hi + pad)
        if residual_values:
            vals = np.concatenate([arr[np.isfinite(arr)] for arr in residual_values if arr.size])
            if vals.size:
                lim = max(5.0, float(np.nanmax(np.abs(vals))) * 1.15)
                rax.set_ylim(-lim, lim)

    axes[0, 1].legend(frameon=False, fontsize=7.8, loc="best", ncol=1)
    fig.subplots_adjust(left=0.11, right=0.98, bottom=0.10, top=0.985)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def plot_powerlaw_vs_other_law_diameters(df: pd.DataFrame, *, output: Path, dpi: int) -> None:
    x = values(df, ["theta_PL", "PL_diam"])
    xe = values(df, ["theta_PL_err", "PL_diam_err", "PL_diam_err_ivw"])
    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7.35, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.25], "hspace": 0.06},
    )
    all_values: list[np.ndarray] = []
    all_residuals: list[np.ndarray] = []

    for label, ycols, yerrcols, marker, color in OTHER_LAW_DIAMETERS:
        if not col_exists(df, ycols):
            continue
        y = values(df, ycols)
        ye = values(df, yerrcols)
        mask = finite_xy(x, y)
        if not mask.any():
            continue
        xx, yy = x[mask], y[mask]
        residual = 100.0 * (yy - xx) / xx
        ax.errorbar(xx, yy, xerr=xe[mask], yerr=ye[mask], fmt="none",
                    ecolor=color, elinewidth=0.8, capsize=0, alpha=0.16, zorder=1)
        ax.scatter(xx, yy, s=46, color=color, marker=marker, edgecolors="white",
                   linewidths=0.45, label=label, alpha=0.95, zorder=3)
        tx, ty = binned_trend(xx, yy)
        ax.plot(tx, ty, color=color, lw=2.0, alpha=0.82, zorder=2)
        ax_res.scatter(xx, residual, s=40, color=color, marker=marker,
                       edgecolors="white", linewidths=0.45, alpha=0.9, zorder=3)
        rx, ry = binned_trend(xx, residual)
        ax_res.plot(rx, ry, color=color, lw=1.9, alpha=0.82, zorder=2)
        all_values.extend([xx, yy])
        all_residuals.append(residual)

    finish_diameter_axes(
        ax, ax_res, all_values, all_residuals,
        xlabel="Power-law diameter (mas)",
        ylabel="Other-law diameter (mas)",
        residual_ylabel="Other - PL (%)",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.52, 0.985), ncol=3,
              frameon=True, fancybox=False, facecolor="white", edgecolor="0.85",
              framealpha=0.92, handletextpad=0.5, columnspacing=1.1, borderaxespad=0.0)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.08, top=0.98, hspace=0.05)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def plot_mps2_vs_direct_clv_and_pl(df: pd.DataFrame, *, output: Path, dpi: int) -> None:
    x = values(df, ["theta_CLV_MPS2"])
    xe = values(df, ["theta_CLV_MPS2_err"])
    specs = [
        ("Power law", ["theta_PL", "PL_diam"], ["theta_PL_err", "PL_diam_err", "PL_diam_err_ivw"], "P", "#0072B2"),
        ("SATLAS", ["theta_CLV_SATLAS"], ["theta_CLV_SATLAS_err"], "X", "#7B3294"),
        ("Stagger", ["theta_CLV_Stagger"], ["theta_CLV_Stagger_err"], "o", "#000000"),
        ("Kurucz", ["theta_CLV_Kurucz"], ["theta_CLV_Kurucz_err"], "s", "#7F7F7F"),
        ("MPS1", ["theta_CLV_MPS1"], ["theta_CLV_MPS1_err"], "^", "#999999"),
    ]
    print("\nMPS2 direct-CLV diameter comparison")
    print("Residual definition: 100 * (comparison - MPS2) / MPS2")
    print(f"{'Comparison':<12}{'N':>5}{'median (%)':>14}{'RMS (%)':>12}{'mean (%)':>12}{'min (%)':>12}{'max (%)':>12}")

    fig, (ax, ax_res) = plt.subplots(
        2, 1, figsize=(7.4, 8.4), sharex=True,
        gridspec_kw={"height_ratios": [3.4, 1.25], "hspace": 0.05},
    )
    all_values: list[np.ndarray] = []
    all_residuals: list[np.ndarray] = []

    for label, ycols, yerrcols, marker, color in specs:
        if not col_exists(df, ycols):
            continue
        y = values(df, ycols)
        ye = values(df, yerrcols)
        mask = finite_xy(x, y)
        if not mask.any():
            continue
        xx, yy = x[mask], y[mask]
        residual = 100.0 * (yy - xx) / xx
        print(
            f"{label:<12}{residual.size:>5d}"
            f"{np.nanmedian(residual):>14.3f}"
            f"{np.sqrt(np.nanmean(residual**2)):>12.3f}"
            f"{np.nanmean(residual):>12.3f}"
            f"{np.nanmin(residual):>12.3f}"
            f"{np.nanmax(residual):>12.3f}"
        )
        ax.errorbar(xx, yy, xerr=xe[mask], yerr=ye[mask], fmt="none",
                    ecolor=color, elinewidth=0.9, alpha=0.16, zorder=1)
        ax.scatter(xx, yy, s=60, marker=marker, color=color, edgecolors="white",
                   linewidths=0.7, alpha=0.96, label=label, zorder=3)
        ax_res.scatter(xx, residual, s=54, marker=marker, color=color,
                       edgecolors="white", linewidths=0.7, alpha=0.94, zorder=3)
        all_values.extend([xx, yy])
        all_residuals.append(residual)

    finish_diameter_axes(
        ax, ax_res, all_values, all_residuals,
        xlabel="MPS2 direct CLV-fit diameter (mas)",
        ylabel="Comparison diameter (mas)",
        residual_ylabel="(Y - MPS2) / MPS2 (%)",
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.50, 0.995), ncol=2,
              frameon=True, fancybox=False, facecolor="white", edgecolor="0.82",
              framealpha=0.95, handletextpad=0.45, columnspacing=0.95, borderaxespad=0.0)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.08, top=0.98, hspace=0.05)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def plot_teff_vs_pl_minus_direct_clv(df: pd.DataFrame, *, output: Path, dpi: int) -> None:
    teff = teff_values(df)
    theta_pl = values(df, ["theta_PL", "PL_diam"])
    theta_pl_err = values(df, ["theta_PL_err", "PL_diam_err", "PL_diam_err_ivw"])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    all_residuals: list[np.ndarray] = []

    for grid, style in DIAMETER_GRID_STYLES.items():
        theta_clv = values(df, [f"theta_CLV_{grid}"])
        theta_clv_err = values(df, [f"theta_CLV_{grid}_err"])
        mask = finite_xy(teff, theta_pl, theta_clv) & (theta_clv != 0)
        if not mask.any():
            continue

        residual = 100.0 * (theta_pl[mask] - theta_clv[mask]) / theta_clv[mask]
        err = np.full(mask.sum(), np.nan)
        good_err = np.isfinite(theta_pl_err[mask]) | np.isfinite(theta_clv_err[mask])
        if good_err.any():
            pl_err = np.nan_to_num(theta_pl_err[mask], nan=0.0)
            clv_err = np.nan_to_num(theta_clv_err[mask], nan=0.0)
            err = 100.0 * np.sqrt((pl_err / theta_clv[mask]) ** 2 + ((theta_pl[mask] * clv_err) / (theta_clv[mask] ** 2)) ** 2)

        ax.errorbar(
            teff[mask], residual, yerr=err, fmt="none",
            ecolor=style["color"], elinewidth=0.7, capsize=1.5, alpha=0.16, zorder=1,
        )
        ax.scatter(
            teff[mask], residual, s=42, marker=style["marker"], color=style["color"],
            edgecolors="white", linewidths=0.45, alpha=0.90, label=grid, zorder=3,
        )
        all_residuals.append(residual)

    ax.axhline(0.0, color="0.45", ls="--", lw=1.0, zorder=0)
    if all_residuals:
        vals = np.concatenate(all_residuals)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            pad = max(0.08, 0.16 * (hi - lo)) if hi > lo else 0.2
            ax.set_ylim(lo - pad, hi + pad)

    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel(r"$(\theta_{\rm PL}-\theta_{\rm CLV}^g)/\theta_{\rm CLV}^g$ (%)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(direction="in", which="both", top=True, right=True)
    ax.legend(frameon=False, fontsize=9, loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def diameter_correction(
    numerator: np.ndarray,
    numerator_err: np.ndarray,
    reference: np.ndarray,
    reference_err: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    residual = 100.0 * (numerator - reference) / reference
    err = np.full_like(residual, np.nan, dtype=float)
    good = np.isfinite(numerator_err) | np.isfinite(reference_err)
    if good.any():
        num_err = np.nan_to_num(numerator_err, nan=0.0)
        ref_err = np.nan_to_num(reference_err, nan=0.0)
        err = 100.0 * np.sqrt(
            (num_err / reference) ** 2
            + ((numerator * ref_err) / (reference ** 2)) ** 2
        )
    return residual, err


def plot_teff_vs_ud_pl_diameter_correction(df: pd.DataFrame, *, output: Path, dpi: int) -> None:
    teff = teff_values(df)
    theta_mps2 = values(df, ["theta_CLV_MPS2"])
    theta_mps2_err = values(df, ["theta_CLV_MPS2_err"])
    theta_pl = values(df, ["theta_PL", "PL_diam"])
    theta_pl_err = values(df, ["theta_PL_err", "PL_diam_err", "PL_diam_err_ivw"])

    fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.2), sharey=True)
    panel_specs = [
        ("H", values(df, ["theta_ud_H", "H_UD_diam"]), values(df, ["theta_ud_H_err", "H_UD_diam_err", "H_UD_diam_err_ivw"])),
        ("K", values(df, ["theta_ud_K", "K_UD_diam"]), values(df, ["theta_ud_K_err", "K_UD_diam_err", "K_UD_diam_err_ivw"])),
    ]
    all_residuals: list[np.ndarray] = []

    for ax, (band, theta_ud, theta_ud_err) in zip(axes, panel_specs):
        panel_residuals: list[np.ndarray] = []
        clv_mask = finite_xy(teff, theta_ud, theta_mps2) & (theta_ud != 0)
        if clv_mask.any():
            clv_resid, clv_err = diameter_correction(
                theta_mps2[clv_mask], theta_mps2_err[clv_mask],
                theta_ud[clv_mask], theta_ud_err[clv_mask],
            )
            ax.errorbar(
                teff[clv_mask], clv_resid, yerr=clv_err, fmt="none",
                ecolor="#1F78B4", elinewidth=0.75, capsize=1.5, alpha=0.20, zorder=1,
            )
            ax.scatter(
                teff[clv_mask], clv_resid, s=45, marker="o", color="#1F78B4",
                edgecolors="white", linewidths=0.45, alpha=0.92,
                label=r"$\theta_{\rm CLV}^{\rm MPS2}$", zorder=3,
            )
            tx, ty = binned_trend(teff[clv_mask], clv_resid)
            ax.plot(tx, ty, color="#1F78B4", lw=1.9, alpha=0.82, zorder=2)
            all_residuals.append(clv_resid)
            panel_residuals.append(clv_resid)

        pl_mask = finite_xy(teff, theta_pl, theta_ud) & (theta_ud != 0)
        if pl_mask.any():
            pl_resid, pl_err = diameter_correction(
                theta_pl[pl_mask], theta_pl_err[pl_mask],
                theta_ud[pl_mask], theta_ud_err[pl_mask],
            )
            ax.errorbar(
                teff[pl_mask], pl_resid, yerr=pl_err, fmt="none",
                ecolor="#D95F02", elinewidth=0.75, capsize=1.5, alpha=0.16, zorder=1,
            )
            ax.scatter(
                teff[pl_mask], pl_resid, s=42, marker="s", color="#D95F02",
                edgecolors="white", linewidths=0.45, alpha=0.88,
                label=r"$\theta_{\rm PL}$", zorder=3,
            )
            tx, ty = binned_trend(teff[pl_mask], pl_resid)
            ax.plot(tx, ty, color="#D95F02", lw=1.8, alpha=0.75, zorder=2)
            all_residuals.append(pl_resid)
            panel_residuals.append(pl_resid)

        if panel_residuals:
            panel_vals = np.concatenate(panel_residuals)
            panel_vals = panel_vals[np.isfinite(panel_vals)]
            if panel_vals.size:
                panel_median = float(np.nanmedian(panel_vals))
                ax.axhline(panel_median, color="0.35", ls=(0, (4, 2)), lw=1.0, alpha=0.9, zorder=0)
                ax.annotate(
                    f"median = {panel_median:.1f}%",
                    xy=(0.98, panel_median), xycoords=("axes fraction", "data"),
                    xytext=(0, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=8, color="0.30",
                    bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.75),
                )

        ax.axhline(0.0, color="0.45", ls="--", lw=1.0, zorder=0)
        ax.set_title(rf"${band}$ band", fontsize=12)
        ax.set_xlabel(r"$T_{\rm eff}$ (K)")
        ax.grid(True, which="major", alpha=0.24, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.09, linewidth=0.5)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(direction="in", which="both", top=True, right=True)
        ax.legend(frameon=False, fontsize=9, loc="best")
        

    if all_residuals:
        vals = np.concatenate(all_residuals)
        vals = vals[np.isfinite(vals)]
        if vals.size:
            lo, hi = float(vals.min()), float(vals.max())
            pad = max(0.20, 0.12 * (hi - lo)) if hi > lo else 0.4
            for ax in axes:
                ax.set_ylim(lo - pad, hi + pad)

    ax.set_ylim([1.4, 5.0])
    axes[0].set_ylabel(
        r"$100(\theta-\theta_{\rm UD,b})/\theta_{\rm UD,b}$ (%)"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=DEFAULT_CSV)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--cmap", default="turbo_r")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df.columns = df.columns.str.strip()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print_percent_residual_tables(df)

    plot_hk_comparison(
        df, source="I", dpi=args.dpi, cmap_name=args.cmap,
        output=outdir / "CHARA_H_vs_K_LDCs_same_night.png",
    )
    plot_hk_comparison(
        df, source="SVAM", dpi=args.dpi, cmap_name=args.cmap,
        output=outdir / "svam_CHARA_H_vs_K_LDCs_same_night.png",
    )
    plot_teff_comparison(
        df, source="I", dpi=args.dpi, cmap_name=args.cmap,
        residual_grid="MPS2", label_targets=True,
        output=outdir / "composite_HK_all_grids_mps2_residuals_labeled.png",
    )
    plot_chara_vs_model_coeff_1to1(
        df, source="I", dpi=args.dpi, law="power",
        output=outdir / "chara_vs_model_intensity_powerlaw_1to1.png",
    )
    plot_chara_vs_model_coeff_1to1(
        df, source="I", dpi=args.dpi, law="linear",
        output=outdir / "chara_vs_model_intensity_linear_1to1.png",
    )
    if has_any_linear_svam(df):
        plot_chara_vs_model_coeff_1to1(
            df, source="SVAM", dpi=args.dpi, law="linear",
            output=outdir / "chara_vs_model_svam_linear_1to1.png",
        )
    else:
        print("[skip] linear SVAM 1:1 plot: no a_svam_* columns found in input table")
    '''
    plot_teff_comparison(
        df, source="I", dpi=args.dpi, cmap_name=args.cmap,
        residual_grid="MPS2", law="linear",
        output=outdir / "composite_HK_all_grids_linear_mps2_residuals.png",
    )
    '''
    plot_teff_comparison(
        df, source="SVAM", dpi=args.dpi, cmap_name=args.cmap,
        residual_grid="MPS2", label_targets=True,
        output=outdir / "svam_composite_HK_all_grids_mps2_residuals_labeled.png",
    )
    plot_i_vs_svam_teff_comparison(
        df, dpi=args.dpi,
        output=outdir / "I_vs_SVAM_all_grids_teff_comparison.png",
    )
    plot_powerlaw_vs_other_law_diameters(
        df, dpi=args.dpi,
        output=outdir / "PowerLaw_diameter_vs_other_laws_spherical.png",
    )
    plot_mps2_vs_direct_clv_and_pl(
        df, dpi=args.dpi,
        output=outdir / "MPS2_vs_other_direct_CLV_and_PL_diameters.png",
    )
    plot_teff_vs_pl_minus_direct_clv(
        df, dpi=args.dpi,
        output=outdir / "Teff_vs_PL_minus_direct_CLV_diameters.png",
    )
    plot_teff_vs_ud_pl_diameter_correction(
        df, dpi=args.dpi,
        output=outdir / "Teff_vs_CLV_PL_minus_UD_diameter_corrections_HK.png",
    )


if __name__ == "__main__":
    main()
