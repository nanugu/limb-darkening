#!/usr/bin/env python3
"""Plot split-band CHARA power-law coefficients versus Teff.

Inputs are the public/reproducible fit_visibility_laws*.csv files:
  - fit_visibility_laws.csv        full H and full K
  - fit_visibility_laws_Hlow.csv   lower H, full K
  - fit_visibility_laws_Hhigh.csv  higher H, full K
  - fit_visibility_laws_Klow.csv   full H, lower K
  - fit_visibility_laws_Khigh.csv  full H, higher K
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CSV_DIR = ROOT / "csv"


def default_csv_dir() -> Path:
    return CSV_DIR


def read_fit_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    required = {
        "target",
        "date",
        "teff",
        "alpha_chara_H",
        "alpha_chara_H_err",
        "alpha_chara_K",
        "alpha_chara_K_err",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {', '.join(missing)}")
    return df


def clean_target(value: object) -> str:
    return str(value).strip().replace("_", " ")


def key_series(df: pd.DataFrame) -> pd.Series:
    target = df["target"].astype(str).str.strip().str.lower()
    date = df["date"].astype(str).str.strip()
    return target + "|" + date


def fit_subset(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": key_series(df),
            f"alpha_{prefix}_H": pd.to_numeric(df["alpha_chara_H"], errors="coerce"),
            f"err_{prefix}_H": pd.to_numeric(df["alpha_chara_H_err"], errors="coerce"),
            f"alpha_{prefix}_K": pd.to_numeric(df["alpha_chara_K"], errors="coerce"),
            f"err_{prefix}_K": pd.to_numeric(df["alpha_chara_K_err"], errors="coerce"),
        }
    ).drop_duplicates("key", keep="first")


def build_split_table(csv_dir: Path) -> pd.DataFrame:
    full = read_fit_csv(csv_dir / "fit_visibility_laws.csv")
    h_low = read_fit_csv(csv_dir / "fit_visibility_laws_Hlow.csv")
    h_high = read_fit_csv(csv_dir / "fit_visibility_laws_Hhigh.csv")
    k_low = read_fit_csv(csv_dir / "fit_visibility_laws_Klow.csv")
    k_high = read_fit_csv(csv_dir / "fit_visibility_laws_Khigh.csv")

    data = pd.DataFrame(
        {
            "key": key_series(full),
            "target": full["target"].map(clean_target),
            "date": full["date"].astype(str),
            "teff": pd.to_numeric(full["teff"], errors="coerce"),
            "alpha_full_H": pd.to_numeric(full["alpha_chara_H"], errors="coerce"),
            "err_full_H": pd.to_numeric(full["alpha_chara_H_err"], errors="coerce"),
            "alpha_full_K": pd.to_numeric(full["alpha_chara_K"], errors="coerce"),
            "err_full_K": pd.to_numeric(full["alpha_chara_K_err"], errors="coerce"),
        }
    ).drop_duplicates("key", keep="first")

    for prefix, df in (
        ("Hlow", h_low),
        ("Hhigh", h_high),
        ("Klow", k_low),
        ("Khigh", k_high),
    ):
        data = data.merge(fit_subset(df, prefix), on="key", how="left")

    data = data.sort_values(["teff", "target"], ascending=[True, True]).reset_index(drop=True)
    return data


def finite_xy(df: pd.DataFrame, y_col: str, err_col: str) -> pd.DataFrame:
    cols = ["teff", y_col, err_col]
    out = df.loc[:, cols].copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[np.isfinite(out["teff"]) & np.isfinite(out[y_col])]


def residual_table(df: pd.DataFrame, value_col: str, err_col: str, full_col: str, full_err_col: str) -> pd.DataFrame:
    out = df[["teff", value_col, err_col, full_col, full_err_col]].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["delta"] = out[value_col] - out[full_col]
    out["err"] = np.sqrt(out[err_col] ** 2 + out[full_err_col] ** 2)
    return out[np.isfinite(out["teff"]) & np.isfinite(out["delta"])]


def overplot_binned_medians(ax, sub: pd.DataFrame, color: str) -> None:
    if len(sub) < 8:
        return
    bins = np.linspace(3600, 6700, 6)
    centers = []
    medians = []
    spreads = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        chunk = sub[(sub["teff"] >= lo) & (sub["teff"] < hi)]
        if len(chunk) < 3:
            continue
        centers.append(float(np.nanmedian(chunk["teff"])))
        medians.append(float(np.nanmedian(chunk["delta"])))
        q16, q84 = np.nanpercentile(chunk["delta"], [16, 84])
        spreads.append([[medians[-1] - q16], [q84 - medians[-1]]])
    if not centers:
        return
    yerr = np.array(spreads, dtype=float).reshape(len(spreads), 2).T
    ax.errorbar(
        centers,
        medians,
        yerr=yerr,
        fmt="D",
        ms=6.5,
        mfc=color,
        mec="k",
        mew=0.5,
        ecolor=color,
        elinewidth=1.5,
        capsize=3,
        zorder=5,
    )


def draw_residual_panel(ax, df: pd.DataFrame, band: str) -> None:
    if band == "H":
        series = [
            ("alpha_Hlow_H", "err_Hlow_H", "alpha_full_H", "err_full_H", "#0072B2", r"$H_{\rm low}-H_{\rm full}$"),
            ("alpha_Hhigh_H", "err_Hhigh_H", "alpha_full_H", "err_full_H", "#D55E00", r"$H_{\rm high}-H_{\rm full}$"),
        ]
        ylabel = r"$\Delta\alpha_H$"
        title = "(a) H split minus full band"
    else:
        series = [
            ("alpha_Klow_K", "err_Klow_K", "alpha_full_K", "err_full_K", "#0072B2", r"$K_{\rm low}-K_{\rm full}$"),
            ("alpha_Khigh_K", "err_Khigh_K", "alpha_full_K", "err_full_K", "#D55E00", r"$K_{\rm high}-K_{\rm full}$"),
        ]
        ylabel = r"$\Delta\alpha_K$"
        title = "(b) K split minus full band"

    for value_col, err_col, full_col, full_err_col, color, label in series:
        sub = residual_table(df, value_col, err_col, full_col, full_err_col)
        ax.errorbar(
            sub["teff"],
            sub["delta"],
            yerr=sub["err"],
            fmt="o",
            ms=4.0,
            color=color,
            alpha=0.40,
            ecolor=mpl.colors.to_rgba(color, 0.13),
            elinewidth=0.8,
            capsize=0,
            label=label,
        )
        overplot_binned_medians(ax, sub, color)

    ax.axhline(0.0, color="0.45", ls="--", lw=1.0)
    ax.set_title(title)
    ax.set_xlabel(r"$T_{\rm eff}$ (K)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.set_ylim(-0.085, 0.065)


def plot_split_figure(data: pd.DataFrame, outdir: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharex=True, sharey=False)

    draw_residual_panel(axes[0], data, "H")
    draw_residual_panel(axes[1], data, "K")

    for ax in axes:
        ax.invert_xaxis()
        ax.tick_params(direction="in", top=True, right=True)

    fig.tight_layout(w_pad=1.3)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / "CHARA_HK_split_alpha_vs_Teff.png"
    fig.savefig(outfile, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return outfile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-dir", type=Path, default=default_csv_dir())
    parser.add_argument("--outdir", type=Path, default=ROOT / "Figures")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    data = build_split_table(args.csv_dir)
    print(f"[data] matched {len(data)} full-band rows from {args.csv_dir}")
    outfile = plot_split_figure(data, args.outdir, args.dpi)
    print(f"[saved] {outfile}")


if __name__ == "__main__":
    main()
