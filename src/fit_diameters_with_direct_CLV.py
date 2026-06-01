#!/usr/bin/env python3
"""Fit real CHARA H+K V2 data with full tabulated CLV profiles from atmosphere grids.

This is a complementary analysis to the analytic limb-darkening law fits.
For each target/file pair, the H- and K-band CLV profiles from ExoTiC-LD are
held fixed for a chosen atmosphere grid, while a common angular diameter is
fit directly to the real CHARA squared visibilities.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any
from chara_fit_common import (
    add_transfer_function_params,
    append_csv_row,
    boxcar_throughput,
    iter_target_runs,
    resolve_oifits_path,
)

import numpy as np
import pmoired
import matplotlib.pyplot as plt
from exotic_ld import StellarLimbDarkening

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"
TARGET_SAMPLE_CSV = CSV_DIR / "merged_four_branches_wide.csv"
LOG_DIR_DEFAULT = "."
OUTPUT_FIT_RESULTS_CSV = CSV_DIR / "fit_diameters_with_direct_CLV.csv"
DEFAULT_DATA_DIR = os.environ.get("EXOTIC_LD_DATA_DIR", str(ROOT / "exotic_ld_data"))
OIFITS_ROOT = os.environ.get("OIFITS_DIR", str(ROOT / "oifits"))
WAVELENGTH_TABLE = [(1.5, 1.72), (2.00, 2.37)]
GRID_ORDER = ("stagger", "kurucz", "mps1", "mps2")
GRID_PUBLIC = {"stagger": "Stagger", "kurucz": "Kurucz", "mps1": "MPS1", "mps2": "MPS2", "satlas": "SATLAS"}
H_RANGE = [15000.0, 17200.0]
K_RANGE = [20000.0, 23700.0]


W_H, T_H = boxcar_throughput(*H_RANGE)
W_K, T_K = boxcar_throughput(*K_RANGE)


def _exotic_kwargs(wavelengths: np.ndarray, throughputs: np.ndarray) -> dict[str, Any]:
    return {
        "mode": "custom",
        "wavelength_range": [float(wavelengths.min()), float(wavelengths.max())],
        "custom_wavelengths": wavelengths,
        "custom_throughput": throughputs,
    }


def _compute_grid_cliv(teff: float, logg: float, feh: float, *, band: str, data_dir: str, grid: str):
    sld = StellarLimbDarkening(
        Teff=float(teff),
        logg=float(logg),
        M_H=float(feh),
        ld_model=grid,
        ld_data_path=data_dir,
        interpolate_type="trilinear",
    )
    if band.upper() == "H":
        wavelengths, throughputs = W_H, T_H
    else:
        wavelengths, throughputs = W_K, T_K
    kwargs = _exotic_kwargs(wavelengths, throughputs)
    if hasattr(sld, "compute_passband_intensity"):
        mu, intens = sld.compute_passband_intensity(**kwargs, normalize=True)
    else:
        # Older ExoTiC-LD releases do not expose the public helper; use the
        # same internal integration path and then read back the cached arrays.
        sld._integrate_I_mu(
            kwargs["wavelength_range"],
            kwargs["mode"],
            kwargs["custom_wavelengths"],
            kwargs["custom_throughput"],
        )
        mu = sld.mus.copy()
        intens = sld.I_mu.copy()
    mu = np.asarray(mu, dtype=float).reshape(-1)
    intens = np.asarray(intens, dtype=float).reshape(-1)
    n = min(mu.size, intens.size)
    mu = mu[:n]
    intens = intens[:n]
    good = np.isfinite(mu) & np.isfinite(intens) & (mu >= 0.0) & (mu <= 1.0) & (intens >= 0.0)
    mu = mu[good]
    intens = intens[good]
    order = np.argsort(mu)
    mu = mu[order]
    intens = intens[order]
    mu, idx = np.unique(mu, return_index=True)
    intens = intens[idx]
    if mu.size < 4:
        raise ValueError(f"Not enough valid CLV points for {grid} {band}")
    return mu, intens


def build_combined_clv_model(grid: str, teff: float, logg: float, feh: float, diam_seed: float, data_dir: str) -> dict[str, Any]:
    h_mu, h_i = _compute_grid_cliv(teff, logg, feh, band="H", data_dir=data_dir, grid=grid)
    k_mu, k_i = _compute_grid_cliv(teff, logg, feh, band="K", data_dir=data_dir, grid=grid)
    return {
        "diam": float(diam_seed),
        "H,_MUtab": repr(h_mu.tolist()),
        "H,_Itab": repr(h_i.tolist()),
        "H,profile": "np.interp($MU, $H,_MUtab, $H,_Itab, left=0.0, right=1.0)",
        "H,diam": "$diam",
        "H,spectrum": "($WL<1.8)",
        "K,_MUtab": repr(k_mu.tolist()),
        "K,_Itab": repr(k_i.tolist()),
        "K,profile": "np.interp($MU, $K,_MUtab, $K,_Itab, left=0.0, right=1.0)",
        "K,diam": "$diam",
        "K,spectrum": "($WL>1.8)",
    }


def _wl_flat(merged_i: dict[str, Any], w_mask: np.ndarray) -> np.ndarray:
    vis2_all = merged_i["OI_VIS2"]["all"]
    if "WL" in vis2_all:
        return np.asarray(vis2_all["WL"][w_mask, :], dtype=float).flatten()
    wl_axis = merged_i.get("WL")
    if wl_axis is None:
        return np.full(vis2_all["B/wl"][w_mask, :].size, np.nan)
    wl_axis = np.asarray(wl_axis, dtype=float).flatten()
    return np.tile(wl_axis, int(np.sum(w_mask)))


def plot_chara_v2_grid_models(
    oi,
    models: dict[str, dict[str, Any]],
    *,
    fig_name: str,
    title: str,
    v2_0: float,
) -> None:
    vmodels = {grid: pmoired.oimodels.VmodelOI(oi._merged, model) for grid, model in models.items()}
    v2_norm = max(float(v2_0), 1e-8)
    plt.close("all")
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    data_colors = {"H": "0.15", "K": "0.55", "HK": "0.35"}
    model_colors = {
        "stagger": "#0072B2",
        "kurucz": "#D55E00",
        "mps1": "#009E73",
        "mps2": "#CC79A7",
    }
    seen_data: set[str] = set()
    seen_model: set[str] = set()

    for i, merged_i in enumerate(oi._merged):
        vis2 = merged_i["OI_VIS2"]["all"]
        for name in set(vis2["NAME"]):
            w = vis2["NAME"] == name
            f = ~vis2["FLAG"][w, :].flatten()
            if not np.any(f):
                continue

            bl = np.asarray(vis2["B/wl"][w, :], dtype=float).flatten()[f]
            v2 = np.asarray(vis2["V2"][w, :], dtype=float).flatten()[f] / v2_norm
            ev2 = np.asarray(vis2["EV2"][w, :], dtype=float).flatten()[f] / v2_norm
            wl = _wl_flat(merged_i, w)[f]

            finite = np.isfinite(bl) & np.isfinite(v2) & np.isfinite(ev2)
            if not np.any(finite):
                continue
            bl, v2, ev2, wl = bl[finite], v2[finite], ev2[finite], wl[finite]

            band_masks = (
                ("H", wl < 1.8),
                ("K", wl > 1.8),
            ) if np.isfinite(wl).any() else (("HK", np.ones_like(bl, dtype=bool)),)

            for band, band_mask in band_masks:
                if not np.any(band_mask):
                    continue
                color = data_colors[band]
                data_label = f"CHARA {band} V2" if band not in seen_data else None
                for ax in (ax_lin, ax_log):
                    ax.errorbar(
                        bl[band_mask],
                        v2[band_mask],
                        yerr=ev2[band_mask],
                        fmt=".",
                        color=color,
                        ecolor=color,
                        alpha=0.25,
                        markersize=2.5,
                        elinewidth=0.6,
                        capsize=0,
                        label=data_label if ax is ax_lin else None,
                        zorder=2,
                    )
                seen_data.add(band)

            for grid, vmodel in vmodels.items():
                mod_vis2 = vmodel[i]["OI_VIS2"]["all"]
                m_bl = np.asarray(mod_vis2["B/wl"][w, :], dtype=float).flatten()[f][finite]
                m_v2 = np.asarray(mod_vis2["V2"][w, :], dtype=float).flatten()[f][finite]
                m_finite = np.isfinite(m_bl) & np.isfinite(m_v2)
                if not np.any(m_finite):
                    continue
                color = model_colors.get(grid, "0.2")
                label = f"{grid} CLV model" if grid not in seen_model else None
                for ax in (ax_lin, ax_log):
                    ax.scatter(
                        m_bl[m_finite],
                        m_v2[m_finite],
                        s=5,
                        color=color,
                        alpha=0.75,
                        linewidths=0,
                        label=label if ax is ax_lin else None,
                        zorder=3,
                    )
                seen_model.add(grid)

    ax_lin.set_ylabel(r"$V^2$")
    ax_lin.set_title("Linear V2")
    ax_log.set_yscale("log")
    ax_log.set_title("Log V2")
    for ax in (ax_lin, ax_log):
        ax.set_xlabel(r"$B/\lambda$")
        ax.set_ylim(1e-4, 1.1)
        ax.grid(True, alpha=0.25)
    ax_lin.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(fig_name, bbox_inches="tight", dpi=150)
    plt.close(fig)


def fit_clv_grid(
    mircx_path: str,
    mystic_path: str,
    *,
    grid: str,
    diam_seed: float,
    teff: float,
    logg: float,
    feh: float,
    data_dir: str,
):
    oi = pmoired.OI([mircx_path, mystic_path], verbose=False)
    oi.setupFit({
        'obs': ['V2'],
        'min relative error': {'V2': 0.03},
        'max error': {'V2': 0.2},
        'wl ranges': WAVELENGTH_TABLE,
    })
    model = build_combined_clv_model(grid, teff, logg, feh, diam_seed, data_dir)
    model = add_transfer_function_params(oi, model, v2_0=0.8)
    oi.doFit(model, verbose=1, maxfev=10000)
    best = oi.bestfit.get("best", {}).copy()
    uncer = oi.bestfit.get("uncer", {}).copy()

    
    oi.bootstrapFit(50)
    best = oi.boot['best'].copy()
    uncer  = oi.boot['uncer'].copy()
    chi2   = oi.boot['chi2']
    

    best_model = model.copy()
    best_model.update(best)

    return {
        "diam": float(best.get("diam", np.nan)),
        "diam_err": float(uncer.get("diam", np.nan)),
        "V2_0": float(best.get("V2_0", np.nan)),
        "V2_0_err": float(uncer.get("V2_0", np.nan)),
        "chi2": float(oi.bestfit.get("chi2", np.nan)),
        "model": best_model,
        "oi": oi,
    }


def run_fits(
    *,
    limit: int | None,
    grids: tuple[str, ...] = GRID_ORDER,
    output_csv: str | None = None,
):
    csv_path = output_csv or os.path.join(LOG_DIR_DEFAULT, OUTPUT_FIT_RESULTS_CSV)
    runs = list(iter_target_runs(limit=limit, target_csv=TARGET_SAMPLE_CSV))
    print(f"[fit_clv] selected pairs: {len(runs)}")
    for run in runs:
        mircx_path = resolve_oifits_path(run.mircx_file, OIFITS_ROOT)
        mystic_path = resolve_oifits_path(run.mystic_file, OIFITS_ROOT)
        print(
            f"[fit_clv] {run.target} {run.mircx_file} + {run.mystic_file} "
            f"seed={run.seed_diam:.2f}"
        )

        row = {
            "target": run.target,
            "date": run.date,
            "oifits_h": os.path.basename(mircx_path),
            "oifits_k": os.path.basename(mystic_path),
            "teff": run.teff,
            "logg": run.logg,
            "feh": run.feh,
            "mass": run.mass,
            "theta_init": run.seed_diam,
        }
        plot_oi = None
        plot_models = {}
        for grid in grids:
            try:
                fit = fit_clv_grid(
                    mircx_path,
                    mystic_path,
                    grid=grid,
                    diam_seed=run.seed_diam,
                    teff=run.teff,
                    logg=run.logg,
                    feh=run.feh,
                    data_dir=DEFAULT_DATA_DIR,
                )
                grid_name = GRID_PUBLIC.get(grid, grid)
                row[f"theta_CLV_{grid_name}"] = fit["diam"]
                row[f"theta_CLV_{grid_name}_err"] = fit["diam_err"]
                row[f"v2_0_CLV_{grid_name}"] = fit["V2_0"]
                row[f"v2_0_CLV_{grid_name}_err"] = fit["V2_0_err"]
                row[f"chi2_red_CLV_{grid_name}"] = fit["chi2"]
                plot_oi = fit["oi"] if plot_oi is None else plot_oi
                plot_models[grid] = fit["model"]
            except Exception as exc:
                print(f"[fit_clv] {run.target} {grid} failed: {exc}")
                grid_name = GRID_PUBLIC.get(grid, grid)
                row[f"theta_CLV_{grid_name}"] = np.nan
                row[f"theta_CLV_{grid_name}_err"] = np.nan
                row[f"v2_0_CLV_{grid_name}"] = np.nan
                row[f"v2_0_CLV_{grid_name}_err"] = np.nan
                row[f"chi2_red_CLV_{grid_name}"] = np.nan
                row[f"note_CLV_{grid_name}"] = str(exc)
        if plot_oi is not None and plot_models:
            fig_name = (
                f"{Path(mircx_path).stem}_{Path(mystic_path).stem}"
                "_chara_v2_grid_models_clv_direct_fit.png"
            )
            plot_chara_v2_grid_models(
                plot_oi,
                plot_models,
                fig_name=fig_name,
                title=f"{run.target} CHARA H+K V2 and CLV Grid Models",
                v2_0=next(iter(plot_models.values())).get("V2_0", 1.0),
            )
            print(f"[saved-plot] {fig_name}")
        append_csv_row(csv_path, row)
        print(f"[saved-row] {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit real CHARA H+K data with full tabulated CLV profiles.")
    parser.add_argument("--limit", type=int, default=None, help="Number of target/file pairs to run.")
    parser.add_argument("--grid", action="append", choices=list(GRID_ORDER), help="Restrict to one or more grids.")
    parser.add_argument("--output-csv", default=None, help="Override output CSV path.")
    args = parser.parse_args()
    grids = tuple(args.grid) if args.grid else GRID_ORDER
    run_fits(limit=args.limit, grids=grids, output_csv=args.output_csv)
