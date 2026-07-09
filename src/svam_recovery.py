#!/usr/bin/env python3
"""Compare grid power-law coefficients from ExoTiC I(mu) and from synthetic V2.

For each target in the merged CSV this script:

1. reads the real CHARA OIFITS sampling (MIRC-X + MYSTIC),
2. builds full H- and K-band ExoTiC intensity profiles I(mu) for a chosen grid,
3. measures power-law coefficients alpha_I from those intensity profiles using
   ExoTiC's power-1 fit for the same passband,
4. generates synthetic V2 from the full tabulated profiles using the real PL_diam,
5. refits the synthetic V2 with the standard H+K power-law model to recover alpha_V.
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pmoired
from exotic_ld import StellarLimbDarkening
from compute_exotic_coefficients import _compute_satlas_cliv, _satlas_mu_to_rosseland_mu, band_coeffs_satlas
from chara_fit_common import add_transfer_function_params, boxcar_throughput, resolve_oifits_path
from public_schema import base_metadata_from_row, clean_grid


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"

DEFAULT_MERGED_CSV = CSV_DIR / "merged_four_branches_wide.csv"
DEFAULT_OIFITS_DIR = os.environ.get("OIFITS_DIR", str(ROOT / "oifits"))
DEFAULT_OUTPUT_CSV = CSV_DIR / "svam_recovery.csv"
DEFAULT_FIG_DIR = "figs_svam_recovery_power"
DEFAULT_DATA_DIR = os.environ.get("EXOTIC_LD_DATA_DIR", str(ROOT / "exotic_ld_data"))
GRID_CHOICES = ["stagger", "kurucz", "mps1", "mps2", "satlasross", "all"]
LAW_CHOICES=['power1', 'power2']
# ExoTiC-LD expects wavelength ranges and custom throughput grids in angstroms.
# PMOIRED OIFITS wavelengths are in microns, so keep the two conventions separate.
H_RANGE = [15000.0, 17200.0]
K_RANGE = [20000.0, 23700.0]


def _sigmoid_expr(param_ref: str) -> str:
    return f"1/(1+np.exp(-{param_ref}))"


def _logit(value: float, eps: float = 1e-6) -> float:
    clipped = float(np.clip(value, eps, 1.0 - eps))
    return float(np.log(clipped / (1.0 - clipped)))


def _sigmoid_value(value: Any, default: float = np.nan) -> float:
    x = _as_float(value, default=np.nan)
    if not np.isfinite(x):
        return default
    return float(1.0 / (1.0 + np.exp(-x)))


W_H, T_H = boxcar_throughput(*H_RANGE)
W_K, T_K = boxcar_throughput(*K_RANGE)


def _as_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def _is_finite(value: Any) -> bool:
    return np.isfinite(_as_float(value))


def _safe_filename(value: Any) -> str:
    text = str(value).strip() or "unknown_target"
    keep = [char if char.isalnum() or char in "._-" else "_" for char in text]
    return "".join(keep).strip("_") or "unknown_target"


def _power_law_model(diam: float, h_alpha: float, k_alpha: float) -> dict[str, Any]:
    return {
        "diam": float(diam),
        "H,alpha": float(h_alpha),
        "K,alpha": float(k_alpha),
        "H,diam": "$diam",
        "H,profile": "$MU**$H,alpha",
        "H,spectrum": "($WL<1.8)",
        "K,diam": "$diam",
        "K,profile": "$MU**$K,alpha",
        "K,spectrum": "($WL>1.8)",
    }


def _power_law_model_fixed_diam(diam: float, h_alpha: float, k_alpha: float) -> dict[str, Any]:
    model={
        "diam": float(diam),
        "H,alpha": float(h_alpha),
        "K,alpha": float(k_alpha),
        "H,diam": "$diam",
        "H,profile": "$MU**$H,alpha",
        "H,spectrum": "($WL<1.8)",
        "K,diam": "$diam",
        "K,profile": "$MU**$K,alpha",
        "K,spectrum": "($WL>1.8)",
    }
    return model

def _power2_model_fixed_diam(diam: float, h_p1: float, h_p2: float, k_p1: float, k_p2: float) -> dict[str, Any]:
    return {
        "diam": float(diam),
        "H,logit_a1": _logit(h_p1),
        "H,logit_alpha1": _logit(h_p2),
        "K,logit_a1": _logit(k_p1),
        "K,logit_alpha1": _logit(k_p2),
        "H,a1": _sigmoid_expr("$H,logit_a1"),
        "H,alpha1": _sigmoid_expr("$H,logit_alpha1"),
        "K,a1": _sigmoid_expr("$K,logit_a1"),
        "K,alpha1": _sigmoid_expr("$K,logit_alpha1"),
        "H,diam": "$diam",
        "H,profile": "1-$H,a1*(1-($MU**$H,alpha1))",
        "H,spectrum": "($WL<1.8)",
        "K,diam": "$diam",
        "K,profile": "1-$K,a1*(1-($MU**$K,alpha1))",
        "K,spectrum": "($WL>1.8)",
    }

def _exotic_kwargs(wavelengths: np.ndarray, throughputs: np.ndarray) -> dict[str, Any]:
    return {
        "mode": "custom",
        "wavelength_range": [float(wavelengths.min()), float(wavelengths.max())],
        "custom_wavelengths": wavelengths,
        "custom_throughput": throughputs,
    }


def _compute_grid_cliv(
    teff: float,
    logg: float,
    feh: float,
    *,
    band: str,
    data_dir: str,
    grid: str,
    law: str,
) -> tuple[np.ndarray, np.ndarray, tuple[float, ...]]:
    sld = StellarLimbDarkening(Teff=float(teff), logg=float(logg), M_H=float(feh), 
                               ld_model=grid, ld_data_path=data_dir,  interpolate_type="trilinear")
    if band.upper() == "H":
        wavelengths, throughputs = W_H, T_H
    else:
        wavelengths, throughputs = W_K, T_K
    kwargs = _exotic_kwargs(wavelengths, throughputs)
    if law == "power2":
        coeffs = np.atleast_1d(np.asarray(sld.compute_power2_ld_coeffs(**kwargs, mu_min=0.0, return_sigmas=False), dtype=float))
        if coeffs.size != 2:
            raise ValueError(f"Expected 2 {law} coefficients for {grid} {band}, got {coeffs}")
        coeff_tuple = (_as_float(coeffs[0]), _as_float(coeffs[1]))
    else:
        alpha = sld.compute_power1_ld_coeffs(**kwargs, mu_min=0.0, return_sigmas=False)
        coeff_tuple = (_as_float(np.atleast_1d(alpha)[0]),)
    mu = np.asarray(sld.mus, dtype=float).reshape(-1)
    intens = np.asarray(sld.I_mu, dtype=float).reshape(-1)
    n = min(mu.size, intens.size)
    if n == 0:
        raise ValueError(f"No {grid} CLIV samples returned for band {band}")
    mu = mu[:n]
    intens = intens[:n]
    good = np.isfinite(mu) & np.isfinite(intens) & (mu >= 0.0) & (mu <= 1.0) & (intens >= 0.0)
    mu = mu[good]
    intens = intens[good]
    if mu.size < 4:
        raise ValueError(f"Not enough valid {grid} CLIV points for band {band}")
    # PMOIRED evaluates the tabulated profile with np.interp, which expects an
    # increasing x-grid. ExoTiC-LD may return mus in a different order, so sort
    # and de-duplicate here to avoid constructing an invalid interpolant.
    order = np.argsort(mu)
    mu = mu[order]
    intens = intens[order]
    mu, unique_idx = np.unique(mu, return_index=True)
    intens = intens[unique_idx]
    if mu.size < 4:
        raise ValueError(f"Not enough unique sorted {grid} CLIV points for band {band}")
    return mu, intens, coeff_tuple


def _build_stagger_profile_model(
    diam: float,
    h_mu: np.ndarray,
    h_intens: np.ndarray,
    k_mu: np.ndarray,
    k_intens: np.ndarray,
) -> dict[str, Any]:
    return {
        "H,_MUtab": repr(h_mu.tolist()),
        "H,_Itab": repr(h_intens.tolist()),
        "H,profile": "np.interp($MU, $H,_MUtab, $H,_Itab, left=0.0, right=1.0)",
        "H,diam": float(diam),
        "H,spectrum": "($WL<1.8)",
        "K,_MUtab": repr(k_mu.tolist()),
        "K,_Itab": repr(k_intens.tolist()),
        "K,profile": "np.interp($MU, $K,_MUtab, $K,_Itab, left=0.0, right=1.0)",
        "K,diam": float(diam),
        "K,spectrum": "($WL>1.8)",
    }


def _build_satlas_profile_model(
    diam_ross: float,
    h_mu: np.ndarray,
    h_intens: np.ndarray,
    k_mu: np.ndarray,
    k_intens: np.ndarray,
) -> dict[str, Any]:
    h_mu0 = float(np.nanmin(h_mu))
    k_mu0 = float(np.nanmin(k_mu))
    return {
        "DIAM H+K": float(diam_ross),
        "H,ROSSDIAM": "$DIAM H+K",
        "K,ROSSDIAM": "$DIAM H+K",
        "H,mu0": h_mu0,
        "K,mu0": k_mu0,
        "H,_MUtab": repr(h_mu.tolist()),
        "H,_Itab": repr(h_intens.tolist()),
        "H,diam": "$H,ROSSDIAM/(1-$H,mu0**2)**0.5",
        "H,profile": "np.interp($MU, $H,_MUtab, $H,_Itab, left=0.0, right=1.0)*($MU>=$H,mu0)",
        "H,spectrum": "($WL<1.8)",
        "K,_MUtab": repr(k_mu.tolist()),
        "K,_Itab": repr(k_intens.tolist()),
        "K,diam": "$K,ROSSDIAM/(1-$K,mu0**2)**0.5",
        "K,profile": "np.interp($MU, $K,_MUtab, $K,_Itab, left=0.0, right=1.0)*($MU>=$K,mu0)",
        "K,spectrum": "($WL>1.8)",
    }


def _law_model_fixed_diam(diam: float, *, law: str, h_coeffs: tuple[float, ...], k_coeffs: tuple[float, ...]) -> dict[str, Any]:
    if law == "power2":
        return _power2_model_fixed_diam(diam, h_coeffs[0], h_coeffs[1], k_coeffs[0], k_coeffs[1])
    return _power_law_model_fixed_diam(diam, h_coeffs[0], k_coeffs[0])


def _add_v2_transfer_function_params(oi: pmoired.OI, model: dict[str, Any]) -> dict[str, Any]:
    out = dict(model)
    for data in oi.data:
        for base in data.get("OI_VIS2", {}):
            out[f"#TF_V2_{base}_*"] = "$V2_0"
    out["V2_0"] = 1.0
    return out


def _setup_v2_fit(oi: pmoired.OI, min_rel_v2: float) -> None:
    oi.setupFit({"obs": ["V2"], "min relative error": {"V2": float(min_rel_v2)}})


def _replace_v2_with_model(oi: pmoired.OI, input_model: dict[str, Any]) -> None:
    model_v2 = pmoired.oimodels.VmodelOI(oi.data, input_model)
    for data_idx, data in enumerate(oi.data):
        for base, vis2 in data.get("OI_VIS2", {}).items():
            synthetic = model_v2[data_idx]["OI_VIS2"][base]["V2"]
            vis2["V2"] = np.asarray(synthetic, dtype=float).copy()


def _collect_band_v2_points(
    oi: pmoired.OI,
    model_v2: list[dict[str, Any]],
    *,
    band: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xvals: list[np.ndarray] = []
    yvals: list[np.ndarray] = []
    evals: list[np.ndarray] = []
    mvals: list[np.ndarray] = []
    want_h = band.upper() == "H"

    for data_idx, data in enumerate(oi.data):
        wl = np.asarray(data.get("WL", []), dtype=float)
        if wl.size == 0:
            continue
        is_h = np.nanmedian(wl) < 1.8
        if is_h != want_h:
            continue
        for base, vis2 in data.get("OI_VIS2", {}).items():
            y = np.asarray(vis2["V2"], dtype=float)
            e = np.asarray(vis2.get("EV2", np.full_like(y, np.nan)), dtype=float)
            x = np.asarray(vis2["B/wl"], dtype=float)
            flag = np.asarray(vis2.get("FLAG", np.zeros_like(y, dtype=bool)), dtype=bool)
            model_y = np.asarray(model_v2[data_idx]["OI_VIS2"][base]["V2"], dtype=float)
            good = (~flag) & np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & np.isfinite(model_y) & (y > 0)
            if np.any(good):
                xvals.append(x[good]); yvals.append(y[good]); evals.append(e[good]); mvals.append(model_y[good])

    if not xvals:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty
    return np.concatenate(xvals), np.concatenate(yvals), np.concatenate(evals), np.concatenate(mvals)


def _band_residual_metrics(
    oi: pmoired.OI,
    model: dict[str, Any],
    *,
    band: str,
) -> dict[str, float]:
    model_v2 = pmoired.oimodels.VmodelOI(oi.data, model)
    x, y, e, model_y = _collect_band_v2_points(oi, model_v2, band=band)
    if x.size == 0:
        return {"n": 0.0, "wrms": np.nan, "median_abs": np.nan, "max_abs": np.nan}
    resid = y - model_y
    good_e = np.isfinite(e) & (e > 0)
    wrms = float(np.sqrt(np.mean((resid[good_e] / e[good_e]) ** 2))) if np.any(good_e) else np.nan
    return {
        "n": float(x.size),
        "wrms": wrms,
        "median_abs": float(np.nanmedian(np.abs(resid))),
        "max_abs": float(np.nanmax(np.abs(resid))),
    }


def _count_v2_samples(oi: pmoired.OI) -> tuple[int, int]:
    n_h = n_k = 0
    for data in oi.data:
        wl = np.asarray(data.get("WL", []), dtype=float)
        if wl.size == 0:
            continue
        is_h = np.nanmedian(wl) < 1.8
        for vis2 in data.get("OI_VIS2", {}).values():
            flag = np.asarray(vis2.get("FLAG", np.zeros_like(vis2["V2"], dtype=bool)), dtype=bool)
            good = (~flag) & np.isfinite(vis2["V2"]) & np.isfinite(vis2.get("EV2", vis2["V2"]))
            if is_h:
                n_h += int(np.sum(good))
            else:
                n_k += int(np.sum(good))
    return n_h, n_k


def _save_oi_show_png(
    oi: pmoired.OI,
    model: dict[str, Any],
    png_path: str,
    *,
    logV: bool = True,
) -> None:
    plt.figure(0, figsize=(12.0, 7.0))
    oi.show(model, logV=1, allInOne=0, showUV=0, fig=0, spectro=False, showChi2=True)
    plt.savefig(png_path, bbox_inches="tight", dpi=140)
    plt.close(0)


def _collect_band_data_points(
    oi: pmoired.OI,
    *,
    band: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xvals: list[np.ndarray] = []
    yvals: list[np.ndarray] = []
    evals: list[np.ndarray] = []
    want_h = band.upper() == "H"
    for data in oi.data:
        wl = np.asarray(data.get("WL", []), dtype=float)
        if wl.size == 0:
            continue
        is_h = np.nanmedian(wl) < 1.8
        if is_h != want_h:
            continue
        for vis2 in data.get("OI_VIS2", {}).values():
            y = np.asarray(vis2["V2"], dtype=float)
            e = np.asarray(vis2.get("EV2", np.full_like(y, np.nan)), dtype=float)
            x = np.asarray(vis2["B/wl"], dtype=float)
            flag = np.asarray(vis2.get("FLAG", np.zeros_like(y, dtype=bool)), dtype=bool)
            good = (~flag) & np.isfinite(x) & np.isfinite(y) & np.isfinite(e) & (y > 0)
            if np.any(good):
                xvals.append(x[good])
                yvals.append(y[good])
                evals.append(e[good])
    if not xvals:
        empty = np.array([], dtype=float)
        return empty, empty, empty
    return np.concatenate(xvals), np.concatenate(yvals), np.concatenate(evals)


def _plot_real_chara_v2_png(
    oi_real: pmoired.OI,
    target_name: str,
    png_path: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), sharex="col")
    band_info = [("H", "H-band (MIRC-X)"), ("K", "K-band (MYSTIC)")]
    for col, (band, title) in enumerate(band_info):
        x, y, e = _collect_band_data_points(oi_real, band=band)
        if x.size == 0:
            continue
        order = np.argsort(x)
        for row in range(2):
            ax = axes[row, col]
            ax.errorbar(
                x[order], y[order], yerr=e[order], fmt=".",
                markersize=3.0, linewidth=0.4, elinewidth=0.4,
                color="0.15", ecolor="0.65", alpha=0.8,
            )
            ax.grid(True, which="both", alpha=0.25)
        axes[0, col].set_title(title)
        axes[0, col].set_yscale("linear")
        axes[1, col].set_yscale("log")
        axes[1, col].set_xlabel(r"$B_\mathrm{max}/\lambda$ (M$\lambda$)")
        axes[0, col].text(0.04, 0.06, f"N={x.size}", transform=axes[0, col].transAxes, ha="left", va="bottom", fontsize=9)
    axes[0, 0].set_ylabel(r"$V^2$")
    axes[1, 0].set_ylabel(r"$V^2$")
    fig.suptitle(f"{target_name}: real CHARA H/K-band V$^2$ sampling", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _plot_v2_fit_png(
    oi: pmoired.OI,
    exact_model: dict[str, Any],
    alpha_i_model: dict[str, Any],
    recovered_model: dict[str, Any],
    target_name: str,
    png_path: str,
    *,
    diam_input: float,
    h_alpha_i: float,
    k_alpha_i: float,
    best: dict[str, Any],
) -> None:
    exact_v2 = pmoired.oimodels.VmodelOI(oi.data, exact_model)
    alpha_i_v2 = pmoired.oimodels.VmodelOI(oi.data, alpha_i_model)
    recovered_v2 = pmoired.oimodels.VmodelOI(oi.data, recovered_model)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    band_info = [
        ("H", "H-band (MIRC-X)", h_alpha_i, _as_float(best.get("H,alpha")), "tab:blue"),
        ("K", "K-band (MYSTIC)", k_alpha_i, _as_float(best.get("K,alpha")), "tab:orange"),
    ]
    for ax, (band, title, alpha_i, alpha_v, color) in zip(axes, band_info):
        x, y, e, exact_y = _collect_band_v2_points(oi, exact_v2, band=band)
        if x.size == 0:
            continue
        _, _, _, alpha_i_y = _collect_band_v2_points(oi, alpha_i_v2, band=band)
        _, _, _, recovered_y = _collect_band_v2_points(oi, recovered_v2, band=band)
        order = np.argsort(x)
        ax.errorbar(x, y, yerr=e, fmt=".", markersize=3.0, linewidth=0.4, elinewidth=0.4, color="0.35", ecolor="0.70", alpha=0.75, zorder=2)
        ax.plot(x[order], exact_y[order], ".", color="0.05", markersize=2.8, alpha=0.95, zorder=5, label="exact Stagger synthetic V$^2$")
        ax.plot(x[order], alpha_i_y[order], ".", color=color, markersize=2.1, alpha=0.90, zorder=4, label=fr"power-law from I($\mu$): $\alpha_I$={alpha_i:.4f}")
        ax.plot(x[order], recovered_y[order], ".", color="tab:red", markersize=2.1, alpha=0.90, zorder=3, label=fr"recovered power-law: $\alpha_V$={alpha_v:.4f}")
        ax.set_title(title)
        ax.set_xlabel(r"$B_\mathrm{max}/\lambda$ (M$\lambda$)")
        ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        ax.text(0.04, 0.04, f"alpha_I={alpha_i:.4f}\nalpha_V={alpha_v:.4f}", transform=ax.transAxes, ha="left", va="bottom", fontsize=9)
        ax.legend(loc="lower left", fontsize=7)
    axes[0].set_ylabel(r"$V^2$")
    fig.suptitle(f"{target_name}: full Stagger I(mu) -> synthetic V2 -> power-law recovery; diam={diam_input:.4f} mas", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _plot_intensity_png(
    target_name: str,
    png_path: str,
    *,
    h_mu: np.ndarray,
    h_intens: np.ndarray,
    k_mu: np.ndarray,
    k_intens: np.ndarray,
    h_coeffs_i: tuple[float, ...],
    k_coeffs_i: tuple[float, ...],
    h_coeffs_v: tuple[float, ...],
    k_coeffs_v: tuple[float, ...],
    law: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    band_info = [
        ("H-band", h_mu, h_intens, h_coeffs_i, h_coeffs_v, "tab:blue"),
        ("K-band", k_mu, k_intens, k_coeffs_i, k_coeffs_v, "tab:blue"),
    ]
    for ax, (title, mu, intens, coeffs_i, coeffs_v, color) in zip(axes, band_info):
        order = np.argsort(mu)
        mu = mu[order]
        intens = intens[order]
        mu_grid = np.linspace(max(1e-4, np.nanmin(mu)), 1.0, 300)
        ax.plot(mu, intens, color="0.20", linewidth=2.0, label="Stagger I(mu)")
        if law == "power2":
            i_curve = 1.0 - coeffs_i[0] * (1.0 - np.power(mu_grid, coeffs_i[1]))
            v_curve = 1.0 - coeffs_v[0] * (1.0 - np.power(mu_grid, coeffs_v[1]))
            i_label = fr"power-2 fit to I($\mu$): p1={coeffs_i[0]:.4f}, p2={coeffs_i[1]:.4f}"
            v_label = fr"power-2 fit to V$^2$: p1={coeffs_v[0]:.4f}, p2={coeffs_v[1]:.4f}"
        else:
            i_curve = np.power(mu_grid, coeffs_i[0])
            v_curve = np.power(mu_grid, coeffs_v[0])
            i_label = fr"power-law fit to I($\mu$): $\alpha_I$={coeffs_i[0]:.4f}"
            v_label = fr"power-law fit to V$^2$: $\alpha_V$={coeffs_v[0]:.4f}"
        ax.plot(mu_grid, i_curve, color=color, linestyle="--", linewidth=2.0, label=i_label)
        ax.plot(mu_grid, v_curve, color="tab:red", linestyle=":", linewidth=2.4, label=v_label)
        ax.set_title(title)
        ax.set_xlabel(r"$\mu$")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel(r"$I(\mu)/I(1)$")
    fig.suptitle(f"{target_name}: Stagger intensity vs power-law projections", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def _scan_alpha_mismatch(
    oi: pmoired.OI,
    diam: float,
    *,
    band: str,
    alpha_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    wrms = np.full(alpha_grid.shape, np.nan, dtype=float)
    max_abs = np.full(alpha_grid.shape, np.nan, dtype=float)
    for i, alpha in enumerate(alpha_grid):
        if band.upper() == "H":
            model = _power_law_model_fixed_diam(diam, float(alpha), 0.2)
        else:
            model = _power_law_model_fixed_diam(diam, 0.2, float(alpha))
        metrics = _band_residual_metrics(oi, model, band=band)
        wrms[i] = metrics["wrms"]
        max_abs[i] = metrics["max_abs"]
    return alpha_grid, wrms, max_abs


def _plot_alpha_scan_png(
    oi: pmoired.OI,
    target_name: str,
    png_path: str,
    *,
    diam_input: float,
    h_alpha_i: float,
    k_alpha_i: float,
    h_alpha_v: float,
    k_alpha_v: float,
) -> dict[str, float]:
    alpha_grid = np.linspace(0.0, 0.5, 201)
    h_alpha_grid, h_wrms, h_max = _scan_alpha_mismatch(oi, diam_input, band="H", alpha_grid=alpha_grid)
    k_alpha_grid, k_wrms, k_max = _scan_alpha_mismatch(oi, diam_input, band="K", alpha_grid=alpha_grid)

    h_best_idx = int(np.nanargmin(h_wrms)) if np.any(np.isfinite(h_wrms)) else 0
    k_best_idx = int(np.nanargmin(k_wrms)) if np.any(np.isfinite(k_wrms)) else 0
    h_best_alpha = float(h_alpha_grid[h_best_idx])
    k_best_alpha = float(k_alpha_grid[k_best_idx])

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.0), sharex="col")
    band_rows = [
        ("H-band", h_alpha_grid, h_wrms, h_max, h_alpha_i, h_alpha_v, h_best_alpha),
        ("K-band", k_alpha_grid, k_wrms, k_max, k_alpha_i, k_alpha_v, k_best_alpha),
    ]
    for row, (title, grid, wrms, max_abs, alpha_i, alpha_v, alpha_best) in enumerate(band_rows):
        ax0 = axes[row, 0]
        ax1 = axes[row, 1]
        ax0.plot(grid, wrms, color="tab:blue", lw=2)
        ax1.plot(grid, max_abs, color="tab:orange", lw=2)
        for ax in (ax0, ax1):
            ax.axvline(alpha_i, color="0.25", ls="--", lw=1.3, label=fr"$\alpha_I$={alpha_i:.4f}")
            ax.axvline(alpha_v, color="tab:red", ls=":", lw=1.6, label=fr"$\alpha_V$={alpha_v:.4f}")
            ax.axvline(alpha_best, color="tab:green", ls="-.", lw=1.4, label=fr"best scan={alpha_best:.4f}")
            ax.grid(True, alpha=0.25)
            ax.set_title(title)
            ax.set_yscale("log")
        ax0.set_ylabel("WRMS residual")
        ax1.set_ylabel(r"max $|V^2_\mathrm{syn}-V^2_\alpha|$")
    axes[1, 0].set_xlabel(r"power-law $\alpha$")
    axes[1, 1].set_xlabel(r"power-law $\alpha$")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        axes[0, 1].legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle(f"{target_name}: fixed-diameter V$^2$ mismatch scan versus power-law $\\alpha$", fontsize=11)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight", dpi=140)
    plt.close(fig)
    return {
        "H_alpha_scan_best": h_best_alpha,
        "K_alpha_scan_best": k_best_alpha,
        "H_alpha_scan_best_wrms": float(h_wrms[h_best_idx]) if np.any(np.isfinite(h_wrms)) else np.nan,
        "K_alpha_scan_best_wrms": float(k_wrms[k_best_idx]) if np.any(np.isfinite(k_wrms)) else np.nan,
    }


def _fit_target(
    target_name: str,
    mircx_path: str,
    mystic_path: str,
    *,
    diam_input: float,
    teff: float,
    logg: float,
    feh: float,
    mass: float | None,
    grid: str,
    law: str,
    data_dir: str,
    min_rel_v2: float,
    maxfev: int,
    fig_dir: str | None,
) -> dict[str, Any]:
    if grid == "satlasross":
        h_mu_native, h_intens = _compute_satlas_cliv(teff, logg, band="H", mass=mass)
        k_mu_native, k_intens = _compute_satlas_cliv(teff, logg, band="K", mass=mass)
        h_mu = _satlas_mu_to_rosseland_mu(teff, logg, mass, h_mu_native)
        k_mu = _satlas_mu_to_rosseland_mu(teff, logg, mass, k_mu_native)

        h_good = np.isfinite(h_mu) & np.isfinite(h_intens) & (h_mu >= 0.0) & (h_mu <= 1.0) & (h_intens >= 0.0)
        k_good = np.isfinite(k_mu) & np.isfinite(k_intens) & (k_mu >= 0.0) & (k_mu <= 1.0) & (k_intens >= 0.0)
        h_mu = np.asarray(h_mu[h_good], dtype=float)
        h_intens = np.asarray(h_intens[h_good], dtype=float)
        k_mu = np.asarray(k_mu[k_good], dtype=float)
        k_intens = np.asarray(k_intens[k_good], dtype=float)

        h_order = np.argsort(h_mu)
        k_order = np.argsort(k_mu)
        h_mu = h_mu[h_order]
        h_intens = h_intens[h_order]
        k_mu = k_mu[k_order]
        k_intens = k_intens[k_order]
        h_mu, h_idx = np.unique(h_mu, return_index=True)
        h_intens = h_intens[h_idx]
        k_mu, k_idx = np.unique(k_mu, return_index=True)
        k_intens = k_intens[k_idx]
        if h_mu.size < 4 or k_mu.size < 4:
            raise ValueError("Not enough valid SATLASRoss CLIV points for synthetic visibility generation")

        h_satlas = band_coeffs_satlas(teff=teff, logg=logg, band_label="H", mass=mass, mu_min=0.0, use_rosseland_mu=True)
        k_satlas = band_coeffs_satlas(teff=teff, logg=logg, band_label="K", mass=mass, mu_min=0.0, use_rosseland_mu=True)
        h_coeffs_i = tuple(float(x) for x in np.asarray(h_satlas[law]["coeffs"], dtype=float))
        k_coeffs_i = tuple(float(x) for x in np.asarray(k_satlas[law]["coeffs"], dtype=float))
        input_model = _build_satlas_profile_model(diam_input, h_mu, h_intens, k_mu, k_intens)
    else:
        h_mu, h_intens, h_coeffs_i = _compute_grid_cliv(teff, logg, feh, band="H", data_dir=data_dir, grid=grid, law=law)
        k_mu, k_intens, k_coeffs_i = _compute_grid_cliv(teff, logg, feh, band="K", data_dir=data_dir, grid=grid, law=law)
        input_model = _build_stagger_profile_model(diam_input, h_mu, h_intens, k_mu, k_intens)

    oi = pmoired.OI([mircx_path, mystic_path], verbose=False)
    oi_real = copy.deepcopy(oi)
    _setup_v2_fit(oi, min_rel_v2)
    #oi.show(logV=1, allInOne=0, showUV=0, fig=0, spectro=False)
    #plt.savefig('V2_real.png',)

    _replace_v2_with_model(oi, input_model)
    #oi.show(logV=1, allInOne=0, showUV=0, fig=0, spectro=False)
    #plt.savefig('V2_synthetic.png')

    fit_model = _law_model_fixed_diam(diam_input, law=law, h_coeffs=h_coeffs_i, k_coeffs=k_coeffs_i)
    fit_model_with_V2_0 = add_transfer_function_params(oi, fit_model, v2_0=0.8)
    oi.doFit(fit_model_with_V2_0, verbose=1, maxfev=maxfev)
    best = oi.bestfit.get("best", {})
    uncer = oi.bestfit.get("uncer", {})

    #oi.bootstrapFit(50, verbose=1)
    #best = oi.boot['best'].copy()
    #uncer  = oi.boot['uncer'].copy()
        
    n_h, n_k = _count_v2_samples(oi)
    alpha_i_model = _law_model_fixed_diam(diam_input, law=law, h_coeffs=h_coeffs_i, k_coeffs=k_coeffs_i)
    h_input_metrics = _band_residual_metrics(oi, input_model, band="H")
    k_input_metrics = _band_residual_metrics(oi, input_model, band="K")
    h_alpha_i_metrics = _band_residual_metrics(oi, alpha_i_model, band="H")
    k_alpha_i_metrics = _band_residual_metrics(oi, alpha_i_model, band="K")

    synthetic_png = ""
    intensity_png = ""
    synthetic_show_png = ""
    synthetic_show_recovered_png = ""
    real_chara_png = ""
    alpha_scan_png = ""
    alpha_scan = {
        "H_alpha_scan_best": np.nan,
        "K_alpha_scan_best": np.nan,
        "H_alpha_scan_best_wrms": np.nan,
        "K_alpha_scan_best_wrms": np.nan,
    }
    '''
    if fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        synthetic_png = os.path.join(fig_dir, f"{_safe_filename(target_name)}_{grid}_synthetic_hk_powerlaw_fit.png")
        intensity_png = os.path.join(fig_dir, f"{_safe_filename(target_name)}_{grid}_synthetic_hk_intensity_match.png")
        synthetic_show_png = os.path.join(fig_dir, f"{_safe_filename(target_name)}_{grid}_synthetic_hk_oi_show_input.png")
        synthetic_show_recovered_png = os.path.join(fig_dir, f"{_safe_filename(target_name)}_{grid}_synthetic_hk_oi_show_recovered.png")
        real_chara_png = os.path.join(fig_dir, f"{_safe_filename(target_name)}_real_chara_hk_v2_data.png")
        alpha_scan_png = os.path.join(fig_dir, f"{_safe_filename(target_name)}_{grid}_alpha_scan.png")
        if law == "power2":
            h_coeffs_v = (_sigmoid_value(best.get("H,logit_a1")), _sigmoid_value(best.get("H,logit_alpha1")))
            k_coeffs_v = (_sigmoid_value(best.get("K,logit_a1")), _sigmoid_value(best.get("K,logit_alpha1")))
        else:
            h_coeffs_v = (_as_float(best.get("H,alpha")),)
            k_coeffs_v = (_as_float(best.get("K,alpha")),)
        recovered_model = _law_model_fixed_diam(diam_input, law=law, h_coeffs=h_coeffs_v, k_coeffs=k_coeffs_v)
        _plot_real_chara_v2_png(oi_real, target_name, real_chara_png)
        _save_oi_show_png(oi, input_model, synthetic_show_png, logV=True)
        _save_oi_show_png(oi, recovered_model, synthetic_show_recovered_png, logV=True)
        if law == "power1":
            alpha_scan = _plot_alpha_scan_png(
                oi,
                target_name,
                alpha_scan_png,
                diam_input=diam_input,
                h_alpha_i=h_coeffs_i[0],
                k_alpha_i=k_coeffs_i[0],
                h_alpha_v=_as_float(best.get("H,alpha")),
                k_alpha_v=_as_float(best.get("K,alpha")),
            )
        else:
            alpha_scan_png = ""
        _plot_v2_fit_png(
            oi,
            input_model,
            alpha_i_model,
            recovered_model,
            target_name,
            synthetic_png,
            diam_input=diam_input,
            h_alpha_i=h_coeffs_i[0],
            k_alpha_i=k_coeffs_i[0],
            best=best,
        )
        _plot_intensity_png(
            target_name,
            intensity_png,
            h_mu=h_mu,
            h_intens=h_intens,
            k_mu=k_mu,
            k_intens=k_intens,
            h_coeffs_i=h_coeffs_i,
            k_coeffs_i=k_coeffs_i,
            h_coeffs_v=h_coeffs_v,
            k_coeffs_v=k_coeffs_v,
            law=law,
        )
    '''
    if law == "power2":
        h_p1_i, h_p2_i = h_coeffs_i
        k_p1_i, k_p2_i = k_coeffs_i
        h_p1_v = _sigmoid_value(best.get("H,logit_a1"))
        h_p2_v = _sigmoid_value(best.get("H,logit_alpha1"))
        k_p1_v = _sigmoid_value(best.get("K,logit_a1"))
        k_p2_v = _sigmoid_value(best.get("K,logit_alpha1"))
        # The formal uncertainties returned by the optimizer live in logit-space.
        # Leave transformed coefficient errors blank rather than misreporting them.
        h_p1_err = np.nan
        h_p2_err = np.nan
        k_p1_err = np.nan
        k_p2_err = np.nan
    else:
        h_p1_i, h_p2_i = h_coeffs_i[0], np.nan
        k_p1_i, k_p2_i = k_coeffs_i[0], np.nan
        h_p1_v, h_p2_v = _as_float(best.get("H,alpha")), np.nan
        k_p1_v, k_p2_v = _as_float(best.get("K,alpha")), np.nan
        h_p1_err, h_p2_err = _as_float(uncer.get("H,alpha")), np.nan
        k_p1_err, k_p2_err = _as_float(uncer.get("K,alpha")), np.nan

    return {
        "synthetic_mode": f"{grid}_{law}_full_profile_real_uvwl_real_EV2",
        "input_coeff_source": (
            f"satlas_{law}_fit_to_ImuRoss"
            if grid == "satlasross"
            else f"exotic_ld_{grid}_{law}_fit_to_Imu"
        ),
        "grid_model": grid,
        "law_model": law,
        "PL_diam_input": diam_input,
        "H_PL_alpha_input": h_p1_i,
        "K_PL_alpha_input": k_p1_i,
        "H_PL_alpha_input_exotic": h_p1_i,
        "K_PL_alpha_input_exotic": k_p1_i,
        "H_PL_alpha_fit": h_p1_v,
        "H_PL_alpha_fit_err": h_p1_err,
        "K_PL_alpha_fit": k_p1_v,
        "K_PL_alpha_fit_err": k_p1_err,
        "H_PL2_alpha_input": h_p2_i,
        "K_PL2_alpha_input": k_p2_i,
        "H_PL2_alpha_fit": h_p2_v,
        "K_PL2_alpha_fit": k_p2_v,
        "H_PL2_alpha_fit_err": h_p2_err,
        "K_PL2_alpha_fit_err": k_p2_err,
        "PL2_diam_fit": diam_input,
        "PL2_V2_fit": 1.0,
        "PL_diam_fit": diam_input,
        "PL_diam_fit_err": 0.0,
        "PL_V2_fit": 1.0,
        "PL_V2_fit_err": 0.0,
        "chi2_PL_fit": _as_float(oi.bestfit.get("chi2")),
        "n_v2_H": n_h,
        "n_v2_K": n_k,
        "synthetic_png": synthetic_png,
        "synthetic_intensity_png": intensity_png,
        "synthetic_show_png": synthetic_show_png,
        "synthetic_show_recovered_png": synthetic_show_recovered_png,
        "real_chara_png": real_chara_png,
        "alpha_scan_png": alpha_scan_png,
        "H_alpha_scan_best": alpha_scan["H_alpha_scan_best"],
        "K_alpha_scan_best": alpha_scan["K_alpha_scan_best"],
        "H_alpha_scan_best_wrms": alpha_scan["H_alpha_scan_best_wrms"],
        "K_alpha_scan_best_wrms": alpha_scan["K_alpha_scan_best_wrms"],
        "H_input_wrms": h_input_metrics["wrms"],
        "K_input_wrms": k_input_metrics["wrms"],
        "H_alphaI_wrms": h_alpha_i_metrics["wrms"],
        "K_alphaI_wrms": k_alpha_i_metrics["wrms"],
        "H_alphaI_max_abs": h_alpha_i_metrics["max_abs"],
        "K_alphaI_max_abs": k_alpha_i_metrics["max_abs"],
        "synthetic_status": "ok",
    }


as_float = _as_float
is_finite = _is_finite
fit_svam_target = _fit_target


def _make_output_row(input_row: pd.Series, fit: dict[str, Any], *, mircx_path: str, mystic_path: str) -> dict[str, Any]:
    out = copy.deepcopy(input_row.to_dict())
    out.update({
        "synthetic_mode": fit["synthetic_mode"],
        "input_coeff_source": fit["input_coeff_source"],
        "grid_model": fit["grid_model"],
        "law_model": fit["law_model"],
        "source_MIRCX_file": os.path.basename(mircx_path),
        "source_MYSTIC_file": os.path.basename(mystic_path),
        "synthetic_png": fit["synthetic_png"],
        "synthetic_intensity_png": fit["synthetic_intensity_png"],
        "synthetic_show_png": fit["synthetic_show_png"],
        "synthetic_show_recovered_png": fit["synthetic_show_recovered_png"],
        "real_chara_png": fit["real_chara_png"],
        "alpha_scan_png": fit["alpha_scan_png"],
        "synthetic_status": fit["synthetic_status"],
        "PL_diam_input": fit["PL_diam_input"],
        "H_PL_alpha_input": fit["H_PL_alpha_input"],
        "K_PL_alpha_input": fit["K_PL_alpha_input"],
        "H_PL_alpha_input_exotic": fit["H_PL_alpha_input_exotic"],
        "K_PL_alpha_input_exotic": fit["K_PL_alpha_input_exotic"],
        "H_PL_alpha_Imu": fit["H_PL_alpha_input"],
        "K_PL_alpha_Imu": fit["K_PL_alpha_input"],
        "H_PL2_alpha_input": fit["H_PL2_alpha_input"],
        "K_PL2_alpha_input": fit["K_PL2_alpha_input"],
        "H_PL2_alpha_fit": fit["H_PL2_alpha_fit"],
        "K_PL2_alpha_fit": fit["K_PL2_alpha_fit"],
        "H_PL2_alpha_fit_err": fit["H_PL2_alpha_fit_err"],
        "K_PL2_alpha_fit_err": fit["K_PL2_alpha_fit_err"],
        "PL2_diam_fit": fit["PL2_diam_fit"],
        "PL2_V2_fit": fit["PL2_V2_fit"],
        "PL_diam_fit": fit["PL_diam_fit"],
        "PL_diam_fit_err": fit["PL_diam_fit_err"],
        "H_PL_alpha_fit": fit["H_PL_alpha_fit"],
        "H_PL_alpha_fit_err": fit["H_PL_alpha_fit_err"],
        "K_PL_alpha_fit": fit["K_PL_alpha_fit"],
        "K_PL_alpha_fit_err": fit["K_PL_alpha_fit_err"],
        "PL_V2_fit": fit["PL_V2_fit"],
        "PL_V2_fit_err": fit["PL_V2_fit_err"],
        "chi2_PL_fit": fit["chi2_PL_fit"],
        "n_v2_H": fit["n_v2_H"],
        "n_v2_K": fit["n_v2_K"],
        "H_input_wrms": fit["H_input_wrms"],
        "K_input_wrms": fit["K_input_wrms"],
        "H_alpha_scan_best": fit["H_alpha_scan_best"],
        "K_alpha_scan_best": fit["K_alpha_scan_best"],
        "H_alpha_scan_best_wrms": fit["H_alpha_scan_best_wrms"],
        "K_alpha_scan_best_wrms": fit["K_alpha_scan_best_wrms"],
        "H_alphaI_wrms": fit["H_alphaI_wrms"],
        "K_alphaI_wrms": fit["K_alphaI_wrms"],
        "H_alphaI_max_abs": fit["H_alphaI_max_abs"],
        "K_alphaI_max_abs": fit["K_alphaI_max_abs"],
    })
    out["PL_diam"] = fit["PL_diam_fit"]
    out["PL_diam_err_ivw"] = fit["PL_diam_fit_err"]
    out["H_PL_alpha"] = fit["H_PL_alpha_fit"]
    out["H_PL_alpha_err_ivw"] = fit["H_PL_alpha_fit_err"]
    out["K_PL_alpha"] = fit["K_PL_alpha_fit"]
    out["K_PL_alpha_err_ivw"] = fit["K_PL_alpha_fit_err"]
    out["PL_V2"] = fit["PL_V2_fit"]
    out["chi2_PL"] = fit["chi2_PL_fit"]
    return out


def _make_unfit_output_row(input_row: pd.Series, status: str) -> dict[str, Any]:
    out = copy.deepcopy(input_row.to_dict())
    out["synthetic_status"] = status
    for col in [
        "PL_diam", "PL_diam_err_ivw", "H_PL_alpha", "H_PL_alpha_err_ivw", "K_PL_alpha", "K_PL_alpha_err_ivw",
        "PL_V2", "chi2_PL", "PL_diam_input", "H_PL_alpha_input", "K_PL_alpha_input", "H_PL_alpha_Imu",
        "K_PL_alpha_Imu", "H_PL2_alpha_input", "K_PL2_alpha_input", "H_PL2_alpha_fit", "K_PL2_alpha_fit",
        "H_PL2_alpha_fit_err", "K_PL2_alpha_fit_err", "PL2_diam_fit", "PL2_V2_fit",
        "PL_diam_fit", "PL_diam_fit_err", "H_PL_alpha_fit", "H_PL_alpha_fit_err",
        "K_PL_alpha_fit", "K_PL_alpha_fit_err", "PL_V2_fit", "PL_V2_fit_err", "chi2_PL_fit",
        "n_v2_H", "n_v2_K", "H_input_wrms", "K_input_wrms", "H_alphaI_wrms", "K_alphaI_wrms",
        "H_alphaI_max_abs", "K_alphaI_max_abs", "H_alpha_scan_best", "K_alpha_scan_best",
        "H_alpha_scan_best_wrms", "K_alpha_scan_best_wrms",
    ]:
        out[col] = np.nan
    out["synthetic_png"] = ""
    out["synthetic_intensity_png"] = ""
    out["synthetic_show_png"] = ""
    out["synthetic_show_recovered_png"] = ""
    out["real_chara_png"] = ""
    out["alpha_scan_png"] = ""
    return out


def run_recovery(
    merged_csv: str,
    oifits_dir: str,
    output_csv: str,
    *,
    data_dir: str,
    grid: str,
    law: str,
    target: str | None,
    limit: int | None,
    min_rel_v2: float,
    maxfev: int,
    fig_dir: str | None,
) -> pd.DataFrame:
    merged = pd.read_csv(merged_csv)
    if target:
        key = target.strip().lower().replace(" ", "_")
        target_series = merged.get("target", merged.get("Target", pd.Series("", index=merged.index)))
        chara_series = merged.get("Target_chara", pd.Series("", index=merged.index))
        merged = merged[
            target_series.astype(str).str.lower().str.replace(" ", "_").eq(key)
            | chara_series.astype(str).str.lower().str.replace(" ", "_").eq(key)
        ]
    if limit is not None:
        merged = merged.head(int(limit))

    rows: list[dict[str, Any]] = []
    for idx, row in merged.iterrows():
        target_name = str(row.get("target") or row.get("Target_chara") or row.get("Target") or idx)
        mircx_path = resolve_oifits_path(row.get("oifits_h") or row.get("MIRCX_file_first"), oifits_dir, require_exists=True, mirc_aliases=True)
        mystic_path = resolve_oifits_path(row.get("oifits_k") or row.get("MYSTIC_file_first"), oifits_dir, require_exists=True, mirc_aliases=True)
        diam_input = _as_float(row.get("theta_PL") if "theta_PL" in row else row.get("PL_diam", row.get("theta_init")))
        teff = _as_float(row.get("teff", row.get("Teff")))
        logg = _as_float(row.get("logg"))
        feh = _as_float(row.get("feh", row.get("FeH")))
        mass = _as_float(row.get("mass", row.get("Mass")))
        missing = []
        if mircx_path is None:
            missing.append("MIRCX_file_first")
        if mystic_path is None:
            missing.append("MYSTIC_file_first")
        if not _is_finite(diam_input):
            missing.append("PL_diam")
        if not _is_finite(teff):
            missing.append("Teff")
        if not _is_finite(logg):
            missing.append("logg")
        if grid != "satlasross" and not _is_finite(feh):
            missing.append("FeH")
        if missing:
            rows.append(_make_unfit_output_row(row, "skipped_missing_" + ",".join(missing)))
            print(f"[SKIP] {target_name}: missing {', '.join(missing)}")
            continue
        try:
            fit = _fit_target(
                target_name,
                mircx_path,
                mystic_path,
                diam_input=diam_input,
                teff=teff,
                logg=logg,
                feh=feh,
                mass=mass,
                grid=grid,
                law=law,
                data_dir=data_dir,
                min_rel_v2=min_rel_v2,
                maxfev=maxfev,
                fig_dir=fig_dir,
            )
            rows.append(_make_output_row(row, fit, mircx_path=mircx_path, mystic_path=mystic_path))
            print(
                f"[FIT] {target_name} [{grid},{law}]: p1_H(I)={fit['H_PL_alpha_input']:.4f} p1_H(V)={fit['H_PL_alpha_fit']:.4f} "
                f"p1_K(I)={fit['K_PL_alpha_input']:.4f} p1_K(V)={fit['K_PL_alpha_fit']:.4f} "
                f"| diam_in/fit={fit['PL_diam_input']:.4f}/{fit['PL_diam_fit']:.4f} "
                f"| chi2={fit['chi2_PL_fit']:.4f} "
                f"| wrms input H/K={fit['H_input_wrms']:.3g}/{fit['K_input_wrms']:.3g} "
                f"| wrms alphaI H/K={fit['H_alphaI_wrms']:.3g}/{fit['K_alphaI_wrms']:.3g}"
            )
        except Exception as exc:
            rows.append(_make_unfit_output_row(row, f"failed_{type(exc).__name__}: {exc}"))
            print(f"[FAIL] {target_name}: {type(exc).__name__}: {exc}")

    legacy = pd.DataFrame(rows)
    public_rows: list[dict[str, Any]] = []
    for _, item in legacy.iterrows():
        public_row = base_metadata_from_row(item)
        grid_name = clean_grid(item.get("grid_model", grid))
        law_name = str(item.get("law_model", law))
        public_row.update({
            "grid": grid_name,
            "law": "power" if law_name == "power1" else law_name,
            "synthetic_status": item.get("synthetic_status", ""),
            "theta_input": item.get("PL_diam_input", np.nan),
            "theta_recovered": item.get("PL_diam_fit", np.nan),
            "theta_recovered_err": item.get("PL_diam_fit_err", np.nan),
            "v2_0_recovered": item.get("PL_V2_fit", np.nan),
            "chi2_red_recovered": item.get("chi2_PL_fit", np.nan),
            "n_v2_H": item.get("n_v2_H", np.nan),
            "n_v2_K": item.get("n_v2_K", np.nan),
        })
        if law_name == "power2":
            public_row.update({
                f"p1_input_H_{grid_name}": item.get("H_PL_alpha_input", np.nan),
                f"p1_recovered_H_{grid_name}": item.get("H_PL_alpha_fit", np.nan),
                f"p1_recovered_H_{grid_name}_err": item.get("H_PL_alpha_fit_err", np.nan),
                f"p1_input_K_{grid_name}": item.get("K_PL_alpha_input", np.nan),
                f"p1_recovered_K_{grid_name}": item.get("K_PL_alpha_fit", np.nan),
                f"p1_recovered_K_{grid_name}_err": item.get("K_PL_alpha_fit_err", np.nan),
                f"alpha1_input_H_{grid_name}": item.get("H_PL2_alpha_input", np.nan),
                f"alpha1_recovered_H_{grid_name}": item.get("H_PL2_alpha_fit", np.nan),
                f"alpha1_recovered_H_{grid_name}_err": item.get("H_PL2_alpha_fit_err", np.nan),
                f"alpha1_input_K_{grid_name}": item.get("K_PL2_alpha_input", np.nan),
                f"alpha1_recovered_K_{grid_name}": item.get("K_PL2_alpha_fit", np.nan),
                f"alpha1_recovered_K_{grid_name}_err": item.get("K_PL2_alpha_fit_err", np.nan),
            })
        else:
            public_row.update({
                f"alpha_input_H_{grid_name}": item.get("H_PL_alpha_input", np.nan),
                f"alpha_recovered_H_{grid_name}": item.get("H_PL_alpha_fit", np.nan),
                f"alpha_recovered_H_{grid_name}_err": item.get("H_PL_alpha_fit_err", np.nan),
                f"alpha_input_K_{grid_name}": item.get("K_PL_alpha_input", np.nan),
                f"alpha_recovered_K_{grid_name}": item.get("K_PL_alpha_fit", np.nan),
                f"alpha_recovered_K_{grid_name}_err": item.get("K_PL_alpha_fit_err", np.nan),
            })
        public_rows.append(public_row)
    result = pd.DataFrame(public_rows)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False)
    print(f"Wrote {len(result)} rows to {output_csv}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-csv", default=DEFAULT_MERGED_CSV)
    parser.add_argument("--oifits-dir", default=DEFAULT_OIFITS_DIR)
    parser.add_argument("--output-csv", default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--grid", choices=GRID_CHOICES, default="stagger")
    parser.add_argument("--law", choices=LAW_CHOICES, default="power1")
    parser.add_argument("--fig-dir", default=DEFAULT_FIG_DIR)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--target", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-rel-v2", type=float, default=0.0)
    parser.add_argument("--maxfev", type=int, default=10000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    grids = ["stagger", "kurucz", "mps1", "mps2", "satlasross"] if args.grid == "all" else [args.grid]
    for grid in grids:
        output_csv = args.output_csv
        fig_dir = None if args.no_png else args.fig_dir
        if args.grid == "all" or args.output_csv == DEFAULT_OUTPUT_CSV:
            output_csv = CSV_DIR / f"svam_recovery_{grid}_{args.law}_realuv.csv"
        if fig_dir is not None and (args.grid == "all" or args.fig_dir == DEFAULT_FIG_DIR):
            fig_dir = f"figs_svam_recovery_{grid}_{args.law}"
        print(f"[GRID] {grid}, law={args.law}: output_csv={output_csv}")
        run_recovery(
            args.merged_csv,
            args.oifits_dir,
            output_csv,
            data_dir=args.data_dir,
            grid=grid,
            law=args.law,
            target=args.target,
            limit=args.limit,
            min_rel_v2=args.min_rel_v2,
            maxfev=args.maxfev,
            fig_dir=fig_dir,
        )


if __name__ == "__main__":
    main()
