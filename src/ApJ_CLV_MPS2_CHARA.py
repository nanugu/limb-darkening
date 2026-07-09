#!/usr/bin/env python3
"""Compare CHARA and ExoTiC-LD limb darkening in the intensity domain."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, Sequence, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import product
from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import linear_ld_law, power1_ld_law, power2_ld_law, quadratic_ld_law
from compute_exotic_coefficients import _compute_satlas_cliv, _satlas_mu_to_rosseland_mu

ROOT = Path(__file__).resolve().parents[1]
CSV = str(ROOT / 'csv' / 'merged_four_branches_wide.csv')
MODEL_COEFF_CSV = str(ROOT / 'csv' / 'merged_four_branches_wide.csv')

CHARA_COLS = {
    "H": {
        "linear": "a_chara_H",
        "linear_err": "a_chara_H_err",
        "power1": "alpha_chara_H",
        "power1_err": "alpha_chara_H_err",
        "power2_p1": "p1_chara_H",
        "power2_p1_err": "p1_chara_H_err",
        "power2_alpha": "alpha1_chara_H",
        "power2_alpha_err": "alpha1_chara_H_err",
        "quad_u1": "u1_chara_H",
        "quad_u1_err": "u1_chara_H_err",
        "quad_u2": "u2_chara_H",
        "quad_u2_err": "u2_chara_H_err",
    },
    "K": {
        "linear": "a_chara_K",
        "linear_err": "a_chara_K_err",
        "power1": "alpha_chara_K",
        "power1_err": "alpha_chara_K_err",
        "power2_p1": "p1_chara_K",
        "power2_p1_err": "p1_chara_K_err",
        "power2_alpha": "alpha1_chara_K",
        "power2_alpha_err": "alpha1_chara_K_err",
        "quad_u1": "u1_chara_K",
        "quad_u1_err": "u1_chara_K_err",
        "quad_u2": "u2_chara_K",
        "quad_u2_err": "u2_chara_K_err",
    },
}


def grid_columns(model: str) -> Dict[str, Dict[str, str]]:
    public_grid = {
        "kurucz": "Kurucz",
        "stagger": "Stagger",
        "mps1": "MPS1",
        "mps2": "MPS2",
        "satlasross": "SATLASRoss",
    }[model.lower()]
    return {
        "H": {
            "linear": f"a_I_H_{public_grid}",
            "power1": f"alpha_I_H_{public_grid}",
            "power2_p1": f"p1_I_H_{public_grid}",
            "power2_alpha": f"alpha1_I_H_{public_grid}",
            "quad_u1": f"u1_I_H_{public_grid}",
            "quad_u2": f"u2_I_H_{public_grid}",
        },
        "K": {
            "linear": f"a_I_K_{public_grid}",
            "power1": f"alpha_I_K_{public_grid}",
            "power2_p1": f"p1_I_K_{public_grid}",
            "power2_alpha": f"alpha1_I_K_{public_grid}",
            "quad_u1": f"u1_I_K_{public_grid}",
            "quad_u2": f"u2_I_K_{public_grid}",
        },
    }


WAVELENGTHS = {
    "H": (np.linspace(15000.0, 17200.0, 120), np.ones(120, dtype=float)),
    "K": (np.linspace(20000.0, 23700.0, 120), np.ones(120, dtype=float)),
}


def compute_intensity_envelope(
    mu: Sequence[float],
    func: Callable[..., np.ndarray],
    params: Sequence[float],
    errs: Sequence[float],
) -> Tuple[np.ndarray, np.ndarray] | None:
    """Compute min/max envelopes by perturbing params by ±1σ (no covariance)."""

    if len(params) == 0 or len(params) != len(errs):
        return None
    if any(not (np.isfinite(p) and np.isfinite(e) and e >= 0.0) for p, e in zip(params, errs)):
        return None
    if not any(e > 0.0 for e in errs):
        return None

    mu_arr = np.asarray(mu, dtype=float)
    if mu_arr.size == 0:
        return None

    curves = []
    for signs in product((-1.0, 1.0), repeat=len(params)):
        perturbed = [p + s * e for p, e, s in zip(params, errs, signs)]
        try:
            curve = np.asarray(func(mu_arr, *perturbed), dtype=float)
        except Exception:
            return None
        if curve.shape != mu_arr.shape:
            return None
        curves.append(curve)

    if not curves:
        return None

    curve_stack = np.vstack(curves)
    lower = np.nanmin(curve_stack, axis=0)
    upper = np.nanmax(curve_stack, axis=0)
    return lower, upper


def safe_float(value) -> float:
    try:
        val = float(value)
    except Exception:
        return np.nan
    return val if np.isfinite(val) else np.nan


def get_target_name(row: pd.Series) -> str:
    for key in ("target", "target_norm", "Target", "Target_catalog", "Name"):
        if key in row and pd.notna(row[key]):
            return str(row[key]).strip().replace(" ", "_")
    if "hd" in row and pd.notna(row["hd"]):
        try:
            return f"HD_{int(float(row['hd']))}"
        except Exception:
            pass
    if "HD" in row and pd.notna(row["HD"]):
        try:
            return f"HD_{int(float(row['HD']))}"
        except Exception:
            pass
    return "Unknown"


def format_target_title(name: str) -> str:
    text = str(name).strip().replace("_", " ")
    greek = {
        "alf": r"$\alpha$",
        "bet": r"$\beta$",
        "gam": r"$\gamma$",
        "del": r"$\delta$",
        "eps": r"$\epsilon$",
        "zet": r"$\zeta$",
        "eta": r"$\eta$",
        "iot": r"$\iota$",
        "ksi": r"$\xi$",
        "lam": r"$\lambda$",
        "mu.": r"$\mu$",
        "mu": r"$\mu$",
        "phi02": r"$\phi^2$",
        "rho": r"$\rho$",
        "sig": r"$\sigma$",
        "ups": r"$\upsilon$",
    }
    parts = text.split()
    if not parts:
        return text
    key = parts[0].lower()
    if key in greek:
        parts[0] = greek[key]
    return " ".join(parts)


def compute_cliv(sld: StellarLimbDarkening, band: str) -> Tuple[np.ndarray, np.ndarray]:
    wavelengths, throughput = WAVELENGTHS[band]
    mu_min = float(np.asarray(getattr(sld, "mus", []), dtype=float).min()) if np.asarray(getattr(sld, "mus", []), dtype=float).size else None
    kwargs = dict(
        mode="custom",
        wavelength_range=[wavelengths.min(), wavelengths.max()],
        custom_wavelengths=wavelengths,
        custom_throughput=throughput,
        return_sigmas=False,
    )
    if mu_min is not None and np.isfinite(mu_min):
        kwargs["mu_min"] = mu_min
    try:
        sld.compute_linear_ld_coeffs(**kwargs)
        sld.compute_quadratic_ld_coeffs(**kwargs)
        sld.compute_power1_ld_coeffs(**kwargs)
        sld.compute_power2_ld_coeffs(**kwargs)
    except Exception:
        return np.array([]), np.array([])

    mu = np.asarray(getattr(sld, "mus", []), dtype=float).reshape(-1)
    intens = np.asarray(getattr(sld, "I_mu", []), dtype=float).reshape(-1)
    n = min(mu.size, intens.size)
    if n == 0:
        return np.array([]), np.array([])
    if mu.size != intens.size:
        print(f"      CLIV size mismatch (mu={mu.size}, I={intens.size}); truncating to {n}")
    return mu[:n], intens[:n]


def intensity_power2(mu: np.ndarray, p1: float, alpha: float) -> np.ndarray:
    return power2_ld_law(mu, p1, alpha)


def model_display_name(model: str) -> str:
    aliases = {
        "mps1": "MPS1",
        "mps2": "MPS2",
        "stagger": "Stagger",
        "kurucz": "Kurucz",
        "satlasross": r"SATLAS ($\mu_{\rm Ross}$)",
    }
    return aliases.get(str(model).lower(), str(model))


def compute_reference_cliv(
    *,
    model: str,
    teff: float,
    logg: float,
    feh: float,
    band: str,
    ld_data_dir: str,
    mass: float | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if str(model).lower() == "satlasross":
        mu, intens = _compute_satlas_cliv(teff, logg, band=band, mass=mass)
        mu = _satlas_mu_to_rosseland_mu(teff, logg, mass, mu)
        mask = np.isfinite(mu) & np.isfinite(intens)
        mu = mu[mask]
        intens = intens[mask]
        if mu.size > 1:
            order = np.argsort(mu)
            mu, intens = mu[order], intens[order]
        return mu, intens

    sld = StellarLimbDarkening(Teff=teff, logg=logg, M_H=feh, ld_model=model, ld_data_path=ld_data_dir)
    return compute_cliv(sld, band)


def _normalize_target_key(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip().replace(" ", "_").lower()


def lookup_model_coeff_row(row: pd.Series, model_coeff_df: pd.DataFrame | None) -> pd.Series | None:
    if model_coeff_df is None or model_coeff_df.empty:
        return None

    target_keys = [
        _normalize_target_key(row.get("target_norm")),
        _normalize_target_key(row.get("target")),
        _normalize_target_key(get_target_name(row)),
    ]
    target_keys = [key for key in target_keys if key]

    for col in ("target_norm", "target", "Target"):
        if col not in model_coeff_df.columns or not target_keys:
            continue
        matches = model_coeff_df[model_coeff_df[col].map(_normalize_target_key).isin(target_keys)]
        if not matches.empty:
            return matches.iloc[0]

    hd_value = row.get("hd", row.get("HD"))
    hd_float = safe_float(hd_value)
    if np.isfinite(hd_float):
        for col in ("hd", "HD"):
            if col not in model_coeff_df.columns:
                continue
            hd_col = pd.to_numeric(model_coeff_df[col], errors="coerce")
            matches = model_coeff_df[np.isfinite(hd_col) & (np.abs(hd_col - hd_float) < 0.5)]
            if not matches.empty:
                return matches.iloc[0]
    return None


def clv_distance(
    mu: Sequence[float],
    I1: Sequence[float],
    I2: Sequence[float],
    *,
    weight: str = "uniform",
    mu_max: float | None = None,
) -> float:
    """Compute CLV distance metric D(I1, I2) as defined in the manuscript.

    D(I1, I2) = [∫ w(μ) (I1(μ) - I2(μ))^2 dμ]^{1/2}, optionally restricted to μ ≤ mu_max.
    """

    mu_arr = np.asarray(mu, dtype=float).reshape(-1)
    I1_arr = np.asarray(I1, dtype=float).reshape(-1)
    I2_arr = np.asarray(I2, dtype=float).reshape(-1)
    n = min(mu_arr.size, I1_arr.size, I2_arr.size)
    if n < 2:
        return np.nan
    mu_arr = mu_arr[:n]
    I1_arr = I1_arr[:n]
    I2_arr = I2_arr[:n]

    mask = np.isfinite(mu_arr) & np.isfinite(I1_arr) & np.isfinite(I2_arr)
    if mu_max is not None:
        mask &= mu_arr <= float(mu_max)
    mu_arr = mu_arr[mask]
    I1_arr = I1_arr[mask]
    I2_arr = I2_arr[mask]
    if mu_arr.size < 2:
        return np.nan

    order = np.argsort(mu_arr)
    mu_arr = mu_arr[order]
    I1_arr = I1_arr[order]
    I2_arr = I2_arr[order]

    if weight == "uniform":
        w = np.ones_like(mu_arr)
    elif weight == "mu":
        w = mu_arr
    else:
        raise ValueError(f"Unknown weight='{weight}' (expected 'uniform' or 'mu').")

    diff2 = (I1_arr - I2_arr) ** 2
    integral = np.trapz(w * diff2, mu_arr)
    if not np.isfinite(integral) or integral < 0:
        return np.nan
    return float(np.sqrt(integral))


def plot_powerlaw_reference_summary(metrics_df: pd.DataFrame, outdir: Path) -> None:
    df = metrics_df.copy()
    if df.empty:
        return
    df = df[df["metric_kind"] == "to_chara_power1"].copy()
    if df.empty:
        return

    law_order = ["linear", "power2", "quadratic", "model_clv"]
    law_labels = {
        "linear": r"$a$",
        "power2": r"$(p_1,\alpha_1)$",
        "quadratic": r"$(u_1,u_2)$",
        "model_clv": r"model $I(\mu)$",
    }
    source_offsets = {"model": -0.14, "chara": 0.14}
    source_colors = {"model": cm.tab10(0), "chara": cm.tab10(1)}
    source_labels = {"model": "model", "chara": "CHARA"}

    grouped = (
        df.groupby(["target", "band", "law", "source"], as_index=False)["D"]
        .median()
    )

    for band in ("H", "K"):
        band_df = grouped[grouped["band"] == band].copy()
        if band_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(6.1, 4.6))
        for i, law in enumerate(law_order):
            law_df = band_df[band_df["law"] == law]
            if law_df.empty:
                continue
            for source in ("model", "chara"):
                sub = law_df[law_df["source"] == source]
                if sub.empty:
                    continue
                x = np.full(len(sub), i + source_offsets[source], dtype=float)
                ax.scatter(
                    x,
                    sub["D"].to_numpy(float),
                    s=32,
                    color=source_colors[source],
                    alpha=0.65,
                    edgecolors="none",
                    label=source_labels[source] if i == 0 else None,
                )
                med = np.nanmedian(sub["D"].to_numpy(float))
                ax.plot(
                    [i + source_offsets[source] - 0.08, i + source_offsets[source] + 0.08],
                    [med, med],
                    color=source_colors[source],
                    lw=2.6,
                )

        ax.set_xticks(range(len(law_order)))
        ax.set_xticklabels([law_labels[law] for law in law_order], fontsize=12)
        ax.set_ylabel(r"$D\!\left[I(\mu), I_{\rm pow}^{\rm CHARA}(\mu)\right]$", fontsize=13)
        ax.set_title(f"{band}-band", fontsize=14)
        ax.grid(True, alpha=0.25, axis="y")
        ax.tick_params(labelsize=11)
        ax.set_ylim(0.0, 0.12)
        ax.legend(frameon=False, fontsize=11, loc="best")
        fig.tight_layout()
        outfile = outdir / f"clv_distance_to_chara_powerlaw_{band}.png"
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)


def plot_target(
    row: pd.Series,
    *,
    grid_cols: Dict[str, Dict[str, str]],
    model: str,
    ld_data_dir: str,
    outdir: Path,
    mu_grid: np.ndarray,
    metrics_rows: list[dict] | None = None,
    df_epochs: pd.DataFrame | None = None,
    label_override: str | None = None,
    base_model: str | None = None,
    model_coeff_df: pd.DataFrame | None = None,
) -> None:

    name = label_override or get_target_name(row)
    name = name.replace("_", " ")
    title_name = format_target_title(get_target_name(row))

    Teff = safe_float(row.get("teff", row.get("Teff")))
    logg = safe_float(row.get("logg"))
    feh = safe_float(row.get("feh", row.get("M_H", row.get("FeH", np.nan))))

    if not np.all(np.isfinite([Teff, logg, feh])):
        return

    mass = safe_float(row.get("mass", row.get("Mass", np.nan)))
    reference_model = (base_model or model).lower()

    sld = None
    if reference_model != "satlasross":
        try:
            sld = StellarLimbDarkening(Teff=Teff, logg=logg, M_H=feh, ld_model=reference_model, ld_data_path=ld_data_dir)
        except Exception as exc:
            print(f"    Could not load CLIV for {name}: {exc}")
            return

    inst = str(row.get("instrument", row.get("Instrument", ""))).upper()
    # enforce instrument-band pairing: MIRCX->H, MYSTIC->K
    bands = []
    if inst == "MIRCX":
        bands = ["H"]
    elif inst == "MYSTIC":
        bands = ["K"]
    else:
        bands = ["H", "K"]

    for band in bands:
        chara_cols = CHARA_COLS[band]
        seed_cols = grid_cols[band]
        seed_row = lookup_model_coeff_row(row, model_coeff_df)
        if seed_row is None:
            seed_row = row
        coeffs = {
            "linear": (safe_float(row.get(chara_cols["linear"])),),
            "power1": (safe_float(row.get(chara_cols["power1"])),),
            "power2": (
                safe_float(row.get(chara_cols["power2_p1"])),
                safe_float(row.get(chara_cols["power2_alpha"])),
            ),
            "quadratic": (
                safe_float(row.get(chara_cols["quad_u1"])),
                safe_float(row.get(chara_cols["quad_u2"])),
            ),
        }
        seed_coeffs = {
            "linear": (safe_float(seed_row.get(seed_cols["linear"])),),
            "power1": (safe_float(seed_row.get(seed_cols["power1"])),),
            "power2": (
                safe_float(seed_row.get(seed_cols["power2_p1"])),
                safe_float(seed_row.get(seed_cols["power2_alpha"])),
            ),
            "quadratic": (
                safe_float(seed_row.get(seed_cols["quad_u1"])),
                safe_float(seed_row.get(seed_cols["quad_u2"])),
            ),
        }

        if not any(np.isfinite(val) for group in coeffs.values() for val in group):
            continue

        cliv_mu, cliv_I = compute_reference_cliv(
            model=reference_model,
            teff=Teff,
            logg=logg,
            feh=feh,
            band=band,
            ld_data_dir=ld_data_dir,
            mass=mass,
        )

        coeff_errs = {
            "linear": safe_float(row.get(chara_cols["linear_err"])),
            "power1": safe_float(row.get(chara_cols["power1_err"])),
            "power2": (
                safe_float(row.get(chara_cols["power2_p1_err"])),
                safe_float(row.get(chara_cols["power2_alpha_err"])),
            ),
            "quadratic": (
                safe_float(row.get(chara_cols["quad_u1_err"])),
                safe_float(row.get(chara_cols["quad_u2_err"])),
            ),
        }

        fig = plt.figure(figsize=(8.6, 6.8))
        gs = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.2], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_resid = fig.add_subplot(gs[1], sharex=ax_main)
        ax_main.set_title(f"{title_name} ({band}-band)", fontsize=17)
        ax_main.grid(True, alpha=0.3)
        ax_main.tick_params(labelbottom=False)
        ax_resid.axhline(0.0, color="0.2", lw=1.0, alpha=0.4)
        ax_resid.grid(True, alpha=0.3)
        ax_main.tick_params(labelsize=16)
        ax_resid.tick_params(labelsize=15)

        has_cliv = bool(cliv_mu.size and cliv_I.size)

        # --- CLV distance metrics (CHARA-law vs model CLV) ---
        if metrics_rows is not None:
            mu_limb = 0.3
            record_base = {
                "target": get_target_name(row),
                "label": (label_override or get_target_name(row)),
                "date": str(row.get("Date", "")) if "Date" in row else "",
                "instrument": str(row.get("Instrument", "")) if "Instrument" in row else "",
                "band": band,
                "grid": reference_model,
                "Teff": Teff,
                "logg": logg,
                "M_H": feh,
                "mu_limb": mu_limb,
                "mu_grid_n": int(np.asarray(mu_grid).size),
            }

            def add_metric_row(law: str, I_curve: np.ndarray | None) -> None:
                if I_curve is None or not has_cliv:
                    return
                metrics_rows.append(
                    {
                        **record_base,
                        "law": law,
                        "source": "chara",
                        "metric_kind": "to_model_clv",
                        "D": clv_distance(cliv_mu, I_curve, cliv_I, weight="uniform"),
                        "D_mu": clv_distance(cliv_mu, I_curve, cliv_I, weight="mu"),
                        "D_limb": clv_distance(cliv_mu, I_curve, cliv_I, weight="uniform", mu_max=mu_limb),
                        "D_limb_mu": clv_distance(cliv_mu, I_curve, cliv_I, weight="mu", mu_max=mu_limb),
                    }
                )

            def add_reference_metric_row(law: str, source: str, I_curve: np.ndarray | None, I_ref: np.ndarray | None) -> None:
                if I_curve is None or I_ref is None or not has_cliv:
                    return
                metrics_rows.append(
                    {
                        **record_base,
                        "law": law,
                        "source": source,
                        "metric_kind": "to_chara_power1",
                        "D": clv_distance(cliv_mu, I_curve, I_ref, weight="uniform"),
                        "D_mu": clv_distance(cliv_mu, I_curve, I_ref, weight="mu"),
                        "D_limb": clv_distance(cliv_mu, I_curve, I_ref, weight="uniform", mu_max=mu_limb),
                        "D_limb_mu": clv_distance(cliv_mu, I_curve, I_ref, weight="mu", mu_max=mu_limb),
                    }
                )

            (c_a,) = coeffs["linear"]
            add_metric_row(
                "linear",
                linear_ld_law(cliv_mu, c_a) if has_cliv and np.isfinite(c_a) else None,
            )

            (c_alpha,) = coeffs["power1"]
            add_metric_row(
                "power1",
                power1_ld_law(cliv_mu, c_alpha) if has_cliv and np.isfinite(c_alpha) else None,
            )

            c_p1, c_p2 = coeffs["power2"]
            add_metric_row(
                "power2",
                power2_ld_law(cliv_mu, c_p1, c_p2) if has_cliv and np.isfinite(c_p1) and np.isfinite(c_p2) else None,
            )

            c_u1, c_u2 = coeffs["quadratic"]
            add_metric_row(
                "quadratic",
                quadratic_ld_law(cliv_mu, c_u1, c_u2) if has_cliv and np.isfinite(c_u1) and np.isfinite(c_u2) else None,
            )

            (c_alpha_ref,) = coeffs["power1"]
            I_ref = power1_ld_law(cliv_mu, c_alpha_ref) if has_cliv and np.isfinite(c_alpha_ref) else None
            add_reference_metric_row("model_clv", "model", cliv_I if has_cliv else None, I_ref)
            add_reference_metric_row(
                "linear",
                "model",
                linear_ld_law(cliv_mu, *seed_coeffs["linear"]) if has_cliv and all(np.isfinite(seed_coeffs["linear"])) else None,
                I_ref,
            )
            add_reference_metric_row(
                "linear",
                "chara",
                linear_ld_law(cliv_mu, *coeffs["linear"]) if has_cliv and all(np.isfinite(coeffs["linear"])) else None,
                I_ref,
            )
            add_reference_metric_row(
                "power2",
                "model",
                intensity_power2(cliv_mu, *seed_coeffs["power2"]) if has_cliv and all(np.isfinite(seed_coeffs["power2"])) else None,
                I_ref,
            )
            add_reference_metric_row(
                "power2",
                "chara",
                intensity_power2(cliv_mu, *coeffs["power2"]) if has_cliv and all(np.isfinite(coeffs["power2"])) else None,
                I_ref,
            )
            add_reference_metric_row(
                "quadratic",
                "model",
                quadratic_ld_law(cliv_mu, *seed_coeffs["quadratic"]) if has_cliv and all(np.isfinite(seed_coeffs["quadratic"])) else None,
                I_ref,
            )
            add_reference_metric_row(
                "quadratic",
                "chara",
                quadratic_ld_law(cliv_mu, *coeffs["quadratic"]) if has_cliv and all(np.isfinite(coeffs["quadratic"])) else None,
                I_ref,
            )

        def add_cliv(ax):
            ax.plot(
                cliv_mu,
                cliv_I,
                "-",
                color=cm.inferno(0.15),
                lw=2.2,
                label=r"model $I(\mu)$",
            )


        def plot_residual(ax_resid, model_intensity, linestyle, color, accum):
            if has_cliv and model_intensity.size:
                resid = cliv_I - model_intensity
                ax_resid.plot(cliv_mu, resid, linestyle, lw=1.6, color=color)
                accum.append(resid)
                return resid
            return None

        def finalize_residual_axis(ax_resid, residual_sets):
            ax_resid.set_xlim(0, 1)
            if not residual_sets:
                ax_resid.set_ylim(-0.05, 0.05)
                return
            residual_concat = np.hstack([res[np.isfinite(res)] for res in residual_sets if res.size])
            if residual_concat.size == 0:
                ax_resid.set_ylim(-0.05, 0.05)
                return
            max_abs = np.abs(residual_concat).max()
            if max_abs == 0:
                max_abs = 0.05
            ax_resid.set_ylim(-1.1 * max_abs, 1.1 * max_abs)

        ax_resid.set_ylabel(r"$\Delta I$", fontsize=17)
        ax_resid.set_xlabel(r"$\mu$", fontsize=17)

        def apply_chara_uncertainty(ax_main, ax_resid, law_func, params, errs, color, residual_sets):
            envelope = compute_intensity_envelope(mu_grid, law_func, params, errs)
            if envelope is None:
                return
            lower, upper = envelope
            ax_main.fill_between(mu_grid, lower, upper, color=color, alpha=0.2, linewidth=0)
            if has_cliv:
                cliv_env = compute_intensity_envelope(cliv_mu, law_func, params, errs)
                if cliv_env is None:
                    return
                lower_c, upper_c = cliv_env
                resid_lower = cliv_I - upper_c
                resid_upper = cliv_I - lower_c
                ax_resid.fill_between(cliv_mu, resid_lower, resid_upper, color=color, alpha=0.15, linewidth=0)
                residual_sets.extend([resid_lower, resid_upper])

        residual_sets = []
        law_specs = [
            (r"$a$", "linear", linear_ld_law, coeffs["linear"], coeff_errs["linear"], "-"),
            (r"$\alpha$", "power1", power1_ld_law, coeffs["power1"], coeff_errs["power1"], "-"),
            (r"$(p_1,\alpha_1)$", "power2", intensity_power2, coeffs["power2"], coeff_errs["power2"], "-"),
            (r"$(u_1,u_2)$", "quadratic", quadratic_ld_law, coeffs["quadratic"], coeff_errs["quadratic"], "-"),
        ]
        law_colors = {
            r"$a$": cm.tab10(0),
            r"$\alpha$": cm.tab10(1),
            r"$(p_1,\alpha_1)$": cm.tab10(3),
            r"$(u_1,u_2)$": cm.tab10(2),
        }

        if has_cliv:
            add_cliv(ax_main)

        for law_label, law_key, law_func, law_coeffs, law_errs, ls_chara in law_specs:
            color = law_colors[law_label]

            chara_params = law_coeffs
            seed_params = seed_coeffs[law_key]

            if all(np.isfinite(seed_params)):
                line = ax_main.plot(
                    mu_grid,
                    law_func(mu_grid, *seed_params),
                    "--",
                    lw=1.8,
                    color=color,
                    alpha=0.85,
                    label=rf"{law_label}: model",
                )
                if has_cliv:
                    model_vals = law_func(cliv_mu, *seed_params)
                    plot_residual(ax_resid, model_vals, "--", line[0].get_color(), residual_sets)

            if all(np.isfinite(chara_params)):
                line = ax_main.plot(
                    mu_grid,
                    law_func(mu_grid, *chara_params),
                    ls_chara,
                    lw=2.3,
                    color=color,
                    label=rf"{law_label}: CHARA",
                )
                if has_cliv:
                    model_vals = law_func(cliv_mu, *chara_params)
                    plot_residual(ax_resid, model_vals, ls_chara, line[0].get_color(), residual_sets)

                errs_tuple = law_errs if isinstance(law_errs, tuple) else (law_errs,)
                if any(np.isfinite(e) and e > 0 for e in errs_tuple):
                    apply_chara_uncertainty(
                        ax_main,
                        ax_resid,
                        law_func,
                        chara_params,
                        errs_tuple,
                        line[0].get_color(),
                        residual_sets,
                    )

        ax_main.set_ylabel(r"$I(\mu)/I(1)$", fontsize=18)
        ax_main.set_xlim(0, 1)
        ax_main.set_ylim(-0.05, 1.05)
        ax_main.legend(fontsize=12, ncol=2, loc="lower right")
        finalize_residual_axis(ax_resid, residual_sets)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        name = label_override or get_target_name(row)
        outfile = outdir / f"{name}_{band}_{reference_model}_intensity.png"
        outdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close(fig)


def iter_rows(df: pd.DataFrame, limit: int | None) -> Iterable[pd.Series]:
    for idx, row in df.iterrows():
        if limit is not None and idx >= limit:
            break
        yield row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot CHARA vs ExoTiC-LD intensity-domain comparisons.")
    parser.add_argument("--csv", default=CSV, help="Merged CSV containing CHARA and ExoTiC-LD coefficients.")
    parser.add_argument("--out", default="intensity_chara_exotic_plots_all_epochs", help="Directory for output PNG files.")
    parser.add_argument("--grid", default="mps2", choices=["stagger", "kurucz", "mps1", "mps2"], help="Which ExoTiC-LD grid to use.")
    parser.add_argument("--satlas-mu-ross", action="store_true", help="Use SATLAS mapped onto the Rosseland-mu convention as the base atmospheric CLV and model coefficient set.")
    parser.add_argument("--ld-data", default=os.environ.get("EXOTIC_LD_DATA_DIR", ""), help="Path to the local ExoTiC-LD data cache. If omitted, set EXOTIC_LD_DATA_DIR.")
    parser.add_argument("--metrics-csv", default=None, help="Optional path to write CLV distance metrics as CSV.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of targets to plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(args.csv)
    if not Path(args.csv).expanduser().exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    df_all = pd.read_csv(args.csv)
    model_coeff_df = None
    model_coeff_path = Path(MODEL_COEFF_CSV).expanduser()
    if model_coeff_path.exists():
        model_coeff_df = pd.read_csv(model_coeff_path)
    df = df_all.copy()
    selected_model = "satlasross" if args.satlas_mu_ross else args.grid
    grid_cols = grid_columns(selected_model)
    mu_grid = np.linspace(0, 1, 201)
    outdir = Path(args.out)
    metrics_rows: list[dict] = []

    for row in iter_rows(df, args.limit):
        name = get_target_name(row)
        date = str(row.get("date", row.get("Date", "")))
        inst = str(row.get("instrument", row.get("Instrument", "")))
        label = name
        if date:
            label = f"{name}_{inst}_{date}"
        print(f"=== Plotting {label} | model={selected_model}")
        plot_target(
            row,
            grid_cols=grid_cols,
            model=args.grid,
            ld_data_dir=args.ld_data,
            outdir=outdir,
            mu_grid=mu_grid,
            metrics_rows=metrics_rows,
            label_override=label,
            base_model=selected_model,
            model_coeff_df=model_coeff_df,
        )

    if args.metrics_csv or metrics_rows:
        metrics_path = Path(args.metrics_csv) if args.metrics_csv else (outdir / f"clv_distance_metrics_{selected_model}.csv")
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df.to_csv(metrics_path, index=False)
        print(f"Wrote CLV distance metrics: {metrics_path} ({len(metrics_rows)} rows)")
        plot_powerlaw_reference_summary(metrics_df, outdir)


if __name__ == "__main__":
    main()
