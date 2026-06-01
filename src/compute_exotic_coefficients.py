#!/usr/bin/env python3
"""Build CHARA limb-darkening coefficient catalogs using ExoTiC-LD.

This version relies on ExoTiC-LD's ``return_sigmas=True`` support so the
reported uncertainties come directly from the library rather than an ad-hoc
Monte-Carlo wrapper. Optional diagnostic plots can be generated to visualise
the coefficient-driven intensity profiles for a handful of targets.
"""

from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from exotic_ld import StellarLimbDarkening
from exotic_ld.ld_laws import linear_ld_law, power1_ld_law, power2_ld_law, quadratic_ld_law
from numpy.typing import ArrayLike
from chara_fit_common import boxcar_throughput

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"

DEFAULT_TARGETS_CSV = CSV_DIR / "merged_four_branches_wide.csv"
DEFAULT_DATA_DIR = os.environ.get("EXOTIC_LD_DATA_DIR", str(ROOT / "exotic_ld_data"))
DEFAULT_OUTPUT_DIR = CSV_DIR
DEFAULT_PLOT_DIR = "ldc_check_plots"
OUTPUT_FILE = "compute_exotic_coefficients.csv"
GRID_PUBLIC = {"kurucz": "Kurucz", "stagger": "Stagger", "mps1": "MPS1", "mps2": "MPS2"}
INST_BAND = {"MIRCX": "H", "MYSTIC": "K"}

MODELS = [
    ("kurucz", "Kurucz/1D"),
    ("stagger", "STAGGER/3D"),
    ("mps1", "MPS-ATLAS-1"),
    ("mps2", "MPS-ATLAS-2"),
]

# CHARA bandpasses as boxcar ranges (Å). Replace with measured throughputs if available.
H_RANGE = [15000.0, 17200.0]  # MIRC-X (H)
K_RANGE = [20000.0, 23700.0]  # MYSTIC (K)

DEFAULT_SIG_T = 100.0  # K
DEFAULT_SIG_G = 0.20   # dex
DEFAULT_SIG_Z = 0.20   # dex
DEFAULT_PROP_DRAWS = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H")


def _col(df: pd.DataFrame, name: str) -> str | None:
    canon = {c.strip().lower(): c for c in df.columns}
    return canon.get(name.strip().lower())


W_H, T_H = boxcar_throughput(*H_RANGE)
W_K, T_K = boxcar_throughput(*K_RANGE)


def _to_float_array(values: ArrayLike, length: int | None = None) -> np.ndarray:
    arr = np.atleast_1d(np.array(values, dtype=float))
    if length is not None and arr.size != length:
        arr = np.full(length, np.nan)
    return arr


def _safe_compute(method: Callable, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray, str]:
    """Call an ExoTiC-LD ``compute_*`` method and coerce return + sigma arrays."""

    # Many ExoTiC-LD compute_* helpers learned ``return_sigmas`` recently; this wrapper
    # keeps backward compatibility and normalises the output into (values, sigmas, note).

    note = ""
    try:
        values, sigmas = method(*args, return_sigmas=True, **kwargs)
    except TypeError:
        values = method(*args, **kwargs)
        sigmas = np.full_like(np.atleast_1d(values), np.nan, dtype=float)
        note = "sigmas_unavailable"
    except Exception as exc:  # pragma: no cover - rely on NaNs downstream
        return np.full(1, np.nan), np.full(1, np.nan), f"{method.__name__}_fail:{exc}"

    return _to_float_array(values), _to_float_array(sigmas, length=len(np.atleast_1d(values))), note


def _summary_stats(samples: ArrayLike) -> Dict[str, float]:
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"std": np.nan, "p16": np.nan, "p50": np.nan, "p84": np.nan, "n": 0}
    return {
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "p16": float(np.percentile(arr, 16.0)),
        "p50": float(np.percentile(arr, 50.0)),
        "p84": float(np.percentile(arr, 84.0)),
        "n": int(arr.size),
    }


# ---------------------------------------------------------------------------
# Intensity sampling for diagnostics
# ---------------------------------------------------------------------------

def _law_intensity_single(law: str, mus: np.ndarray, params: np.ndarray) -> np.ndarray:
    if law == "linear":
        return linear_ld_law(mus, params[0])
    if law == "power1":
        return power1_ld_law(mus, params[0])
    if law == "power2":
        return power2_ld_law(mus, params[0], params[1])
    if law == "quadratic":
        return quadratic_ld_law(mus, params[0], params[1])
    raise ValueError(f"Unsupported law: {law}")


def sample_intensity_profiles(law: str, coeffs: np.ndarray, sigmas: np.ndarray, *, mus: np.ndarray, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.all(np.isfinite(sigmas)):
        raise ValueError("Non-finite coefficient uncertainties; cannot sample profile")

    draws = np.random.normal(loc=coeffs, scale=sigmas, size=(n_samples, coeffs.size))
    profiles = np.vstack([_law_intensity_single(law, mus, draw) for draw in draws])
    return np.percentile(profiles, [16.0, 50.0, 84.0], axis=0)


# @dataclass auto-generates __init__/__repr__ so PlotManager stays lightweight
@dataclass
class PlotManager:
    enabled: bool
    outdir: Path
    max_targets: int = 1
    n_samples: int = 1000
    _seen_targets: set = field(default_factory=set)

    def maybe_plot(self, *, target: str, model_label: str, band_label: str, sld: StellarLimbDarkening, band_data: Dict[str, Dict[str, np.ndarray]]) -> None:
        if not self.enabled:
            return

        key = (target, model_label)
        if key not in self._seen_targets:
            #if len(self._seen_targets) >= self.max_targets:
            #    return
            self._seen_targets.add(key)

        self.outdir.mkdir(parents=True, exist_ok=True)
        mus = np.linspace(0.0, 1.0, 150)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        fig.suptitle(f"{target} — {model_label} — {band_label}")

        law_order = [
            ("linear", axes[0, 0]),
            ("power1", axes[0, 1]),
            ("power2", axes[1, 0]),
            ("quadratic", axes[1, 1]),
        ]

        for law, ax in law_order:
            data = band_data.get(law, {})
            coeffs = data.get("coeffs")
            sigmas = data.get("sigmas")
            note = data.get("note", "")

            ax.set_title(law.capitalize())
            ax.set_xlabel(r"$\mu$")
            ax.set_ylabel(r"$I(\mu)/I(1)$")

            if getattr(sld, "mus", None) is not None and getattr(sld, "I_mu", None) is not None:
                ax.scatter(sld.mus, sld.I_mu, color=cm.inferno(0.1), s=10, label="model CLIV")

            if coeffs is None or sigmas is None or not np.all(np.isfinite(sigmas)):
                reason = "missing coeffs" if coeffs is None else "non-finite sigmas"
                print(f"      skip plotting {law} ({reason}); note={note}")
                ax.text(0.5, 0.5, f"No coefficients\n{note}", ha="center", va="center", transform=ax.transAxes)
                ax.legend(loc="best", fontsize=9)
                continue

            try:
                p16, p50, p84 = sample_intensity_profiles(law, coeffs, sigmas, mus=mus, n_samples=self.n_samples)
            except ValueError:
                print(f"      sampling failed for {law}; skipping")
                ax.text(0.5, 0.5, "Sampling failed", ha="center", va="center", transform=ax.transAxes)
                ax.legend(loc="best", fontsize=9)
                continue

            ax.plot(mus, p50, color=cm.inferno(0.6), label="median fit")
            ax.fill_between(mus, p16, p84, color=cm.inferno(0.6), alpha=0.3, label="16–84%")
            if note:
                ax.text(0.05, 0.05, note, transform=ax.transAxes, fontsize=8, color="red")
            ax.legend(loc="best", fontsize=9)

        safe_model = model_label.replace('/', '_')
        outfile = self.outdir / f"{target}_{safe_model}_{band_label}_ld_check.png"
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        print(f"      saving plot to {outfile}")
        fig.savefig(outfile, dpi=200)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Core coefficient extraction
# ---------------------------------------------------------------------------

def band_coeffs_exotic(
    sld: StellarLimbDarkening,
    wl_range: Iterable[float],
    wavelengths: np.ndarray,
    throughputs: np.ndarray,
    *,
    plot_callback: Callable[[Dict[str, Dict[str, np.ndarray]]], None] | None = None,
) -> Dict[str, Dict[str, np.ndarray]]:

    results: Dict[str, Dict[str, np.ndarray]] = {}

    lin_vals, lin_sigmas, lin_note = _safe_compute(
        sld.compute_linear_ld_coeffs,
        mode="custom", wavelength_range=wl_range,
        custom_wavelengths=wavelengths, custom_throughput=throughputs,
        mu_min=sld.mus.min(),
    )
    # return_sigmas=True (supported in newer ExoTiC-LD) supplies library-estimated uncertainties
    results["linear"] = {"coeffs": lin_vals, "sigmas": lin_sigmas, "note": lin_note}

    quad_vals, quad_sigmas, quad_note = _safe_compute(
        sld.compute_quadratic_ld_coeffs,
        mode="custom", wavelength_range=wl_range,
        custom_wavelengths=wavelengths, custom_throughput=throughputs,
        mu_min=sld.mus.min(),
    )
    results["quadratic"] = {"coeffs": quad_vals, "sigmas": quad_sigmas, "note": quad_note}

    pow1_vals, pow1_sigmas, pow1_note = _safe_compute(
        sld.compute_power1_ld_coeffs,
        mode="custom", wavelength_range=wl_range,
        custom_wavelengths=wavelengths, custom_throughput=throughputs,
        mu_min=sld.mus.min(),
    )
    results["power1"] = {"coeffs": pow1_vals, "sigmas": pow1_sigmas, "note": pow1_note}

    pow2_vals, pow2_sigmas, pow2_note = _safe_compute(
        sld.compute_power2_ld_coeffs,
        mode="custom", wavelength_range=wl_range,
        custom_wavelengths=wavelengths, custom_throughput=throughputs,
        mu_min=sld.mus.min(),
    )

    # Fallback: if ExoTiC-LD could not provide sigmas, record NaNs but keep note.
    if pow2_vals.size != 2:
        pow2_vals = _to_float_array([np.nan, np.nan])
        pow2_sigmas = _to_float_array([np.nan, np.nan])
        pow2_note = (pow2_note + ";unexpected_power2_shape").strip(";")

    results["power2"] = {"coeffs": pow2_vals, "sigmas": pow2_sigmas, "note": pow2_note}

    if plot_callback:
        plot_callback(results)

    return results


def compute_one_model(
    model_key: str,
    Teff: float,
    logg: float,
    MH: float,
    *,
    data_dir: str,
    plotter: Callable[[StellarLimbDarkening, Dict[str, Dict[str, np.ndarray]]], None] | None = None,
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:

    sld = StellarLimbDarkening(
        Teff=Teff,
        logg=logg,
        M_H=MH,
        ld_model=model_key,
        ld_data_path=data_dir,
        interpolate_type="trilinear",
    )

    bundle: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    def make_plot_callback(band_label: str) -> Callable[[Dict[str, Dict[str, np.ndarray]]], None] | None:
        if plotter is None:
            return None

        def callback(result: Dict[str, Dict[str, np.ndarray]]) -> None:
            plotter(sld, result, band_label)

        return callback

    bundle["H"] = band_coeffs_exotic(sld, H_RANGE, W_H, T_H, plot_callback=make_plot_callback("H"))
    bundle["K"] = band_coeffs_exotic(sld, K_RANGE, W_K, T_K, plot_callback=make_plot_callback("K"))
    return bundle


def propagate_law_uncertainties(
    model_key: str,
    Teff: float,
    logg: float,
    MH: float,
    *,
    sigT: float,
    sigG: float,
    sigZ: float,
    data_dir: str,
    n_draws: int,
    rng: np.random.Generator,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    draws = {
        "H": {"linear": [], "quadratic_u1": [], "quadratic_u2": [], "power1": [], "power2_p1": [], "power2_p2": []},
        "K": {"linear": [], "quadratic_u1": [], "quadratic_u2": [], "power1": [], "power2_p1": [], "power2_p2": []},
    }

    for _ in range(int(n_draws)):
        teff_i = float(rng.normal(Teff, sigT)) if np.isfinite(sigT) and sigT > 0 else float(Teff)
        logg_i = float(rng.normal(logg, sigG)) if np.isfinite(sigG) and sigG > 0 else float(logg)
        mh_i = float(rng.normal(MH, sigZ)) if np.isfinite(sigZ) and sigZ > 0 else float(MH)

        try:
            sld = StellarLimbDarkening(
                Teff=teff_i,
                logg=logg_i,
                M_H=mh_i,
                ld_model=model_key,
                ld_data_path=data_dir,
                interpolate_type="trilinear",
            )
            h_lin, _, _ = _safe_compute(
                sld.compute_linear_ld_coeffs,
                mode="custom", wavelength_range=H_RANGE,
                custom_wavelengths=W_H, custom_throughput=T_H,
                mu_min=sld.mus.min(),
            )
            k_lin, _, _ = _safe_compute(
                sld.compute_linear_ld_coeffs,
                mode="custom", wavelength_range=K_RANGE,
                custom_wavelengths=W_K, custom_throughput=T_K,
                mu_min=sld.mus.min(),
            )
            h_quad, _, _ = _safe_compute(
                sld.compute_quadratic_ld_coeffs,
                mode="custom", wavelength_range=H_RANGE,
                custom_wavelengths=W_H, custom_throughput=T_H,
                mu_min=sld.mus.min(),
            )
            k_quad, _, _ = _safe_compute(
                sld.compute_quadratic_ld_coeffs,
                mode="custom", wavelength_range=K_RANGE,
                custom_wavelengths=W_K, custom_throughput=T_K,
                mu_min=sld.mus.min(),
            )
            h_vals, _, _ = _safe_compute(
                sld.compute_power1_ld_coeffs,
                mode="custom", wavelength_range=H_RANGE,
                custom_wavelengths=W_H, custom_throughput=T_H,
                mu_min=sld.mus.min(),
            )
            k_vals, _, _ = _safe_compute(
                sld.compute_power1_ld_coeffs,
                mode="custom", wavelength_range=K_RANGE,
                custom_wavelengths=W_K, custom_throughput=T_K,
                mu_min=sld.mus.min(),
            )
            h_pow2, _, _ = _safe_compute(
                sld.compute_power2_ld_coeffs,
                mode="custom", wavelength_range=H_RANGE,
                custom_wavelengths=W_H, custom_throughput=T_H,
                mu_min=sld.mus.min(),
            )
            k_pow2, _, _ = _safe_compute(
                sld.compute_power2_ld_coeffs,
                mode="custom", wavelength_range=K_RANGE,
                custom_wavelengths=W_K, custom_throughput=T_K,
                mu_min=sld.mus.min(),
            )
        except Exception:
            continue

        if h_lin.size >= 1 and np.isfinite(h_lin[0]):
            draws["H"]["linear"].append(float(h_lin[0]))
        if k_lin.size >= 1 and np.isfinite(k_lin[0]):
            draws["K"]["linear"].append(float(k_lin[0]))
        if h_quad.size >= 2 and np.all(np.isfinite(h_quad[:2])):
            draws["H"]["quadratic_u1"].append(float(h_quad[0]))
            draws["H"]["quadratic_u2"].append(float(h_quad[1]))
        if k_quad.size >= 2 and np.all(np.isfinite(k_quad[:2])):
            draws["K"]["quadratic_u1"].append(float(k_quad[0]))
            draws["K"]["quadratic_u2"].append(float(k_quad[1]))
        if h_vals.size >= 1 and np.isfinite(h_vals[0]):
            draws["H"]["power1"].append(float(h_vals[0]))
        if k_vals.size >= 1 and np.isfinite(k_vals[0]):
            draws["K"]["power1"].append(float(k_vals[0]))
        if h_pow2.size >= 2 and np.all(np.isfinite(h_pow2[:2])):
            draws["H"]["power2_p1"].append(float(h_pow2[0]))
            draws["H"]["power2_p2"].append(float(h_pow2[1]))
        if k_pow2.size >= 2 and np.all(np.isfinite(k_pow2[:2])):
            draws["K"]["power2_p1"].append(float(k_pow2[0]))
            draws["K"]["power2_p2"].append(float(k_pow2[1]))

    return {
        band: {name: _summary_stats(vals) for name, vals in band_draws.items()}
        for band, band_draws in draws.items()
    }


# ---------------------------------------------------------------------------
# CSV assembly
# ---------------------------------------------------------------------------

def build_catalog(args: argparse.Namespace) -> None:
    raw = pd.read_csv(args.targets)
    if getattr(args, "limit", None) is not None:
        raw = raw.head(int(args.limit))

    c_target = _col(raw, "Target") or _col(raw, "Name") or raw.columns[0]
    c_teff = _col(raw, "Teff")
    c_eteff = _col(raw, "e_Teff")
    c_logg = _col(raw, "logg")
    c_elogg = _col(raw, "e_logg")
    c_feh = _col(raw, "FeH") or _col(raw, "M_H") or _col(raw, "[M/H]")
    c_efeh = _col(raw, "e_FeH")
    c_hd = (
        _col(raw, "HD")
        or _col(raw, "HD_number")
        or _col(raw, "Henry Draper")
        or _col(raw, "HDnum")
        or _col(raw, "hd")
    )

    if not c_teff or not c_logg or not c_feh:
        raise SystemExit("Input must contain Teff, logg, and FeH columns (case-insensitive).")

    plot_manager = PlotManager(
        enabled=args.plot_check,
        outdir=Path(args.plot_dir),
        max_targets=args.plot_max,
        n_samples=args.plot_samples,
    )

    def clean_target_fallback(row: pd.Series) -> str:
        if c_target and pd.notna(row.get(c_target, np.nan)):
            t = str(row[c_target]).strip()
            if t and t.lower() not in {"nan", "none", "null"}:
                return t.replace(" ", "_")
        if c_hd and pd.notna(row.get(c_hd, np.nan)):
            hdval = row[c_hd]
            try:
                num = int(float(hdval))
            except Exception:
                match = re.search(r"\d+", str(hdval))
                num = int(match.group()) if match else None
            if num is not None:
                return f"HD_{num}"
        return "Unknown"

    rows_long = []
    out_dir = Path(args.outdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    def make_plotter(target: str, model_label: str) -> Callable[[StellarLimbDarkening, Dict[str, Dict[str, np.ndarray]], str], None]:
        def plotter(sld: StellarLimbDarkening, band_result: Dict[str, Dict[str, np.ndarray]], band_label: str) -> None:
            plot_manager.maybe_plot(
                target=target,
                model_label=model_label,
                band_label=band_label,
                sld=sld,
                band_data=band_result,
            )

        return plotter

    for _, row in raw.iterrows():
        target_name = clean_target_fallback(row)
        print(f"=== Target: {target_name}")

        Teff = float(row[c_teff])
        logg = float(row[c_logg])
        MH = float(row[c_feh])
        print(f"  Input params: Teff={Teff}, logg={logg}, [Fe/H]={MH}")

        sigT = float(row[c_eteff]) if c_eteff and pd.notna(row[c_eteff]) and float(row[c_eteff]) > 0 else DEFAULT_SIG_T
        sigG = float(row[c_elogg]) if c_elogg and pd.notna(row[c_elogg]) and float(row[c_elogg]) > 0 else DEFAULT_SIG_G
        sigZ = float(row[c_efeh]) if c_efeh and pd.notna(row[c_efeh]) and float(row[c_efeh]) > 0 else DEFAULT_SIG_Z
        print(f"  Sigmas: dTeff={sigT}, dlogg={sigG}, d[Fe/H]={sigZ}")

        for model_key, model_label in MODELS:
            try:
                print(f"    Model: {model_label}")
                bundle = compute_one_model(
                    model_key,
                    Teff,
                    logg,
                    MH,
                    data_dir=args.data_dir,
                    plotter=make_plotter(target_name, model_label) if args.plot_check else None,
                )
                for band_label in ("H", "K"):
                    band_data = bundle.get(band_label, {})
                    if not band_data:
                        print(f"      {band_label}: no data")
                        continue
                    for law in ("linear", "power1", "power2", "quadratic"):
                        entry = band_data.get(law, {})
                        coeffs = entry.get("coeffs")
                        sigmas = entry.get("sigmas")
                        note = entry.get("note", "")
                        print(
                            f"      {band_label} {law}: coeffs={coeffs} sigmas={sigmas} note={note}"
                        )
                if args.propagate_stellar_errors:
                    propagated = propagate_law_uncertainties(
                        model_key,
                        Teff,
                        logg,
                        MH,
                        sigT=sigT,
                        sigG=sigG,
                        sigZ=sigZ,
                        data_dir=args.data_dir,
                        n_draws=args.propagate_draws,
                        rng=rng,
                    )
                    for band_label in ("H", "K"):
                        stats = propagated[band_label]["power1"]
                        print(
                            f"      {band_label} power1 propagated: "
                            f"std={stats['std']} p16={stats['p16']} "
                            f"p50={stats['p50']} p84={stats['p84']} n={stats['n']}"
                        )
                else:
                    empty = _summary_stats([])
                    propagated = {
                        "H": {
                            "linear": empty, "quadratic_u1": empty, "quadratic_u2": empty,
                            "power1": empty, "power2_p1": empty, "power2_p2": empty,
                        },
                        "K": {
                            "linear": empty, "quadratic_u1": empty, "quadratic_u2": empty,
                            "power1": empty, "power2_p1": empty, "power2_p2": empty,
                        },
                    }
            except Exception as exc:  # pragma: no cover - log failure and continue
                print(f"    Model {model_label} failed: {exc}")
                for inst in ("MIRCX", "MYSTIC"):
                    rows_long.append({
                        "Target": target_name,
                        "Teff": Teff,
                        "logg": logg,
                        "M_H": MH,
                        "sigma_T": sigT,
                        "sigma_logg": sigG,
                        "sigma_M_H": sigZ,
                        "ld_model": model_key,
                        "Model": model_label,
                        "Instrument": inst,
                        "Law": "error",
                        "note": f"skipped ({exc})",
                        "n_mc_used": 0,
                        "a": np.nan,
                        "a_err": np.nan,
                        "u1": np.nan,
                        "u1_err": np.nan,
                        "u2": np.nan,
                        "u2_err": np.nan,
                        "p1": np.nan,
                        "p1_err": np.nan,
                        "p2": np.nan,
                        "p2_err": np.nan,
                        "p1_1p": np.nan,
                        "p1_std": np.nan,
                        "p1_p16": np.nan,
                        "p1_p50": np.nan,
                        "p1_p84": np.nan,
                    })
                continue

            for inst, band in (("MIRCX", "H"), ("MYSTIC", "K")):
                band_data = bundle[band]

                linear = band_data.get("linear", {})
                quadratic = band_data.get("quadratic", {})
                power2 = band_data.get("power2", {})
                power1 = band_data.get("power1", {})
                prop = propagated[band]

                rows_long.extend([
                    {
                        "Target": target_name,
                        "Teff": Teff,
                        "logg": logg,
                        "M_H": MH,
                        "sigma_T": sigT,
                        "sigma_logg": sigG,
                        "sigma_M_H": sigZ,
                        "ld_model": model_key,
                        "Model": model_label,
                        "Instrument": inst,
                        "Law": "linear",
                        "a": float(linear.get("coeffs", [np.nan])[0]),
                        "a_err": float(linear.get("sigmas", [np.nan])[0]),
                        "u1": np.nan,
                        "u1_err": np.nan,
                        "u2": np.nan,
                        "u2_err": np.nan,
                        "p1": np.nan,
                        "p1_err": np.nan,
                        "p2": np.nan,
                        "p2_err": np.nan,
                        "p1_1p": np.nan,
                        "a_std": float(prop["linear"]["std"]),
                        "a_p16": float(prop["linear"]["p16"]),
                        "a_p50": float(prop["linear"]["p50"]),
                        "a_p84": float(prop["linear"]["p84"]),
                        "u1_std": np.nan,
                        "u1_p16": np.nan,
                        "u1_p50": np.nan,
                        "u1_p84": np.nan,
                        "u2_std": np.nan,
                        "u2_p16": np.nan,
                        "u2_p50": np.nan,
                        "u2_p84": np.nan,
                        "p1_std": np.nan,
                        "p1_p16": np.nan,
                        "p1_p50": np.nan,
                        "p1_p84": np.nan,
                        "p2_std": np.nan,
                        "p2_p16": np.nan,
                        "p2_p50": np.nan,
                        "p2_p84": np.nan,
                        "n_mc_used": int(prop["linear"]["n"]),
                        "note": linear.get("note", ""),
                    },
                    {
                        "Target": target_name,
                        "Teff": Teff,
                        "logg": logg,
                        "M_H": MH,
                        "sigma_T": sigT,
                        "sigma_logg": sigG,
                        "sigma_M_H": sigZ,
                        "ld_model": model_key,
                        "Model": model_label,
                        "Instrument": inst,
                        "Law": "quadratic",
                        "a": np.nan,
                        "a_err": np.nan,
                        "u1": float(quadratic.get("coeffs", [np.nan, np.nan])[0]),
                        "u1_err": float(quadratic.get("sigmas", [np.nan, np.nan])[0]),
                        "u2": float(quadratic.get("coeffs", [np.nan, np.nan])[1]),
                        "u2_err": float(quadratic.get("sigmas", [np.nan, np.nan])[1]),
                        "p1": np.nan,
                        "p1_err": np.nan,
                        "p2": np.nan,
                        "p2_err": np.nan,
                        "p1_1p": np.nan,
                        "a_std": np.nan,
                        "a_p16": np.nan,
                        "a_p50": np.nan,
                        "a_p84": np.nan,
                        "u1_std": float(prop["quadratic_u1"]["std"]),
                        "u1_p16": float(prop["quadratic_u1"]["p16"]),
                        "u1_p50": float(prop["quadratic_u1"]["p50"]),
                        "u1_p84": float(prop["quadratic_u1"]["p84"]),
                        "u2_std": float(prop["quadratic_u2"]["std"]),
                        "u2_p16": float(prop["quadratic_u2"]["p16"]),
                        "u2_p50": float(prop["quadratic_u2"]["p50"]),
                        "u2_p84": float(prop["quadratic_u2"]["p84"]),
                        "p1_std": np.nan,
                        "p1_p16": np.nan,
                        "p1_p50": np.nan,
                        "p1_p84": np.nan,
                        "p2_std": np.nan,
                        "p2_p16": np.nan,
                        "p2_p50": np.nan,
                        "p2_p84": np.nan,
                        "n_mc_used": min(int(prop["quadratic_u1"]["n"]), int(prop["quadratic_u2"]["n"])),
                        "note": quadratic.get("note", ""),
                    },
                    {
                        "Target": target_name,
                        "Teff": Teff,
                        "logg": logg,
                        "M_H": MH,
                        "sigma_T": sigT,
                        "sigma_logg": sigG,
                        "sigma_M_H": sigZ,
                        "ld_model": model_key,
                        "Model": model_label,
                        "Instrument": inst,
                        "Law": "power2",
                        "a": np.nan,
                        "a_err": np.nan,
                        "u1": np.nan,
                        "u1_err": np.nan,
                        "u2": np.nan,
                        "u2_err": np.nan,
                        "p1": float(power2.get("coeffs", [np.nan, np.nan])[0]),
                        "p1_err": float(power2.get("sigmas", [np.nan, np.nan])[0]),
                        "p2": float(power2.get("coeffs", [np.nan, np.nan])[1]),
                        "p2_err": float(power2.get("sigmas", [np.nan, np.nan])[1]),
                        "p1_1p": np.nan,
                        "a_std": np.nan,
                        "a_p16": np.nan,
                        "a_p50": np.nan,
                        "a_p84": np.nan,
                        "u1_std": np.nan,
                        "u1_p16": np.nan,
                        "u1_p50": np.nan,
                        "u1_p84": np.nan,
                        "u2_std": np.nan,
                        "u2_p16": np.nan,
                        "u2_p50": np.nan,
                        "u2_p84": np.nan,
                        "p1_std": float(prop["power2_p1"]["std"]),
                        "p1_p16": float(prop["power2_p1"]["p16"]),
                        "p1_p50": float(prop["power2_p1"]["p50"]),
                        "p1_p84": float(prop["power2_p1"]["p84"]),
                        "p2_std": float(prop["power2_p2"]["std"]),
                        "p2_p16": float(prop["power2_p2"]["p16"]),
                        "p2_p50": float(prop["power2_p2"]["p50"]),
                        "p2_p84": float(prop["power2_p2"]["p84"]),
                        "n_mc_used": min(int(prop["power2_p1"]["n"]), int(prop["power2_p2"]["n"])),
                        "note": power2.get("note", ""),
                    },
                    {
                        "Target": target_name,
                        "Teff": Teff,
                        "logg": logg,
                        "M_H": MH,
                        "sigma_T": sigT,
                        "sigma_logg": sigG,
                        "sigma_M_H": sigZ,
                        "ld_model": model_key,
                        "Model": model_label,
                        "Instrument": inst,
                        "Law": "power1",
                        "a": np.nan,
                        "a_err": np.nan,
                        "u1": np.nan,
                        "u1_err": np.nan,
                        "u2": np.nan,
                        "u2_err": np.nan,
                        "p1": float(power1.get("coeffs", [np.nan])[0]),
                        "p1_err": float(power1.get("sigmas", [np.nan])[0]),
                        "p2": np.nan,
                        "p2_err": np.nan,
                        "p1_1p": float(power1.get("coeffs", [np.nan])[0]),
                        "a_std": np.nan,
                        "a_p16": np.nan,
                        "a_p50": np.nan,
                        "a_p84": np.nan,
                        "u1_std": np.nan,
                        "u1_p16": np.nan,
                        "u1_p50": np.nan,
                        "u1_p84": np.nan,
                        "u2_std": np.nan,
                        "u2_p16": np.nan,
                        "u2_p50": np.nan,
                        "u2_p84": np.nan,
                        "p1_std": float(prop["power1"]["std"]),
                        "p1_p16": float(prop["power1"]["p16"]),
                        "p1_p50": float(prop["power1"]["p50"]),
                        "p1_p84": float(prop["power1"]["p84"]),
                        "p2_std": np.nan,
                        "p2_p16": np.nan,
                        "p2_p50": np.nan,
                        "p2_p84": np.nan,
                        "n_mc_used": int(prop["power1"]["n"]),
                        "note": power1.get("note", ""),
                    },
                ])

    df_long = pd.DataFrame(rows_long)
    out_long = out_dir / f"ldc_all_models_HK_LONG_{timestamp()}.csv"
    df_long.to_csv(out_long, index=False)
    print(f"[saved] {out_long}  rows={len(df_long)}")

    public_rows: dict[str, dict] = {}
    for _, item in df_long.iterrows():
        target = item["Target"]
        row = public_rows.setdefault(
            target,
            {
                "target": target,
                "teff": item.get("Teff", np.nan),
                "logg": item.get("logg", np.nan),
                "feh": item.get("M_H", np.nan),
                "sigma_teff": item.get("sigma_T", np.nan),
                "sigma_logg": item.get("sigma_logg", np.nan),
                "sigma_feh": item.get("sigma_M_H", np.nan),
            },
        )
        band = INST_BAND.get(str(item.get("Instrument", "")).upper(), str(item.get("Instrument", "")))
        grid = GRID_PUBLIC.get(str(item.get("ld_model", "")).lower(), str(item.get("ld_model", "")))
        law = str(item.get("Law", ""))
        values = []
        if law == "linear":
            values = [("a", item.get("a", np.nan), item.get("a_err", np.nan), item.get("a_std", np.nan))]
        elif law == "power1":
            values = [("alpha", item.get("p1", np.nan), item.get("p1_err", np.nan), item.get("p1_std", np.nan))]
        elif law == "quadratic":
            values = [
                ("u1", item.get("u1", np.nan), item.get("u1_err", np.nan), item.get("u1_std", np.nan)),
                ("u2", item.get("u2", np.nan), item.get("u2_err", np.nan), item.get("u2_std", np.nan)),
            ]
        elif law == "power2":
            values = [
                ("p1", item.get("p1", np.nan), item.get("p1_err", np.nan), item.get("p1_std", np.nan)),
                ("alpha1", item.get("p2", np.nan), item.get("p2_err", np.nan), item.get("p2_std", np.nan)),
            ]
        for name, val, err, std in values:
            stem = f"{name}_I_{band}_{grid}"
            row[stem] = val
            row[f"{stem}_err"] = err
            row[f"{stem}_std"] = std

    public = pd.DataFrame(public_rows.values())
    public_out = out_dir / f"{OUTPUT_FILE}_LONG.csv"
    public.to_csv(public_out, index=False)
    print(f"[saved] {public_out}  rows={len(public)}")

    metrics = [
        "a", "a_err", "u1", "u1_err", "u2", "u2_err",
        "p1", "p1_err", "p2", "p2_err", "p1_1p",
        "a_std", "a_p16", "a_p50", "a_p84",
        "u1_std", "u1_p16", "u1_p50", "u1_p84",
        "u2_std", "u2_p16", "u2_p50", "u2_p84",
        "p1_std", "p1_p16", "p1_p50", "p1_p84",
        "p2_std", "p2_p16", "p2_p50", "p2_p84",
    ]
    df_subset = df_long[["Target", "Teff", "logg", "M_H", "ld_model", "Instrument", "Law"] + metrics].copy()

    wide = df_subset.pivot_table(
        index=["Target", "Teff", "logg", "M_H"],
        columns=["Law", "ld_model", "Instrument"],
        values=metrics,
        aggfunc="first",
    )

    wide.columns = [
        f"{metric}__{law}__{model}__{inst}"
        for metric, law, model, inst in wide.columns.to_flat_index()
    ]

    wide = wide.reset_index()
    out_wide = out_dir / OUTPUT_FILE
    wide.to_csv(out_wide, index=False)
    print(f"[saved] {out_wide}  rows={len(wide)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build CHARA limb-darkening coefficient catalogs via ExoTiC-LD.")
    parser.add_argument("--targets", default=DEFAULT_TARGETS_CSV, help="Input CSV with Teff/logg/[Fe/H] columns.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="ExoTiC-LD data cache directory.")
    parser.add_argument("--outdir", default=DEFAULT_OUTPUT_DIR, help="Directory for output CSV files (default: csv/).")
    parser.add_argument("--plot-check", action="store_true", help="Generate diagnostic plots for the first few targets.")
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR, help="Directory for diagnostic plots (default: ldc_check_plots).")
    parser.add_argument("--plot-max", type=int, default=1, help="Maximum number of targets to plot when --plot-check is set.")
    parser.add_argument("--plot-samples", type=int, default=1000, help="Number of MC samples for diagnostic intensity envelopes.")
    parser.add_argument("--propagate-stellar-errors", action="store_true", help="Propagate Teff/logg/[Fe/H] uncertainties into additional power-1 summary columns.")
    parser.add_argument("--propagate-draws", type=int, default=DEFAULT_PROP_DRAWS, help="Number of Monte Carlo draws when --propagate-stellar-errors is enabled.")
    parser.add_argument("--seed", type=int, default=12345, help="Random seed for propagated stellar-parameter draws.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N targets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_catalog(args)


if __name__ == "__main__":
    main()
