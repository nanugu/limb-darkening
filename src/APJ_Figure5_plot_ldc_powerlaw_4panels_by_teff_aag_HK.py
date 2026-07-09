
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure generator for power-1 LDC (alpha) vs Teff.

Produces a single 2x2 mosaic:
  Top row   : H and K CHARA coefficients vs Teff, overplotted with four grids
  Bottom row: CHARA-Stagger residuals for H and K

Plot styling is kept visually consistent with the paper's H-vs-K alpha figure:
  - CHARA observations are filled Teff-colored circles with black edges
  - Model grids use open markers plus grayscale trend loci
  - No target-name labels by default; use --label-targets for diagnostic labels
  - A single shared Teff colorbar is used across the figure

Input CSV columns follow the HK combined/per-epoch merge scripts:
  Teff, Target_catalog, H_PL_alpha, H_PL_alpha_err,
  K_PL_alpha, K_PL_alpha_err, and model-grid columns like
  p1__power1__{grid}__{band} or p1_1p__power1__{grid}__{band}.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

CSV_DEFAULTS = [
    "csv/merged_for_plots_merged.csv",
]
OUT_DIR  = "ldc_powerlaw_4panels_by_teff"
GRIDS    = ["stagger", "kurucz", "mps1", "mps2"]
#GRIDS    = ["mps2"]
BANDS    = ["MIRCX", "MYSTIC"]  # H and K
TEFF     = "Teff"
NAME     = "Target_catalog"
CHARA_Y  = {
    "MIRCX":  {"y": "H_PL_alpha", "yerr": ["H_PL_alpha_err", "H_PL_alpha_err_ivw"], "title": "H-band (MIRCX)"},
    "MYSTIC": {"y": "K_PL_alpha", "yerr": ["K_PL_alpha_err", "K_PL_alpha_err_ivw"], "title": "K-band (MYSTIC)"},
}
GRID_PAT = [
    "p1__power1__{grid}__{band}",
    "p1_1p__power1__{grid}__{band}",
]

# Style (larger fonts, clearer markers)
SCATTER_S_DATA   = 52
SCATTER_S_MODEL  = 34
PANEL_DPI        = 300
PANEL_SIZE       = (7, 5)   # inches
CMAP_NAME        = "turbo_r"

GRID_STYLES = {
    "stagger": {"label": r"Fit to Stagger $I(\mu)$", "color": "black", "marker": "o", "linestyle": "-",  "zorder": 5},
    "kurucz":  {"label": r"Fit to Kurucz $I(\mu)$",  "color": "0.35",  "marker": "s", "linestyle": "--", "zorder": 4},
    "mps1":    {"label": r"Fit to MPS1 $I(\mu)$",    "color": "0.55",  "marker": "^", "linestyle": "-.", "zorder": 3},
    "mps2":    {"label": r"Fit to MPS2 $I(\mu)$",    "color": "0.72",  "marker": "D", "linestyle": ":",  "zorder": 2},
}

# Global font sizing suitable for print/PDF
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
})

def get_cmap(name):
    try:
        return mpl.colormaps.get_cmap(name)
    except Exception:
        return mpl.cm.get_cmap(name)

def pick_grid_col(columns, grid, band):
    for pat in GRID_PAT:
        col = pat.format(grid=grid, band=band)
        if col in columns:
            return col
    return None

def pick_yerr_col(columns, band):
    yerr = CHARA_Y[band].get("yerr")
    candidates = yerr if isinstance(yerr, (list, tuple)) else [yerr]
    for col in candidates:
        if col in columns:
            return col
    return None

def short_target_label(value):
    """Compact target label for diagnostic point annotations."""
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
        "omi": r"$\omicron$",
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

def annotate_points(ax, x, y, labels, *, fontsize=6.5, color="0.12", max_labels=None):
    """Annotate plotted points with small deterministic offsets."""
    count = 0
    offsets = [(4, 4), (4, -7), (-18, 4), (-18, -7), (7, 9), (-22, 9)]
    for i, (xi, yi, label) in enumerate(zip(x, y, labels)):
        if max_labels is not None and count >= max_labels:
            break
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        lab = short_target_label(label)
        if not lab:
            continue
        dx, dy = offsets[i % len(offsets)]
        ax.annotate(
            lab, xy=(xi, yi), xytext=(dx, dy), textcoords="offset points",
            fontsize=fontsize, color=color, alpha=0.88, zorder=20,
            ha="left" if dx >= 0 else "right", va="center",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.45, pad=0.4),
        )
        count += 1

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def robust_sigma_mad(x):
    """Robust scatter estimate using MAD (scaled to match std for Gaussian)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    return 1.4826 * mad

def intrinsic_scatter_sigma_int(resid, sigma, dof=None):
    """Estimate extra scatter term sigma_int such that chi^2 ~= dof.

    Solves: sum_i resid_i^2 / (sigma_i^2 + sigma_int^2) = dof, with sigma_int >= 0.
    Returns np.nan if insufficient data.
    """
    r = np.asarray(resid, dtype=float)
    s = np.asarray(sigma, dtype=float)
    m = np.isfinite(r) & np.isfinite(s) & (s > 0)
    r = r[m]
    s = s[m]
    if r.size < 2:
        return np.nan
    if dof is None:
        dof = r.size - 1

    def f(sig_int):
        return np.sum((r * r) / (s * s + sig_int * sig_int)) - dof

    f0 = f(0.0)
    if f0 <= 0:
        return 0.0

    hi = max(np.nanmax(np.abs(r)), np.nanmedian(s)) if np.isfinite(r).any() else 1.0
    if not np.isfinite(hi) or hi <= 0:
        hi = 1.0
    # Expand upper bracket until f(hi) <= 0 (or we give up)
    for _ in range(60):
        if f(hi) <= 0:
            break
        hi *= 1.5
    else:
        return np.nan

    lo = 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

def summarize_dispersion(df, grids=("kurucz", "stagger"), bands=("MIRCX", "MYSTIC")):
    """Print (and return) copy-pasteable dispersion metrics for Δalpha = CHARA - model."""
    lines = []
    lines.append("=== Dispersion summary for power-1 LDC residuals: Delta alpha = alpha_CHARA - alpha_model ===")
    lines.append("Notes: sigma_res(MAD) is robust scatter; chi2_nu uses CHARA sigma_alpha only (model errors not included).")
    lines.append("       sigma_int is additional scatter term so that chi2_nu~=1 (added in quadrature to sigma_alpha).")
    lines.append("       frac is reported as |Delta alpha| / alpha_model (median), matching manuscript wording.")

    for band in bands:
        for grid in grids:
            dd, ycol, yerr, gcol = _get_dd(df, band, grid)
            if dd is None:
                continue
            resid = (dd[ycol] - dd[gcol]).to_numpy(dtype=float)
            n = int(np.isfinite(resid).sum())

            # fractional residual relative to model (for easy interpretation)
            alpha_model = dd[gcol].to_numpy(dtype=float)
            mf = np.isfinite(resid) & np.isfinite(alpha_model) & (alpha_model != 0)
            if mf.any():
                frac = resid[mf] / alpha_model[mf]
                median_abs_frac = float(np.nanmedian(np.abs(frac)))
                median_frac = float(np.nanmedian(frac))
            else:
                median_abs_frac = np.nan
                median_frac = np.nan

            # formal errors (CHARA only)
            if yerr in dd.columns and dd[yerr].notna().any():
                sigma = dd[yerr].to_numpy(dtype=float)
            else:
                sigma = np.full_like(resid, np.nan)

            m = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
            resid_s = resid[m]
            sigma_s = sigma[m]

            alpha_chara = dd[ycol].to_numpy(dtype=float)
            m_pct = np.isfinite(alpha_chara) & np.isfinite(sigma) & (sigma > 0) & (alpha_chara != 0)
            sigma_pct = 100.0 * sigma[m_pct] / np.abs(alpha_chara[m_pct]) if m_pct.any() else np.array([], dtype=float)

            med = np.nanmedian(resid) if np.isfinite(resid).any() else np.nan
            rms = np.sqrt(np.nanmean(resid * resid)) if np.isfinite(resid).any() else np.nan
            sig_std = np.nanstd(resid) if np.isfinite(resid).any() else np.nan
            sig_mad = robust_sigma_mad(resid)
            med_sig = np.nanmedian(sigma_s) if sigma_s.size else np.nan
            med_sig_pct = np.nanmedian(sigma_pct) if sigma_pct.size else np.nan
            dof = max((resid_s.size - 1), 0)
            if dof > 0:
                chi2_nu = float(np.sum((resid_s / sigma_s) ** 2) / dof)
                sig_int = intrinsic_scatter_sigma_int(resid_s, sigma_s, dof=dof)
                # normalized residual width (use robust scatter)
                z = resid_s / np.sqrt(sigma_s * sigma_s + (sig_int * sig_int if np.isfinite(sig_int) else 0.0))
                z_mad = robust_sigma_mad(z)
            else:
                chi2_nu = np.nan
                sig_int = np.nan
                z_mad = np.nan

            band_title = CHARA_Y.get(band, {}).get("title", band)
            lines.append("")
            lines.append(f"{band_title} — {grid.capitalize()}: N={n}")
            lines.append(f"  median(Δα)={med:+.4f}, RMS(|Δα|)={rms:.4f}")
            lines.append(f"  σ_res(std)={sig_std:.4f}, σ_res(MAD)={sig_mad:.4f}")
            if np.isfinite(median_abs_frac):
                lines.append(
                    f"  median(|Δα|/α_model)={median_abs_frac:.3f} ({100.0*median_abs_frac:.1f}%), "
                    f"median(Δα/α_model)={median_frac:+.3f} ({100.0*median_frac:+.1f}%)"
                )
            if np.isfinite(med_sig):
                if np.isfinite(med_sig_pct):
                    lines.append(f"  median(σ_α)={med_sig:.4f} ({med_sig_pct:.1f}%), χ2_nu={chi2_nu:.2f}, σ_int={sig_int:.4f}, σ_z(MAD)={z_mad:.2f}")
                else:
                    lines.append(f"  median(σ_α)={med_sig:.4f}, χ2_nu={chi2_nu:.2f}, σ_int={sig_int:.4f}, σ_z(MAD)={z_mad:.2f}")
                lines.append("  LaTeX sentence:")
                lines.append(
                    "    "
                    f"For {band_title}, the residuals "
                    f"$\\Delta\\alpha\\equiv\\alpha_\\mathrm{{CHARA}}-\\alpha_\\mathrm{{{grid}}}$ "
                    f"have $\\sigma_\\mathrm{{res}}^\\mathrm{{MAD}}\\approx{sig_mad:.3f}$ "
                    f"(median $\\sigma_\\alpha\\approx{med_sig:.3f}$); "
                    f"an additional $\\sigma_\\mathrm{{int}}\\approx{sig_int:.3f}$ is required to obtain "
                    f"$\\chi_\\nu^2\\approx 1$."
                )
            else:
                lines.append("  (No σ_α column available; χ2_nu and σ_int not computed.)")

    return "\n".join(lines)

def _summarize_percentiles(x, *, p=(16, 50, 84)):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return None
    return tuple(np.percentile(x, p))

def summarize_hk_contrast(df, grids=("kurucz", "stagger")):
    """Summarize the expected vs observed H–K contrast in power-law alpha."""
    lines = []
    lines.append("=== H–K contrast summary for power-1 LDC alpha ===")
    lines.append("Definitions: Δα_HK = α_H - α_K; frac_drop = (α_H - α_K)/α_H.")
    lines.append("")

    # CHARA (observed)
    ah = to_num(df.get(CHARA_Y["MIRCX"]["y"]))
    ak = to_num(df.get(CHARA_Y["MYSTIC"]["y"]))
    m = ah.notna() & ak.notna() & (ah > 0) & (ak > 0)
    dhk = (ah[m] - ak[m]).to_numpy(dtype=float)
    fdrop = (dhk / ah[m]).to_numpy(dtype=float)
    q = _summarize_percentiles(fdrop)
    if q:
        p16, p50, p84 = q
        lines.append(f"CHARA: N={int(m.sum())}")
        lines.append(f"  median(Δα_HK) = {np.median(dhk):.4f}")
        lines.append(f"  median frac_drop = {100*p50:.1f}% (16–84%: {100*p16:.1f}% to {100*p84:.1f}%)")
    else:
        lines.append("CHARA: insufficient data for H–K contrast.")

    # Models (expected) under matched priors/passbands
    for grid in grids:
        gH = pick_grid_col(df.columns, grid, "MIRCX")
        gK = pick_grid_col(df.columns, grid, "MYSTIC")
        if not gH or not gK:
            lines.append(f"{grid.capitalize()}: missing model columns for both H and K.")
            continue
        mh = to_num(df[gH])
        mk = to_num(df[gK])
        mm = mh.notna() & mk.notna() & (mh > 0) & (mk > 0)
        dhk_m = (mh[mm] - mk[mm]).to_numpy(dtype=float)
        fdrop_m = (dhk_m / mh[mm]).to_numpy(dtype=float)
        q = _summarize_percentiles(fdrop_m)
        if not q:
            lines.append(f"{grid.capitalize()}: insufficient data for H–K contrast.")
            continue
        p16, p50, p84 = q
        lines.append(f"{grid.capitalize()}: N={int(mm.sum())}")
        lines.append(f"  median(Δα_HK) = {np.median(dhk_m):.4f}")
        lines.append(f"  median frac_drop = {100*p50:.1f}% (16–84%: {100*p16:.1f}% to {100*p84:.1f}%)")

    lines.append("")
    lines.append("LaTeX sentence:")
    if _summarize_percentiles(fdrop):
        p16, p50, p84 = _summarize_percentiles(fdrop)
        parts = [f"the median fractional decrease in the power-law coefficient from $H$ to $K$ is {100*p50:.0f}\\% (16--84\\%: {100*p16:.0f}\\%--{100*p84:.0f}\\%) in our CHARA sample"]
        for grid in grids:
            gH = pick_grid_col(df.columns, grid, "MIRCX")
            gK = pick_grid_col(df.columns, grid, "MYSTIC")
            if not gH or not gK:
                continue
            mh = to_num(df[gH]); mk = to_num(df[gK])
            mm = mh.notna() & mk.notna() & (mh > 0) & (mk > 0)
            fdrop_m = ((mh[mm] - mk[mm]) / mh[mm]).to_numpy(dtype=float)
            q = _summarize_percentiles(fdrop_m)
            if not q:
                continue
            _, p50m, _ = q
            parts.append(f"compared to {100*p50m:.0f}\\% predicted by the {grid.upper()} grid")
        lines.append("  " + "; ".join(parts) + ".")
    else:
        lines.append("  (insufficient data)")

    return "\n".join(lines)

def resolve_csv_path(cli_csv=None):
    if cli_csv:
        return cli_csv if os.path.exists(cli_csv) else None
    for p in CSV_DEFAULTS:
        if os.path.exists(p):
            return p
    return None

def band_teff_range(df):
    """Global Teff range (for x and color) within a band."""
    t = to_num(df.get(TEFF))
    if t is None or not t.notna().any():
        return None, None
    tmin = np.nanpercentile(t.dropna(), 2)
    tmax = np.nanpercentile(t.dropna(), 98)
    return tmin, tmax

def _binned_trend(x, y, nbins=10):
    """Simple binned median trend for guidance."""
    if len(x) < 3:
        return None, None
    x = np.asarray(x); y = np.asarray(y)
    order = np.argsort(x)
    x = x[order]; y = y[order]
    bins = np.linspace(x.min(), x.max(), nbins+1)
    xm, ym = [], []
    for i in range(nbins):
        m = (x >= bins[i]) & (x < bins[i+1]) if i < nbins-1 else (x >= bins[i]) & (x <= bins[i+1])
        if m.sum() >= 2:
            xm.append(np.nanmedian(x[m]))
            ym.append(np.nanmedian(y[m]))
    if not xm:
        return None, None
    return np.array(xm), np.array(ym)

def plot_panel(df, band, grid, xlim, tnorm, cmap, out_png, add_cbar=False, mode="abs"):
    """Create a single panel PNG.
    mode="abs": plot absolute alpha vs Teff with model dots + dashed trend line.
    mode="resid": plot residuals Δalpha = alpha(CHARA) - alpha(model) with ±1σ band.
    """
    ycol   = CHARA_Y[band]["y"]
    yerr   = pick_yerr_col(df.columns, band)
    gcol   = pick_grid_col(df.columns, grid, band)

    if gcol is None or ycol not in df.columns:
        return False, f"[skip] {band} {grid}: missing column(s)"

    cols = [TEFF, NAME, ycol, gcol] + ([yerr] if yerr else [])
    dd = df[cols].copy()
    dd[TEFF] = to_num(dd[TEFF])
    dd[ycol] = to_num(dd[ycol])
    dd[gcol] = to_num(dd[gcol])
    if yerr in dd.columns:
        dd[yerr] = to_num(dd[yerr])

    dd = dd.dropna(subset=[TEFF, ycol, gcol])
    if dd.empty:
        return False, f"[skip] {band} {grid}: no rows after cleaning"

    # Sort by Teff for stable appearance and consistent label order
    order = np.argsort(dd[TEFF].values)
    dd = dd.iloc[order]

    # Colors from Teff with shared normalization
    colors = cmap(tnorm(dd[TEFF].values))

    # Standalone figure (no subplots)
    fig = plt.figure(figsize=PANEL_SIZE)
    ax  = plt.gca()

    if mode == "abs":
        # Model grid as black dots + dashed binned trend
        ax.scatter(dd[TEFF], dd[gcol], s=SCATTER_S_MODEL, c="k", alpha=0.75,
                   label=f"{grid.capitalize()} model")
        xm, ym = _binned_trend(dd[TEFF].values, dd[gcol].values, nbins=10)
        if xm is not None:
            ax.plot(xm, ym, linestyle='--', color='k', linewidth=1.0, alpha=0.9)

        # CHARA with faint error bars
        if yerr in dd.columns and dd[yerr].notna().any():
            dd_e = dd.dropna(subset=[yerr])
            if not dd_e.empty:
                ax.errorbar(dd_e[TEFF], dd_e[ycol], yerr=dd_e[yerr], fmt="none",
                            elinewidth=1, capsize=2, alpha=0.35, color='gray')
        ax.scatter(dd[TEFF], dd[ycol], s=SCATTER_S_DATA, marker="x", c=colors, label="Fit CHARA $V^2$")
        ax.set_ylabel("Power-1 LDC (alpha)")
    else:
        # Residuals: Δalpha = alpha(CHARA) - alpha(model)
        resid = dd[ycol] - dd[gcol]
        # residual error ~ data error
        if yerr in dd.columns and dd[yerr].notna().any():
            dd_e = dd.dropna(subset=[yerr])
            if not dd_e.empty:
                ax.errorbar(dd_e[TEFF], (dd_e[ycol]-dd_e[gcol]), yerr=dd_e[yerr], fmt="none",
                            elinewidth=1, capsize=2, alpha=0.35, color='gray')
        ax.scatter(dd[TEFF], resid, s=SCATTER_S_DATA, marker='x', c=colors, label='Δα (CHARA−model)')
        ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.7)
        ax.set_ylabel("Δα (CHARA−model)")

    # No per‑point name labels (keeps figure readable for NSF)

    #ax.set_ylim([0.0, 0.35])
    ax.set_xlabel("Teff (K)")
    #if band is "MYSTIC":
    #ax.set_title(f"K-band — stellar grid: {grid} vs Obs. Measured")
    #else:
     #   ax.set_title(f"H-band — stellar grid: {grid} vs Obs. Measured")

    #if xlim[0] is not None and xlim[1] is not None and xlim[1] > xlim[0]:
    #    ax.set_xlim(xlim)

    # Explicit legend (marker semantics)
    from matplotlib.lines import Line2D
    if mode == "abs":
        legend_handles = [
            Line2D([0], [0], marker='o', color='k', linestyle='None', markersize=6,
                   label=f'{grid.capitalize()} model (•) + trend (--)'),
            Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=6,
                   label='CHARA (×; color = Teff)'),
        ]
    else:
        legend_handles = [
            Line2D([0], [0], linestyle='-', color='k', linewidth=0.8, label='Δα = 0'),
            Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=6,
                   label='Δα (CHARA−model); color = Teff'),
        ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False)
    # Optional horizontal colorbar per panel (Teff)
    
    if add_cbar:
        sm = mpl.cm.ScalarMappable(norm=tnorm, cmap=cmap)
        sm.set_array([])
        try:
            cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.08, pad=0.12)
            cb.set_label("Teff (K)")
        except Exception as _e:
            try:
                cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.85, pad=0.15)
                cb.set_label("Teff (K)")
            except Exception:
                pass
    

    ax.grid(True, linestyle=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_png, dpi=PANEL_DPI, bbox_inches="tight")
    plt.close(fig)
    return True, f"[saved] {out_png}"

# --- New helpers to draw into provided axes (for custom mosaics) ---
def _get_dd(df, band, grid):
    ycol = CHARA_Y[band]["y"]; yerr = pick_yerr_col(df.columns, band)
    gcol = pick_grid_col(df.columns, grid, band)
    if gcol is None or ycol not in df.columns:
        return None, None, None, None
    cols = [TEFF, NAME, ycol, gcol] + ([yerr] if yerr else [])
    dd = df[cols].copy()
    dd[TEFF] = to_num(dd[TEFF]); dd[ycol] = to_num(dd[ycol]); dd[gcol] = to_num(dd[gcol])
    if yerr in dd.columns:
        dd[yerr] = to_num(dd[yerr])
    dd = dd.dropna(subset=[TEFF, ycol, gcol])
    if dd.empty:
        return None, None, None, None
    dd = dd.iloc[np.argsort(dd[TEFF].values)]
    return dd, ycol, yerr, gcol

def plot_abs_ax(ax, dd, ycol, yerr, gcol, grid, tnorm, cmap, show_ylabel=True):
    colors = cmap(tnorm(dd[TEFF].values))
    # model points + dashed binned trend
    ax.scatter(dd[TEFF], dd[gcol], s=SCATTER_S_MODEL, c="k", alpha=0.75, label=f"{grid.capitalize()} model")
    xm, ym = _binned_trend(dd[TEFF].values, dd[gcol].values, nbins=10)
    if xm is not None:
        ax.plot(xm, ym, linestyle='--', color='k', linewidth=1.0, alpha=0.9)
    # CHARA
    if yerr in dd.columns and dd[yerr].notna().any():
        dd_e = dd.dropna(subset=[yerr])
        if not dd_e.empty:
            ax.errorbar(dd_e[TEFF], dd_e[ycol], yerr=dd_e[yerr], fmt="none",
                        elinewidth=1, capsize=2, alpha=0.35, color='gray')
    ax.scatter(dd[TEFF], dd[ycol], s=SCATTER_S_DATA, marker='x', c=colors, label='CHARA')
    #ax.set_ylim(0.0, 0.35)
    if show_ylabel:
        ax.set_ylabel("Power-1 LDC (alpha)")
    ax.grid(True, linestyle=":", alpha=0.35)

def plot_resid_ax(ax, dd, ycol, yerr, gcol, tnorm, cmap, show_ylabel=True):
    colors = cmap(tnorm(dd[TEFF].values))
    resid = dd[ycol] - dd[gcol]
    if yerr in dd.columns and dd[yerr].notna().any():
        dd_e = dd.dropna(subset=[yerr])
        if not dd_e.empty:
            ax.errorbar(dd_e[TEFF], (dd_e[ycol]-dd_e[gcol]), yerr=dd_e[yerr], fmt="none",
                        elinewidth=1, capsize=2, alpha=0.35, color='gray')
    ax.scatter(dd[TEFF], resid, s=SCATTER_S_DATA, marker='x', c=colors, label='Δα (CHARA−model)')
    ax.axhline(0.0, color='k', linewidth=0.8, alpha=0.7)
    ax.set_ylim(-0.10, 0.10)  # requested ±0.1 window
    if show_ylabel:
        ax.set_ylabel("Δα (CHARA−model)")
    ax.grid(True, linestyle=":", alpha=0.35)
        

def make_colorbar(tmin, tmax, cmap, out_png, label="Teff (K)"):
    """Create a standalone horizontal colorbar image spanning the global Teff range."""
    fig = plt.figure(figsize=(10, 0.5))
    ax  = fig.add_axes([0.04, 0.35, 0.92, 0.4])
    norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    sm   = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=ax, orientation="horizontal")
    cb.set_label(label)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_png

def stitch_2x2_with_colorbar(panel_paths, colorbar_png, title_text, out_png):
    """Stitch four panel PNGs into a 2x2 mosaic and append a horizontal colorbar."""
    try:
        from PIL import Image, ImageDraw
    except Exception as e:
        return False, f"[skip mosaic] Pillow not available: {e}"

    imgs = []
    for p in panel_paths:
        if not os.path.exists(p):
            return False, f"[skip mosaic] Missing panel: {p}"
        imgs.append(Image.open(p).convert("RGB"))

    # Normalize size
    w = max(im.size[0] for im in imgs)
    h = max(im.size[1] for im in imgs)
    pads = []
    for im in imgs:
        canvas = Image.new("RGB", (w, h), (255,255,255))
        canvas.paste(im, ((w-im.size[0])//2, (h-im.size[1])//2))
        pads.append(canvas)

    grid = Image.new("RGB", (w*2, h*2), (255,255,255))
    grid.paste(pads[0], (0,   0))
    grid.paste(pads[1], (w,   0))
    grid.paste(pads[2], (0,   h))
    grid.paste(pads[3], (w,   h))

    # Title bar
    title_bar = Image.new("RGB", (w*2, 50), (255,255,255))
    draw = ImageDraw.Draw(title_bar)
    #draw.text((10, 10), title_text, fill=(0,0,0))

    # Colorbar
    if os.path.exists(colorbar_png):
        cbar = Image.open(colorbar_png).convert("RGB")
        cbar = cbar.resize((w*2, cbar.size[1]))
        total_h = title_bar.size[1] + grid.size[1] + cbar.size[1]
        out = Image.new("RGB", (w*2, total_h), (255,255,255))
        y = 0
        out.paste(title_bar, (0, y)); y += title_bar.size[1]
        out.paste(grid,      (0, y)); y += grid.size[1]
        out.paste(cbar,      (0, y))
    else:
        total_h = title_bar.size[1] + grid.size[1]
        out = Image.new("RGB", (w*2, total_h), (255,255,255))
        y = 0
        out.paste(title_bar, (0, y)); y += title_bar.size[1]
        out.paste(grid,      (0, y))

    out.save(out_png, format="PNG")
    return True, f"[saved mosaic] {out_png}"

def run_combined(df):
    """Produce the 2×2 mosaic: (H,K) × (Kurucz, STAGGER) with a shared colorbar."""
    os.makedirs(OUT_DIR, exist_ok=True)

    # Global Teff normalization across both bands
    tmin, tmax = band_teff_range(df)
    if tmin is None or tmax is None:
        print("[error] No Teff values;")
        return
    tnorm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    cmap  = mpl.cm.get_cmap(CMAP_NAME)

    # Generate the four panels in a fixed order:
    # Top row   (absolute): [0] MIRCX Kurucz, [1] MIRCX STAGGER
    # Bottom row (residual): [2] MYSTIC Kurucz, [3] MYSTIC STAGGER
    combos = [
        ("MIRCX",  "kurucz"),
        ("MIRCX",  "stagger"),
        ("MYSTIC", "kurucz"),
        ("MYSTIC", "stagger"),
    ]
    panel_paths = []
    for i,(band, grid) in enumerate(combos):
        pan = os.path.join(OUT_DIR, f"panel_{band}_{grid}.png")
        mode = "abs" if i < 2 else "resid"
        ok, msg = plot_panel(df, band, grid, (tmin, tmax), tnorm, cmap, pan, add_cbar=False, mode=mode)
        print(msg)
        if not ok:
            print(f"[warn] Missing panel for {band}/{grid}; aborting mosaic.")
            return
        panel_paths.append(pan)

    # Shared colorbar
    cbar = os.path.join(OUT_DIR, "colorbar_teff.png")
    make_colorbar(tmin, tmax, cmap, cbar, label="Teff (K)")

    # Stitch
    title = "Power-1 LDC (alpha) vs Teff — H/K × (Kurucz, STAGGER)"
    out_png = os.path.join(OUT_DIR, "ldc_power1_teff_kurucz_stagger_2x2.png")
    ok, msg = stitch_2x2_with_colorbar(panel_paths, cbar, title, out_png)
    print(msg)

def _plot_grid_reference(ax, dd, gcol, grid):
    style = GRID_STYLES.get(grid, {"label": grid.capitalize(), "color": "0.5", "marker": "o", "linestyle": "-"})
    ax.scatter(
        dd[TEFF], dd[gcol], s=SCATTER_S_MODEL, marker=style["marker"],
        facecolors="none", edgecolors=style["color"], linewidths=0.8,
        alpha=0.75, zorder=style.get("zorder", 2), label=style["label"],
    )
    xm, ym = _binned_trend(dd[TEFF].values, dd[gcol].values, nbins=8)
    if xm is not None:
        ax.plot(
            xm, ym, color=style["color"], linestyle=style["linestyle"],
            linewidth=1.8 if grid == "stagger" else 1.4,
            alpha=0.95, zorder=style.get("zorder", 2),
        )


def _plot_chara_alpha(ax, dd, ycol, yerr, norm, cmap, label_targets=False):
    colors = cmap(norm(dd[TEFF].values))
    if yerr in dd.columns and dd[yerr].notna().any():
        dd_e = dd.dropna(subset=[yerr])
        if not dd_e.empty:
            ax.errorbar(
                dd_e[TEFF], dd_e[ycol], yerr=dd_e[yerr], fmt="none",
                elinewidth=1.0, capsize=2, alpha=0.9,
                ecolor=(0.4, 0.6, 0.8, 0.35), zorder=6,
            )
    ax.scatter(
        dd[TEFF], dd[ycol], s=SCATTER_S_DATA, marker="o", c=colors,
        edgecolors="k", linewidths=0.45, zorder=7, label="Fit CHARA $V^2$",
    )
    if label_targets and NAME in dd.columns:
        annotate_points(ax, dd[TEFF].values, dd[ycol].values, dd[NAME].values)


def _plot_mps2_residual(ax, df, band, norm, cmap, show_ylabel=True):
    dd, ycol, yerr, gcol = _get_dd(df, band, "stagger")
    if dd is None:
        ax.set_visible(False)
        return
    resid = dd[ycol] - dd[gcol]
    colors = cmap(norm(dd[TEFF].values))
    if yerr in dd.columns and dd[yerr].notna().any():
        dd_e = dd.dropna(subset=[yerr])
        if not dd_e.empty:
            ax.errorbar(
                dd_e[TEFF], dd_e[ycol] - dd_e[gcol], yerr=dd_e[yerr],
                fmt="none", elinewidth=1.0, capsize=2, alpha=0.9,
                ecolor=(0.4, 0.6, 0.8, 0.35), zorder=2,
            )
    ax.scatter(
        dd[TEFF], resid, s=SCATTER_S_DATA, marker="o", c=colors,
        edgecolors="k", linewidths=0.45, zorder=3,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.85, zorder=1)
    xm, ym = _binned_trend(dd[TEFF].values, resid.values, nbins=8)
    #if xm is not None:
    #    ax.plot(xm, ym, color="black", linewidth=1.8, alpha=0.9, zorder=4)

    med = np.nanmedian(resid) if np.isfinite(resid).any() else np.nan
    rms = np.sqrt(np.nanmean(resid * resid)) if np.isfinite(resid).any() else np.nan
    ax.text(
        0.03, 0.94, f"median = {med:+.3f}\nrms = {rms:.3f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
    )
    alpha_label = r"$\Delta\alpha_H$" if band == "MIRCX" else r"$\Delta\alpha_K$"
    ax.set_title(f"{CHARA_Y[band]['title']} residuals: CHARA - Stagger")
    ax.set_xlabel("Teff (K)")
    if show_ylabel:
        ax.set_ylabel(fr"{alpha_label} (CHARA $V^2$ - MPS2 $I(\mu)$)")
    ax.set_ylim(-0.20, 0.2)
    ax.grid(True, linestyle=":", alpha=0.35)


def run_composite_HK(df, label_targets=False):
    """Four-panel Figure 5.

    Top: H and K CHARA alpha vs Teff, overplotted with Stagger, Kurucz,
    MPS1, and MPS2 model-grid loci plus binned trends.
    Bottom: CHARA - Stagger residuals for H and K.
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    # Global Teff normalization for color
    tmin, tmax = band_teff_range(df)
    if tmin is None or tmax is None:
        print('[error] No Teff values for composite')
        return
    norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    cmap = get_cmap(CMAP_NAME)

    fig, axes = plt.subplots(
        2, 2, figsize=(11.5, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
        constrained_layout=True,
    )

    # consistent x,y limits
    xvals = to_num(df.get(TEFF)).dropna()
    if not xvals.empty:
        xmin, xmax = np.nanpercentile(xvals, 2), np.nanpercentile(xvals, 98)
    else:
        xmin, xmax = None, None

    for ax, band in zip(axes[0, :], BANDS):
        chara_dd = None
        chara_ycol = None
        chara_yerr = None
        for grid in GRIDS:
            dd, ycol, yerr, gcol = _get_dd(df, band, grid)
            if dd is None:
                print(f"[warn] Missing grid column for {grid} {band}")
                continue
            _plot_grid_reference(ax, dd, gcol, grid)
            if chara_dd is None:
                chara_dd, chara_ycol, chara_yerr = dd, ycol, yerr
        if chara_dd is not None:
            _plot_chara_alpha(ax, chara_dd, chara_ycol, chara_yerr, norm, cmap, label_targets=label_targets)
        #if xmin is not None and xmax is not None and xmax > xmin:
        ax.set_xlim(xmin-500, xmax+500)
        ax.set_ylim(0.05, 0.37)
        ax.set_title(f"{CHARA_Y[band]['title']}: CHARA and model grids")
        ax.set_ylabel(r"$\alpha_H$ (MIRC-X)" if band == "MIRCX" else r"$\alpha_K$ (MYSTIC)")
        ax.grid(True, linestyle=":", alpha=0.35)

    _plot_mps2_residual(axes[1, 0], df, "MIRCX", norm, cmap, show_ylabel=True)
    _plot_mps2_residual(axes[1, 1], df, "MYSTIC", norm, cmap, show_ylabel=True)

    # Compact legend: CHARA plus the four grid loci.
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="#2f80ed", linestyle="None",
               markerfacecolor="#2ad1b3", markeredgecolor="k",
               markersize=9, markeredgewidth=0.8, label="Fit CHARA $V^2$"),
    ]
    for grid in GRIDS:
        style = GRID_STYLES[grid]
        legend_handles.append(
            Line2D([0], [0], marker=style["marker"], linestyle=style["linestyle"],
                   color=style["color"], markerfacecolor="none", markeredgecolor=style["color"],
                   linewidth=1.8 if grid == "stagger" else 1.4,
                   markersize=6, label=style["label"])
        )
    axes[0, 1].legend(handles=legend_handles, loc="upper right", frameon=False, ncol=1)

    # single vertical colorbar at right
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', fraction=0.046, pad=0.02)
    cbar.set_label('Teff (K)')

    out_name = 'composite_HK_all_grids_stagger_residuals_labeled.png' if label_targets else 'composite_HK_all_grids_stagger_residuals.png'
    out_png = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[saved]', out_png)

    if not label_targets:
        # Keep the historical filename fresh for main.tex or older notes.
        legacy_png = os.path.join(OUT_DIR, 'composite_HK_kurucz_stagger_abs.png')
        try:
            import shutil
            shutil.copyfile(out_png, legacy_png)
            print('[saved]', legacy_png)
        except Exception as e:
            print('[warn] Could not write legacy composite filename:', e)

def run_band_4panel(df, band):
    """For a single band, make a 2×2 mosaic where top row shows absolute (Kurucz, STAGGER)
    and bottom row shows residuals with a smaller vertical extent.
    """
    import matplotlib.gridspec as gridspec
    os.makedirs(OUT_DIR, exist_ok=True)

    # Teff normalization
    tmin, tmax = band_teff_range(df)
    if tmin is None or tmax is None:
        print(f"[error] No Teff values for {band}")
        return
    tnorm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    cmap  = get_cmap(CMAP_NAME)

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2.2, 1.2], hspace=0.15, wspace=0.12)

    panels = [
        (0, 0, 'kurucz', 'abs'), (0, 1, 'stagger', 'abs'),
        (1, 0, 'kurucz', 'resid'), (1, 1, 'stagger', 'resid'),
    ]
    for r, c, grid, mode in panels:
        ax = fig.add_subplot(gs[r, c])
        dd, ycol, yerr, gcol = _get_dd(df, band, grid)
        if dd is None:
            ax.set_visible(False)
            continue
        if mode == 'abs':
            plot_abs_ax(ax, dd, ycol, yerr, gcol, grid, tnorm, cmap, show_ylabel=(c==0))
        else:
            plot_resid_ax(ax, dd, ycol, yerr, gcol, tnorm, cmap, show_ylabel=(c==0))
        #if r == 1:
            #ax.set_xlabel("Teff (K)")
        ax.set_title(f"{CHARA_Y[band]['title']} — {grid.capitalize()}")

    # Shared colorbar
    sm = mpl.cm.ScalarMappable(norm=tnorm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.12, 0.04, 0.76, 0.02])
    cb  = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label('Teff (K)')

    out_png = os.path.join(OUT_DIR, f"{band}_4panel_abs_resid.png")
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[saved]', out_png)

def run_models_only(df):
    """Figure 3: Kurucz vs STAGGER — models only (both bands)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    tmin, tmax = band_teff_range(df)
    if tmin is None or tmax is None:
        print('[error] No Teff values for models-only')
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, grid in zip(axes, ['kurucz','stagger']):
        for band, marker in [('MIRCX','o'), ('MYSTIC','^')]:
            dd, ycol, yerr, gcol = _get_dd(df, band, grid)
            if dd is None: continue
            ax.scatter(dd[TEFF], dd[gcol], s=SCATTER_S_MODEL, c='k', alpha=0.7,
                       marker=marker, label=f'{grid.capitalize()} {band}')
        ax.set_xlabel('Teff (K)')
        #ax.set_ylim(0.0, 0.35)
        ax.set_title(f'{grid.capitalize()} — models only')
        ax.grid(True, linestyle=':', alpha=0.35)
    axes[0].set_ylabel('Power-1 LDC (alpha)')
    axes[1].legend(loc='upper right', frameon=False)
    out_png = os.path.join(OUT_DIR, 'models_kurucz_vs_stagger.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[saved]', out_png)

def run_chara_only(df):
    """Figure 4: H vs K — CHARA only."""
    os.makedirs(OUT_DIR, exist_ok=True)
    tmin, tmax = band_teff_range(df)
    if tmin is None or tmax is None:
        print('[error] No Teff values for chara-only')
        return
    norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    cmap = get_cmap(CMAP_NAME)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
    for ax, band in zip(axes, ['MIRCX','MYSTIC']):
        dd, ycol, yerr, gcol = _get_dd(df, band, 'kurucz')  # gcol unused
        if dd is None: continue
        colors = cmap(norm(dd[TEFF].values))
        if yerr in dd.columns and dd[yerr].notna().any():
            dd_e = dd.dropna(subset=[yerr])
            if not dd_e.empty:
                ax.errorbar(dd_e[TEFF], dd_e[ycol], yerr=dd_e[yerr], fmt='none',
                            elinewidth=1, capsize=2, alpha=0.35, color='gray')
        ax.scatter(dd[TEFF], dd[ycol], s=SCATTER_S_DATA, marker='x', c=colors)
        ax.set_title(CHARA_Y[band]['title'])
        ax.set_xlabel('Teff (K)')
        ax.set_ylim(0.0, 0.35)
        ax.grid(True, linestyle=':', alpha=0.35)
    axes[0].set_ylabel('Power-1 LDC (alpha)')
    # shared colorbar
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cax = fig.add_axes([0.12, 0.08, 0.76, 0.03])
    cb  = fig.colorbar(sm, cax=cax, orientation='horizontal'); cb.set_label('Teff (K)')
    out_png = os.path.join(OUT_DIR, 'chara_H_vs_K.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[saved]', out_png)

def run_residual_histograms(df):
    """Mini 2‑panel summary: residual histograms without Teff axes.
    Left: Kurucz (H+K combined); Right: STAGGER (H+K combined).
    Refinements for NSF panel readability:
      - uniform bins and x‑limits (±0.15)
      - density=True so y is Fraction of sample
      - light vertical guides at ±0.05 with small label
      - compact panel titles with (a)/(b) tags
      - semi‑transparent annotation boxes (RMS |Δα|, median Δα)
    """
    os.makedirs(OUT_DIR, exist_ok=True)

    def collect_resid(grid):
        res = []
        for band in ['MIRCX','MYSTIC']:
            dd, ycol, yerr, gcol = _get_dd(df, band, grid)
            if dd is None: 
                continue
            r = (dd[ycol] - dd[gcol]).values
            r = r[np.isfinite(r)]
            if r.size:
                res.append(r)
        if res:
            return np.concatenate(res)
        return np.array([])

    res_k = collect_resid('kurucz')
    res_s = collect_resid('stagger')

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True, constrained_layout=True)
    # uniform binning across panels
    bins = np.linspace(-0.15, 0.15, 20)

    for ax, res, title in [
        (axes[0], res_k, 'Kurucz'),
        (axes[1], res_s, 'STAGGER')]:
        if res.size:
            # histogram as fraction (density) with subtle grayscale contrast per panel
            color = '0.65' if title.lower().startswith('kurucz') else '0.35'
            edge  = '0.30' if title.lower().startswith('kurucz') else '0.20'
            ax.hist(res, bins=bins, color=color, edgecolor=edge, alpha=0.85, density=True)
            rms = np.sqrt(np.nanmean(res**2))
            med = np.nanmedian(res)
            sgn = '+' if med >= 0 else '−'
            med_abs = abs(med)
            ax.axvline(0.0, color='k', linewidth=1.0, alpha=0.7)
            # subtle shaded acceptable band ±0.05
            ax.axvspan(-0.05, 0.05, color='gray', alpha=0.07, zorder=0)
            # light guides at ±0.05
            ax.axvline(0.05, color='gray', alpha=0.3, linestyle='--')
            ax.axvline(-0.05, color='gray', alpha=0.3, linestyle='--')
            ax.set_xlim(-0.15, 0.15)
            ax.set_xlabel('Δα (CHARA−model)')
            ax.set_title(title)
            # annotation box with quantitative takeaway
            ax.text(0.97, 0.92, f'RMS |Δα| = {rms:.3f}\nmedian Δα = {sgn}{med_abs:.3f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.6, ec='none'))
            # label the ±0.05 guides once per panel (small text near top)
            ylim = ax.get_ylim()
            ylab = ylim[1]*0.95
            ax.text(0.052, ylab, ' +0.05', fontsize=8, color='gray')
            ax.text(-0.148, ylab, '−0.05 ', fontsize=8, color='gray', ha='left')
        else:
            ax.set_visible(False)

    axes[0].set_ylabel('Fraction of sample')
    # panel tags
    axes[0].text(0.02, 0.95, '(a)', transform=axes[0].transAxes, ha='left', va='top', fontsize=10, fontweight='bold')
    axes[1].text(0.02, 0.95, '(b)', transform=axes[1].transAxes, ha='left', va='top', fontsize=10, fontweight='bold')
    out_png = os.path.join(OUT_DIR, 'residual_histograms_kurucz_vs_stagger.png')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('[saved]', out_png)

def main():
    ap = argparse.ArgumentParser(description="Generate power-law LDC(alpha) plots and dispersion summary.")
    ap.add_argument("--csv", default=None, help="Input merged CSV (default: auto-detect).")
    ap.add_argument(
        "--label-targets", action="store_true",
        help="Add target-name annotations to the composite H/K figure for outlier identification.",
    )
    args = ap.parse_args()

    csv_path = resolve_csv_path(args.csv)
    if not csv_path:
        opts = ", ".join(CSV_DEFAULTS) if not args.csv else args.csv
        print(f"[error] Input CSV not found. Tried: {opts}")
        return
    df = pd.read_csv(csv_path)
    print("[info] Using CSV:", csv_path)

    # Sanity check
    req = [TEFF, NAME]
    for band in BANDS:
        req += [CHARA_Y[band]["y"]]
    missing = [c for c in req if c not in df.columns]
    if missing:
        print("[error] Missing required columns:", ", ".join(missing))
        return

    # Figure A: concise 2×2 (abs only; H top, K bottom; Kurucz|STAGGER columns) with annotations
    run_composite_HK(df, label_targets=args.label_targets)
    # Figure 1: H-band only, 4 panels (top=abs large, bottom=resid smaller)
    run_band_4panel(df, 'MIRCX')
    # Figure 2: K-band only, 4 panels (top=abs large, bottom=resid smaller)
    run_band_4panel(df, 'MYSTIC')
    # Figure 3: models only (Kurucz vs STAGGER)
    run_models_only(df)
    # Figure 4: CHARA only (H vs K)
    run_chara_only(df)
    # Mini: residual histograms (Kurucz vs STAGGER)
    run_residual_histograms(df)

    # Print a copy-pasteable quantitative dispersion summary for the paper
    os.makedirs(OUT_DIR, exist_ok=True)
    summary_txt = summarize_dispersion(df, grids=GRIDS, bands=BANDS)
    print(summary_txt)
    summary_path = os.path.join(OUT_DIR, "dispersion_summary_alpha.txt")
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_txt + "\n")
        print("[saved]", summary_path)
    except Exception as e:
        print("[warn] Could not write summary file:", e)

    # Print a copy-pasteable H–K contrast summary (expected vs observed)
    hk_txt = summarize_hk_contrast(df, grids=GRIDS)
    print(hk_txt)
    hk_path = os.path.join(OUT_DIR, "hk_contrast_summary_alpha.txt")
    try:
        with open(hk_path, "w", encoding="utf-8") as f:
            f.write(hk_txt + "\n")
        print("[saved]", hk_path)
    except Exception as e:
        print("[warn] Could not write H–K summary file:", e)

    print("[done] Output in:", OUT_DIR)

if __name__ == "__main__":
    main()
