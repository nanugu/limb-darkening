import os
import re
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pmoired
from scipy.signal import medfilt


@dataclass
class PlotContext:
    oi: object
    file_path: str
    target: str
    instrument: str
    date: str


def target_to_latex(target: str) -> str:
    t = str(target).strip().replace('_', ' ').replace('.', ' ')
    t = re.sub(r'\s+', ' ', t).strip()
    if not t:
        return t
    greek = {
        'alf': r'\alpha', 'bet': r'\beta', 'gam': r'\gamma', 'del': r'\delta',
        'eps': r'\epsilon', 'zet': r'\zeta', 'eta': r'\eta', 'the': r'\theta',
        'iot': r'\iota', 'kap': r'\kappa', 'lam': r'\lambda', 'mu': r'\mu',
        'nu': r'\nu', 'ksi': r'\xi', 'omi': r'\mathrm{o}', 'pi': r'\pi',
        'rho': r'\rho', 'sig': r'\sigma', 'tau': r'\tau', 'ups': r'\upsilon',
        'phi': r'\phi', 'chi': r'\chi', 'psi': r'\psi', 'ome': r'\omega',
    }
    parts = t.split()
    m = re.match(r'^([A-Za-z]+)(\d+)?$', parts[0])
    if m:
        key = m.group(1).lower()
        n = m.group(2)
        if key in greek:
            parts[0] = rf'${greek[key]}^{{{int(n)}}}$' if n else rf'${greek[key]}$'
    return ' '.join(parts)

def _synthetic_obs_block(bwl_grid, wl_um, mjd):
    bwl_grid = np.asarray(bwl_grid, dtype=float)
    wl_um = float(wl_um)
    mjd = float(mjd)
    u = bwl_grid * wl_um
    npt = len(bwl_grid)
    mjd_arr = np.full(npt, mjd, dtype=float)
    return {
        'u': u,
        'v': np.zeros_like(u),
        'u/wl': bwl_grid[:, None],
        'v/wl': np.zeros((npt, 1), dtype=float),
        'B/wl': bwl_grid[:, None],
        'PA': np.zeros((npt, 1), dtype=float),
        'MJD': mjd_arr,
        'MJD2': mjd_arr[:, None],
        'FLAG': np.zeros((npt, 1), dtype=bool),
    }

def build_synthetic_v2_template(bwl_grid, wl_um, mjd):
    base = _synthetic_obs_block(bwl_grid, wl_um, mjd)
    return {
        'MJD': np.array([float(mjd)]),
        'WL': np.array([float(wl_um)]),
        'fit': {'obs': ['V2']},
        'OI_VIS': {'grid': dict(base)},
        'OI_VIS2': {'grid': dict(base, EV2=np.zeros((len(bwl_grid), 1), dtype=float))},
    }

def evaluate_continuous_v2_curve(model, bwl_grid, wl_um, mjd):
    template = build_synthetic_v2_template(bwl_grid, wl_um, mjd)
    curve = pmoired.oimodels.VmodelOI(template, model)
    return (
        np.asarray(curve['OI_VIS2']['grid']['B/wl'], dtype=float).flatten(),
        np.asarray(curve['OI_VIS2']['grid']['V2'], dtype=float).flatten(),
    )

def clean_model_for_plot(model):
    if not isinstance(model, dict):
        return model
    return {k: v for k, v in model.items() if not str(k).startswith('#') and k != 'V2_0'}

def build_satlas_plot_model(model):
    if not isinstance(model, dict):
        return None
    clean = clean_model_for_plot(model)
    sat = {}
    for band, spectrum in (('H', '($WL<1.8)'), ('K', '($WL>1.8)')):
        band_keys = {k: v for k, v in clean.items() if str(k).startswith(f'{band},')}
        if band_keys:
            sat.update(band_keys)
            sat.setdefault(f'{band},spectrum', spectrum)
    if sat:
        if 'H,ROSSDIAM' in sat and 'K,ROSSDIAM' not in sat:
            sat['K,ROSSDIAM'] = sat['H,ROSSDIAM']
        if 'K,ROSSDIAM' in sat and 'H,ROSSDIAM' not in sat:
            sat['H,ROSSDIAM'] = sat['K,ROSSDIAM']
        return sat
    return clean if '*,profile' in clean else None


def overplot_grid_models(ctx, fit_bundle, band_view='H', savefig=0):
    oi = ctx.oi
    file_path = ctx.file_path
    target = ctx.target
    view = str(band_view).lower()
    if view not in ('h', 'k', 'both'):
        raise ValueError("band_view must be one of: 'H', 'K', 'both'")

    def _f(x, default=np.nan):
        try:
            return float(x)
        except Exception:
            return float(default)

    display_names = {
        'UD': 'UD fit',
        'SATLAS': 'SATLAS fit',
        'Linear Law': 'Linear LD fit',
        'Power Law': 'Power Law fit',
        'Power2 Law': 'Power-2 fit',
        'Quad Law': 'Quadratic LD fit',
        'Stagger Power Law': 'Stagger',
    }

    ud = fit_bundle['ud']
    ll = fit_bundle['ll']
    pl = fit_bundle['pl']
    pl2 = fit_bundle['pl2']
    ql = fit_bundle['ql']
    chi2 = fit_bundle['chi2']
    V2_0 = _f(pl.get('V2_0', ud.get('V2_0', 1.0)), 1.0)

    models = {
        'UD': {
            'H,ud': _f(ud.get('H,ud')),
            'H,spectrum': '($WL<1.8)',
            'K,ud': _f(ud.get('K,ud')),
            'K,spectrum': '($WL>1.8)',
        },
        'Linear Law': {
            'diam': _f(ll.get('diam')),
            'H,a': _f(ll.get('H,a')),
            'K,a': _f(ll.get('K,a')),
            'H,diam': '$diam',
            'H,profile': '1 - $H,a*(1 - $MU)',
            'H,spectrum': '($WL<1.8)',
            'K,diam': '$diam',
            'K,profile': '1 - $K,a*(1 - $MU)',
            'K,spectrum': '($WL>1.8)',
        },
        'Power2 Law': {
            'diam': _f(pl2.get('diam')),
            'H,a1': _f(pl2.get('H,a1')),
            'K,a1': _f(pl2.get('K,a1')),
            'H,alpha1': _f(pl2.get('H,alpha1')),
            'K,alpha1': _f(pl2.get('K,alpha1')),
            'H,diam': '$diam',
            'H,profile': '1-$H,a1*(1-($MU**$H,alpha1))',
            'H,spectrum': '($WL<1.8)',
            'K,diam': '$diam',
            'K,profile': '1-$K,a1*(1-($MU**$K,alpha1))',
            'K,spectrum': '($WL>1.8)',
        },
        'Quad Law': {
            'diam': _f(ql.get('diam')),
            'H,a': _f(ql.get('H,a')),
            'K,a': _f(ql.get('K,a')),
            'H,b': _f(ql.get('H,b')),
            'K,b': _f(ql.get('K,b')),
            'H,diam': '$diam',
            'H,profile': '1 - $H,a*(1 - $MU) - $H,b*(1 - $MU)**2',
            'H,spectrum': '($WL<1.8)',
            'K,diam': '$diam',
            'K,profile': '1 - $K,a*(1 - $MU) - $K,b*(1 - $MU)**2',
            'K,spectrum': '($WL>1.8)',
        },
        'Power Law': {
            'diam': _f(pl.get('diam')),
            'H,alpha': _f(pl.get('H,alpha')),
            'K,alpha': _f(pl.get('K,alpha')),
            'H,diam': '$diam',
            'H,profile': '$MU**$H,alpha',
            'H,spectrum': '($WL<1.8)',
            'K,diam': '$diam',
            'K,profile': '$MU**$K,alpha',
            'K,spectrum': '($WL>1.8)',
        },
    }

    model_styles = {
        'UD': {'color': '#1F77B4', 'marker': 'o', 'chi2': _f(chi2.get('ud'))},
        'Linear Law': {'color': '#009E73', 'marker': '^', 'chi2': _f(chi2.get('ll'))},
        'Power2 Law': {'color': '#CC79A7', 'marker': 'D', 'chi2': _f(chi2.get('pl2'))},
        'Quad Law': {'color': '#F0E442', 'marker': 'v', 'chi2': _f(chi2.get('ql'))},
        'Power Law': {'color': '#D55E00', 'marker': 's', 'chi2': _f(chi2.get('pl'))},
    }

    plot_models = {k: clean_model_for_plot(models[k]) for k in models}
    Vmodels = {k: pmoired.oimodels.VmodelOI(oi._merged, plot_models[k]) for k in plot_models}

    def _wl_flat(merged_i, w_mask):
        vis2_all = merged_i['OI_VIS2']['all']
        if 'WL' in vis2_all:
            return np.asarray(vis2_all['WL'][w_mask, :]).flatten()
        wl_axis = merged_i.get('WL', None)
        if wl_axis is None:
            n = vis2_all['B/wl'][w_mask, :].size
            return np.full(n, np.nan)
        wl_axis = np.asarray(wl_axis).flatten()
        nbase = int(np.sum(w_mask))
        return np.tile(wl_axis, nbase) if nbase > 0 else np.array([])

    def _collect_band_segments():
        samples = {
            'h': {'bl': [], 'wl': []},
            'k': {'bl': [], 'wl': []},
        }
        for i in range(len(oi._merged)):
            names = set(oi._merged[i]['OI_VIS2']['all']['NAME'])
            for k in names:
                w = oi._merged[i]['OI_VIS2']['all']['NAME'] == k
                f = ~oi._merged[i]['OI_VIS2']['all']['FLAG'][w, :].flatten()
                bl = oi._merged[i]['OI_VIS2']['all']['B/wl'][w, :].flatten()[f]
                wl = _wl_flat(oi._merged[i], w)[f]
                finite = np.isfinite(bl) & np.isfinite(wl)
                if not np.any(finite):
                    continue
                bl = bl[finite]
                wl = wl[finite]
                hmask = wl < 1.8
                kmask = wl > 1.8
                if np.any(hmask):
                    samples['h']['bl'].append(bl[hmask])
                    samples['h']['wl'].append(wl[hmask])
                if np.any(kmask):
                    samples['k']['bl'].append(bl[kmask])
                    samples['k']['wl'].append(wl[kmask])

        segments = []

        def _append_segment(key, linestyle):
            if not samples[key]['bl']:
                return
            bl = np.concatenate(samples[key]['bl'])
            wl = np.concatenate(samples[key]['wl'])
            segments.append({
                'wl_ref': float(np.nanmedian(wl)),
                'bwl_max': float(np.nanmax(bl)),
                'linestyle': linestyle,
            })

        if view in ('h', 'both'):
            _append_segment('h', '-')
        if view in ('k', 'both'):
            _append_segment('k', '-' if view != 'both' else '--')
        return segments

    plt.close('all')
    fig, ((axV2ln, axV2log), (axV2lnfiterr, axV2logfiterr)) = plt.subplots(
        2, 2, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True
    )

    mjd_ref = float(np.nanmean([
        np.nanmean(np.asarray(merged_i['MJD'], dtype=float))
        for merged_i in oi._merged
        if 'MJD' in merged_i
    ]))
    band_segments = _collect_band_segments()

    #look zorder is Matplotlib’s drawing priority.
    #ax.plot(..., zorder=1)       # drawn behind
    #ax.errorbar(..., zorder=5)   # drawn on top

    for j, m in enumerate(plot_models):
        style = model_styles[m]
        for seg in band_segments:
            bwl_grid = np.linspace(0.0, seg['bwl_max'], 600)
            c_bl, c_v2 = evaluate_continuous_v2_curve(plot_models[m], bwl_grid, seg['wl_ref'], mjd_ref)
            c_v2 = np.asarray(c_v2, dtype=float)
            c_v2[c_v2 <= 0] = np.nan
            axV2ln.plot(
                c_bl, c_v2, linestyle=seg['linestyle'], linewidth=2.4,
                color=style['color'], alpha=0.98, zorder=1,
            )
            axV2log.plot(
                c_bl, c_v2, linestyle=seg['linestyle'], linewidth=2.4,
                color=style['color'], alpha=0.98, zorder=1,
            )

        label_str = f"{display_names.get(m, m)} ($\\chi^2$ = {style['chi2']:.2f})"
        axV2log.text(
            0.98, 0.95 - 0.05 * j, label_str, transform=axV2log.transAxes,
            fontsize=10, ha='right', va='top', color=style['color'], zorder=6,
        )

    for i in range(len(oi._merged)):
        names = set(oi._merged[i]['OI_VIS2']['all']['NAME'])
        for k in names:
            w = oi._merged[i]['OI_VIS2']['all']['NAME'] == k
            f = ~oi._merged[i]['OI_VIS2']['all']['FLAG'][w, :].flatten()

            bl = oi._merged[i]['OI_VIS2']['all']['B/wl'][w, :].flatten()[f]
            v2 = oi._merged[i]['OI_VIS2']['all']['V2'][w, :].flatten()[f] / max(V2_0, 1e-8)
            v2_err = oi._merged[i]['OI_VIS2']['all']['EV2'][w, :].flatten()[f]
            wl = _wl_flat(oi._merged[i], w)[f]

            if view == 'h' and np.isfinite(wl).any():
                band_sel = wl < 1.8
            elif view == 'k' and np.isfinite(wl).any():
                band_sel = wl > 1.8
            else:
                band_sel = np.ones_like(bl, dtype=bool)

            if not np.any(band_sel):
                continue

            bl = bl[band_sel]
            v2 = v2[band_sel]
            v2_err = v2_err[band_sel]

            axV2ln.errorbar(
                bl, v2, yerr=v2_err, fmt='.', color='k', capsize=1,
                alpha=0.35, markersize=2.5, elinewidth=0.8, zorder=5,
            )
            axV2log.errorbar(
                bl, v2, yerr=v2_err, fmt='.', color='k', capsize=1,
                alpha=0.35, markersize=2.5, elinewidth=0.8, zorder=5,
            )

            for j, m in enumerate(Vmodels):
                style = model_styles[m]
                m_bl = Vmodels[m][i]['OI_VIS2']['all']['B/wl'][w, :].flatten()[f][band_sel]
                m_v2 = Vmodels[m][i]['OI_VIS2']['all']['V2'][w, :].flatten()[f][band_sel].astype(float)
                m_v2[m_v2 == 0] = np.nan

                v2_tmp_smooth = medfilt(v2_err, kernel_size=5) if len(v2_err) >= 5 else v2_err
                resi = (m_v2 - v2) / np.clip(v2_tmp_smooth, 1e-6, None)
                res_smooth = medfilt(resi, kernel_size=3) if len(resi) >= 3 else resi
                res_plot = np.clip(res_smooth, -8, 8)

                axV2lnfiterr.scatter(
                    bl, res_plot, marker=style['marker'], s=8, color=style['color'], alpha=0.35, linewidths=0,
                )
                axV2logfiterr.scatter(
                    bl, res_plot, marker=style['marker'], s=8, color=style['color'], alpha=0.35, linewidths=0,
                )
    
    

    axV2ln.set_ylabel(r'$V^2$')
    axV2ln.set_ylim([1e-4, 1.1])
    axV2ln.set_title("Squared Visibility (Linear Scale)")

    axV2log.set_yscale('log')
    axV2log.set_ylabel(r'$V^2$')
    axV2log.set_ylim([1e-4, 1.1])
    axV2log.set_title("Squared Visibility (Log Scale)")

    for ax in [axV2lnfiterr, axV2logfiterr]:
        ax.axhline(0, linestyle='--', color='gray', linewidth=0.6)
        ax.set_ylim(-11, 11)
        ax.grid(True, which='major', axis='y', linestyle=':', alpha=0.4)
        ax.set_xlabel(r'$B_\mathrm{max}/\lambda$ (M$\lambda$)')
        ax.set_ylabel(r'Residuals ($\sigma$)')

    band_label = {'h': 'H-band', 'k': 'K-band', 'both': 'H+K'}[view]
    plt.suptitle(f"{target_to_latex(target)} ({band_label})")
    plt.tight_layout()
    out_name = file_path.replace('/', '_') + f'_data_ld_laws_fit_{view}.png'
    if savefig:
        plt.savefig(out_name, bbox_inches='tight', dpi=300)
    plt.close('all')
    print(f"Done: {out_name} [{band_label}]")

def overplot_analytical_laws(ctx, fit_bundle, band_view='H', savefig=0):
    oi = ctx.oi
    file_path = ctx.file_path
    target = ctx.target
    view = str(band_view).lower()
    if view not in ('h', 'k', 'both'):
        raise ValueError("band_view must be one of: 'H', 'K', 'both'")

    def _f(x, default=np.nan):
        try:
            return float(x)
        except Exception:
            return float(default)

    display_names = {
        'Power Law': 'Power Law fit',
        'SATLAS': 'SATLAS fit',
        'Stagger Power Law': 'Stagger fit',
    }

    pl = fit_bundle['pl']
    sal = fit_bundle.get('sal')
    pl_stagger = fit_bundle.get('pl_stagger')
    chi2 = fit_bundle['chi2']
    V2_0 = _f(pl.get('V2_0', sal.get('V2_0', 1.0) if isinstance(sal, dict) else 1.0), 1.0)

    models = {
        'Power Law': {
            'diam': _f(pl.get('diam')),
            'H,alpha': _f(pl.get('H,alpha')),
            'K,alpha': _f(pl.get('K,alpha')),
            'H,diam': '$diam',
            'H,profile': '$MU**$H,alpha',
            'H,spectrum': '($WL<1.8)',
            'K,diam': '$diam',
            'K,profile': '$MU**$K,alpha',
            'K,spectrum': '($WL>1.8)',
        },
    }
    if isinstance(sal, dict):
        satlas_model = build_satlas_plot_model(sal)
        if satlas_model is not None:
            models['SATLAS'] = satlas_model
    if isinstance(pl_stagger, dict):
        models['Stagger Power Law'] = {
            'diam': _f(pl_stagger.get('diam')),
            'H,alpha': _f(pl_stagger.get('H,alpha')),
            'K,alpha': _f(pl_stagger.get('K,alpha')),
            'H,diam': '$diam',
            'H,profile': '$MU**$H,alpha',
            'H,spectrum': '($WL<1.8)',
            'K,diam': '$diam',
            'K,profile': '$MU**$K,alpha',
            'K,spectrum': '($WL>1.8)',
        }

    model_styles = {
        'Power Law': {'color': '#D55E00', 'chi2': _f(chi2.get('pl')), 'linestyle': '-'},
        'SATLAS': {'color': '#0072B2', 'chi2': _f(chi2.get('sal')), 'linestyle': '--'},
        'Stagger Power Law': {'color': '#7A3E9D', 'chi2': _f(chi2.get('pl_stagger')), 'linestyle': '-.'},
    }

    plot_models = {k: clean_model_for_plot(models[k]) for k in models}
    Vmodels = {k: pmoired.oimodels.VmodelOI(oi._merged, plot_models[k]) for k in plot_models}

    def _wl_flat(merged_i, w_mask):
        vis2_all = merged_i['OI_VIS2']['all']
        if 'WL' in vis2_all:
            return np.asarray(vis2_all['WL'][w_mask, :]).flatten()
        wl_axis = merged_i.get('WL', None)
        if wl_axis is None:
            n = vis2_all['B/wl'][w_mask, :].size
            return np.full(n, np.nan)
        wl_axis = np.asarray(wl_axis).flatten()
        nbase = int(np.sum(w_mask))
        return np.tile(wl_axis, nbase) if nbase > 0 else np.array([])

    def _collect_samples():
        bl_all, v2_all, v2err_all = [], [], []
        model_samples = {name: {'bl': [], 'v2': [], 'res': []} for name in plot_models}
        for i in range(len(oi._merged)):
            names = set(oi._merged[i]['OI_VIS2']['all']['NAME'])
            for k in names:
                w = oi._merged[i]['OI_VIS2']['all']['NAME'] == k
                f = ~oi._merged[i]['OI_VIS2']['all']['FLAG'][w, :].flatten()
                bl = oi._merged[i]['OI_VIS2']['all']['B/wl'][w, :].flatten()[f]
                v2 = oi._merged[i]['OI_VIS2']['all']['V2'][w, :].flatten()[f] / max(V2_0, 1e-8)
                v2_err = oi._merged[i]['OI_VIS2']['all']['EV2'][w, :].flatten()[f]
                wl = _wl_flat(oi._merged[i], w)[f]

                if view == 'h' and np.isfinite(wl).any():
                    band_sel = wl < 1.8
                elif view == 'k' and np.isfinite(wl).any():
                    band_sel = wl > 1.8
                else:
                    band_sel = np.ones_like(bl, dtype=bool)
                if not np.any(band_sel):
                    continue

                bl = bl[band_sel]
                v2 = v2[band_sel]
                v2_err = v2_err[band_sel]
                bl_all.append(bl)
                v2_all.append(v2)
                v2err_all.append(v2_err)

                for name, vm in Vmodels.items():
                    m_bl = vm[i]['OI_VIS2']['all']['B/wl'][w, :].flatten()[f][band_sel]
                    m_v2 = vm[i]['OI_VIS2']['all']['V2'][w, :].flatten()[f][band_sel].astype(float)
                    m_v2[m_v2 == 0] = np.nan
                    v2_tmp_smooth = medfilt(v2_err, kernel_size=5) if len(v2_err) >= 5 else v2_err
                    resi = (m_v2 - v2) / np.clip(v2_tmp_smooth, 1e-6, None)
                    res_smooth = medfilt(resi, kernel_size=3) if len(resi) >= 3 else resi
                    model_samples[name]['bl'].append(m_bl)
                    model_samples[name]['v2'].append(m_v2)
                    model_samples[name]['res'].append(np.clip(res_smooth, -8, 8))

        data = {
            'bl': np.concatenate(bl_all) if bl_all else np.array([]),
            'v2': np.concatenate(v2_all) if v2_all else np.array([]),
            'v2_err': np.concatenate(v2err_all) if v2err_all else np.array([]),
        }
        for name in model_samples:
            for key in model_samples[name]:
                model_samples[name][key] = (
                    np.concatenate(model_samples[name][key]) if model_samples[name][key] else np.array([])
                )
        return data, model_samples

    def _collect_band_segments():
        bl_all, wl_all = [], []
        for i in range(len(oi._merged)):
            names = set(oi._merged[i]['OI_VIS2']['all']['NAME'])
            for k in names:
                w = oi._merged[i]['OI_VIS2']['all']['NAME'] == k
                f = ~oi._merged[i]['OI_VIS2']['all']['FLAG'][w, :].flatten()
                bl = oi._merged[i]['OI_VIS2']['all']['B/wl'][w, :].flatten()[f]
                wl = _wl_flat(oi._merged[i], w)[f]
                finite = np.isfinite(bl) & np.isfinite(wl)
                if not np.any(finite):
                    continue
                bl = bl[finite]
                wl = wl[finite]
                if view == 'h':
                    sel = wl < 1.8
                elif view == 'k':
                    sel = wl > 1.8
                else:
                    sel = np.ones_like(wl, dtype=bool)
                if np.any(sel):
                    bl_all.append(bl[sel])
                    wl_all.append(wl[sel])
        if not bl_all:
            return []
        return [{
            'wl_ref': float(np.nanmedian(np.concatenate(wl_all))),
            'bwl_max': float(np.nanmax(np.concatenate(bl_all))),
            'linestyle': '-',
        }]

    def _find_first_null_x(x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) < 5:
            return None

        xv = x[valid]
        yv = np.abs(y[valid])
        if xv.size < 5:
            return None

        min_x = 0.05 * np.nanmax(xv)
        peaks = np.where((yv[1:-1] <= yv[:-2]) & (yv[1:-1] <= yv[2:]))[0] + 1
        peaks = [idx for idx in peaks if xv[idx] >= min_x]
        if peaks:
            threshold = max(1e-4, 0.02 * np.nanmax(yv))
            for idx in peaks:
                if yv[idx] <= threshold:
                    return float(xv[idx])
            return float(xv[peaks[0]])

        deep = np.where((xv >= min_x) & (yv <= max(1e-4, 0.02 * np.nanmax(yv))))[0]
        if deep.size:
            return float(xv[deep[0]])
        return None

    def _compute_post_null_limits(data, model_curves, reference_curve):
        if data['bl'].size == 0 or model_curves['x'].size == 0:
            return None

        xmax_data = float(np.nanmax(data['bl']))
        if not np.isfinite(xmax_data) or xmax_data <= 0:
            return None

        first_null_x = _find_first_null_x(model_curves['x'], reference_curve)
        if first_null_x is None:
            xmin = max(0.0, float(np.nanpercentile(data['bl'], 55)))
        else:
            xmin = max(0.0, 0.9 * first_null_x)
        xmax = 1.02 * xmax_data

        w = (data['bl'] >= xmin) & (data['bl'] <= xmax)
        if np.sum(w) < 3:
            return None

        yvals = [data['v2'][w]]
        x = model_curves['x']
        for name in model_curves:
            if name == 'x':
                continue
            mw = (x >= xmin) & (x <= xmax) & np.isfinite(model_curves[name]) & (model_curves[name] > 0)
            if np.any(mw):
                yvals.append(model_curves[name][mw])

        ycat = np.concatenate(yvals)
        ycat = ycat[np.isfinite(ycat) & (ycat > 0)]
        if ycat.size == 0:
            return None

        ymin = max(1e-4, 0.8 * np.nanmin(ycat))
        ymax = min(1.1, 1.2 * np.nanmax(ycat))
        return xmin, xmax, ymin, ymax, first_null_x

    data, model_samples = _collect_samples()
    band_segments = _collect_band_segments()
    mjd_ref = float(np.nanmean([
        np.nanmean(np.asarray(merged_i['MJD'], dtype=float))
        for merged_i in oi._merged
        if 'MJD' in merged_i
    ]))

    continuous = {'x': np.array([])}
    continuous_raw = {'x': np.array([])}
    if band_segments:
        seg = band_segments[0]
        bwl_grid = np.linspace(0.0, seg['bwl_max'], 900)
        continuous['x'] = bwl_grid
        continuous_raw['x'] = bwl_grid
        for name, model in plot_models.items():
            _, c_v2 = evaluate_continuous_v2_curve(model, bwl_grid, seg['wl_ref'], mjd_ref)
            c_v2 = np.asarray(c_v2, dtype=float)
            continuous_raw[name] = c_v2.copy()
            c_v2_plot = c_v2.copy()
            c_v2_plot[c_v2_plot <= 0] = np.nan
            continuous[name] = c_v2_plot

    zoom = _compute_post_null_limits(
        data,
        continuous,
        continuous_raw.get('Power Law', continuous.get('Power Law', np.array([])))
    )

    plt.close('all')
    fig, ((ax_full, ax_zoom), (ax_res_full, ax_res_zoom)) = plt.subplots(
        2, 2, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex='col'
    )

    for ax in (ax_full, ax_zoom):
        ax.set_yscale('log')
        for name, model in plot_models.items():
            style = model_styles[name]
            label = display_names.get(name, name)
            chi2_val = style.get('chi2', np.nan)
            if np.isfinite(chi2_val):
                label = f"{label} ($\\chi^2$ = {chi2_val:.2f})"
            ax.plot(
                continuous['x'], continuous[name], color=style['color'],
                linewidth=2.6, linestyle=style['linestyle'], alpha=0.98,
                label=label, zorder=1,
            )
        ax.errorbar(
            data['bl'], data['v2'], yerr=data['v2_err'],
            fmt='.', color='k', capsize=1, alpha=0.35,
            markersize=2.5, elinewidth=0.8, label='Data', zorder=5,
        )
        ax.set_ylabel(r'$V^2$')
        ax.set_ylim(1e-4, 1.1)

    ax_full.set_title("Squared Visibility (Log Scale)")
    ax_zoom.set_title("Post-First-Null Regime")
    if zoom is not None:
        ax_zoom.set_xlim(zoom[0], zoom[1])
        ax_zoom.set_ylim(zoom[2], zoom[3])
        if zoom[4] is not None and np.isfinite(zoom[4]):
            ax_full.axvline(zoom[4], color='gray', linestyle=':', linewidth=0.8, alpha=0.7)

    for ax in (ax_res_full, ax_res_zoom):
        ax.axhline(0, linestyle='--', color='gray', linewidth=0.6)
        ax.set_ylim(-11, 11)
        ax.grid(True, which='major', axis='y', linestyle=':', alpha=0.4)
        ax.set_xlabel(r'$B_\mathrm{max}/\lambda$ (M$\lambda$)')
        ax.set_ylabel(r'Residuals ($\sigma$)')
        for name, samp in model_samples.items():
            style = model_styles[name]
            ax.scatter(
                samp['bl'], samp['res'], s=10, color=style['color'],
                alpha=0.35, linewidths=0
            )
    if zoom is not None:
        ax_res_zoom.set_xlim(zoom[0], zoom[1])

    handles, labels = ax_full.get_legend_handles_labels()
    ax_zoom.legend(handles, labels, loc='upper right', frameon=False, fontsize=9)

    band_label = {'h': 'H-band', 'k': 'K-band', 'both': 'H+K'}[view]
    plt.suptitle(f"{target_to_latex(target)} ({band_label})")
    plt.tight_layout()
    out_name = file_path.replace('/', '_') + f'_data_vs_grid_{view}.png'
    if savefig:
        plt.savefig(out_name, bbox_inches='tight', dpi=300)
    plt.close('all')
    print(f"Done: {out_name} [ApJ {band_label}]")
