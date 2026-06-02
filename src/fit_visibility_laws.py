# pmoired313_chara_hk_combined_fit4.py
import os
import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
import pmoired
from pmoired import satlas
from pmoired_help_plot import PlotContext, overplot_grid_models, overplot_analytical_laws
from chara_fit_common import (
    add_transfer_function_params,
    append_csv_row,
    ensure_dirs,
    iter_target_runs,
    norm_name,
    resolve_oifits_path,
    sigma_or,
)
import re
from datetime import datetime
from pathlib import Path

# ============ CONFIG ============
ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"
TARGET_SAMPLE_CSV = CSV_DIR / "merged_four_branches_wide.csv"
OIFITS_ROOT = os.environ.get("OIFITS_DIR", str(ROOT / "oifits"))
LOG_DIR_DEFAULT = "."
FIG_DIR_DEFAULT = "figs_hk_combined"
OUTPUT_FIT_RESULTS_CSV = CSV_DIR / "fit_visibility_laws.csv"
OUTPUT_FIT_RESULTS_TXT = 'fit_visibility_laws.txt'

# H_lower
WAVELENGTH_TABLE = [(1.50, 1.6), (2.0, 2.37)]
#H_higher
WAVELENGTH_TABLE = [(1.6, 1.72), (2.0, 2.37)]

#K_low
WAVELENGTH_TABLE = [(1.50, 1.72), (2.00, 2.18)]

#H_higher
WAVELENGTH_TABLE = [(1.50, 1.72), (2.18, 2.37)]

#Full H and K-band
WAVELENGTH_TABLE = [(1.50, 1.72), (2.0, 2.37)]

SAVEFIG = 1

@dataclass
class LawFit:
    model: dict
    uncertainty: dict
    chi2: float

@dataclass
class FitResults:
    ud: LawFit
    sal: LawFit
    ll: LawFit
    pl: LawFit
    pl_stagger: LawFit
    pl2: LawFit
    ql: LawFit
    exotic: dict

    def plot_bundle(self):
        return {
            'ud': self.ud.model,
            'sal': self.sal.model,
            'll': self.ll.model,
            'pl': self.pl.model,
            'pl_stagger': self.pl_stagger.model,
            'pl2': self.pl2.model,
            'ql': self.ql.model,
            'chi2': {
                'ud': self.ud.chi2,
                'sal': self.sal.chi2,
                'll': self.ll.chi2,
                'pl': self.pl.chi2,
                'pl_stagger': self.pl_stagger.chi2,
                'pl2': self.pl2.chi2,
                'ql': self.ql.chi2,
            },
        }

def get_satlas_diam(model, default=np.nan):
    if not isinstance(model, dict):
        return default
    for key in ('ROSSDIAM', '*,ROSSDIAM', 'H,ROSSDIAM', 'K,ROSSDIAM'):
        if key in model:
            return model.get(key, default)
    return default

def build_combined_satlas_model(filename, rossdiam_seed):
    sat_h = satlas.readFile(filename, band='H', component='H')
    sat_k = satlas.readFile(filename, band='K', component='K')
    model = {}
    model.update(sat_h)
    model.update(sat_k)
    model.update({
        'H,ROSSDIAM': float(rossdiam_seed),
        'K,ROSSDIAM': '$H,ROSSDIAM',
        'H,spectrum': '($WL<1.8)',
        'K,spectrum': '($WL>1.8)',
    })
    return model

# ============ fit wrapper ============
def fit_visibility_model(oi, param, file_path, fig_dir, tag, log_scale_only=False, prior=None, do_not_fit=None):
    param = add_transfer_function_params(oi, param, v2_0=0.8)

    fit_kwargs = {}
    if prior is not None:
        fit_kwargs['prior'] = prior
        print(prior)
        print(param)
    if do_not_fit is not None:
        fit_kwargs['doNotFit'] = do_not_fit

    oi.doFit(param, **fit_kwargs)
    result = oi.bestfit['best'].copy()
    uncer  = oi.bestfit['uncer'].copy()
    chi2   = oi.bestfit['chi2']

    # PMOIRED's verbose correlation-table printer can crash when a degenerate
    # fit produces NaN correlations; the bootstrap uncertainties are still valid.
    try:
        oi.bootstrapFit(50, verbose=1)
        result = oi.boot['best'].copy()
        uncer  = oi.boot['uncer'].copy()
        chi2   = oi.boot['chi2']
    except ValueError as exc:
        if "cannot convert float NaN to integer" not in str(exc):
            raise
        print(f"[bootstrap] WARNING: skipping correlation display for {tag}: {exc}")

    return result, uncer, chi2

# ============ priors (reused) ============
def build_priors_from_model(seed_dict, diam, V2_0):
    """
    seed_dict keys:
      - linear: {'a','a_err'}
      - power2: {'a1','a1_err','alpha1','alpha1_err'}
      - quad:   {'a','a_err','b','b_err'}
    """
    pri = {}

    if seed_dict.get('linear'):
        a = float(seed_dict['linear']['a'])
        a_err = sigma_or(0.10, seed_dict['linear'].get('a_err'))
        pri['linear'] = [
            ('H,a', '>', 0.0), ('H,a', '<', 1.0),
            ('K,a', '>', 0.0), ('K,a', '<', 1.0),
            ('H,a', '=', a, max(0.10, a_err)),
            ('K,a', '=', a, max(0.10, a_err)),
            ('diam', '=', float(diam), 0.2),
            ('V2_0', '=', float(V2_0), 0.2),
        ]

    if seed_dict.get('power1'):
        alpha = float(seed_dict['power1']['alpha'])
        alpha_err = sigma_or(0.10, seed_dict['power1'].get('alpha_err'))
        pri['power1'] = [
            ('H,alpha', '>', 0.0), ('H,alpha', '<', 1.0),
            ('K,alpha', '>', 0.0), ('K,alpha', '<', 1.0),
            ('H,alpha', '=', alpha, max(0.10, alpha_err)),
            ('K,alpha', '=', alpha, max(0.10, alpha_err)),
            ('diam', '=', float(diam), 0.2),
            ('V2_0', '=', float(V2_0), 0.2),
        ]

    if seed_dict.get('power2'):
        a1 = float(seed_dict['power2']['a1'])
        alpha1 = float(seed_dict['power2']['alpha1'])
        a1_e = sigma_or(0.12, seed_dict['power2'].get('a1_err'))
        al_e = sigma_or(0.30, seed_dict['power2'].get('alpha1_err'))
        pri['power2'] = [
            ('H,a1', '>', a1 - 0.2), ('H,a1', '<', a1 + 0.2),
            ('K,a1', '>', a1 - 0.2), ('K,a1', '<', a1 + 0.2),
            ('H,alpha1', '>', 0.1), ('H,alpha1', '<', 1.0),
            ('K,alpha1', '>', 0.1), ('K,alpha1', '<', 1.0),
            ('H,a1', '=', a1, max(0.12, a1_e)),
            ('K,a1', '=', a1, max(0.12, a1_e)),
            ('H,alpha1', '=', alpha1, max(0.30, al_e)),
            ('K,alpha1', '=', alpha1, max(0.30, al_e)),
            ('diam', '=', float(diam), 0.2),
            ('V2_0', '=', float(V2_0), 0.2),
        ]

    if seed_dict.get('quad'):
        qa = float(seed_dict['quad']['a'])
        qb = float(seed_dict['quad']['b'])
        qa_e = sigma_or(0.15, seed_dict['quad'].get('a_err'))
        qb_e = sigma_or(0.15, seed_dict['quad'].get('b_err'))
        pri['quad'] = [
            ('H,a', '>', qa - 0.2), ('H,a', '<', qa + 0.2),
            ('K,a', '>', qa - 0.2), ('K,a', '<', qa + 0.2),
            ('H,b', '>', qb - 0.2), ('H,b', '<', qb + 0.2),
            ('K,b', '>', qb - 0.2), ('K,b', '<', qb + 0.2),
            ('H,a', '=', qa, max(0.15, qa_e)),
            ('K,a', '=', qa, max(0.15, qa_e)),
            ('H,b', '=', qb, max(0.15, qb_e)),
            ('K,b', '=', qb, max(0.15, qb_e)),
            ('diam', '=', float(diam), 0.2),
            ('V2_0', '=', float(V2_0), 0.2),
        ]
    return pri

# ============ stagger seeding ============
def get_stagger_coeffs(band_or_instrument: str, target: str, exotic_csv_path: str = TARGET_SAMPLE_CSV):
    """
    Pull Stagger coefficients from the target table, if present.

    Internal names follow the paper notation:
      linear:    a
      power law: alpha
      power-2:   a1, alpha1
      quadratic: u1, u2

    band_or_instrument can be "H"/"K" or "MIRCX"/"MYSTIC".
    """
    empty = {'linear': None, 'power1': None, 'power2': None, 'quad': None}
    df = pd.read_csv(exotic_csv_path)
    target_col = "Target" if "Target" in df.columns else "target" if "target" in df.columns else None
    if target_col is None:
        print(f"[stagger] WARN: no target column found in {exotic_csv_path}")
        return empty

    tnorm = norm_name(target)
    m = df[target_col].astype(str).apply(norm_name) == tnorm
    if not m.any():
        m = df[target_col].astype(str).apply(lambda s: tnorm in norm_name(s))
    if not m.any():
        print(f"[stagger] WARN: target '{target}' not found in {exotic_csv_path}")
        return empty

    row = df[m].iloc[0]

    key = str(band_or_instrument).strip().upper()
    if key in ("H", "MIRCX", "MIRC"):
        band = "MIRCX"
    elif key in ("K", "MYSTIC"):
        band = "MYSTIC"
    else:
        band = key

    clean_columns = (
        f"a__linear__stagger__{band}",
        f"alpha__power1__stagger__{band}",
        f"a1__power2__stagger__{band}",
        f"alpha1__power2__stagger__{band}",
        f"u1__quadratic__stagger__{band}",
        f"u2__quadratic__stagger__{band}",
    )
    if not any(col in df.columns for col in clean_columns):
        return empty

    def _first_value(*columns):
        for col in columns:
            if col in df.columns and pd.notna(row[col]):
                return float(row[col])
        return None

    a = _first_value(f"a__linear__stagger__{band}")
    alpha = _first_value(f"alpha__power1__stagger__{band}")
    a1 = _first_value(f"a1__power2__stagger__{band}")
    alpha1 = _first_value(f"alpha1__power2__stagger__{band}")
    u1 = _first_value(f"u1__quadratic__stagger__{band}")
    u2 = _first_value(f"u2__quadratic__stagger__{band}")

    out = {'linear': None, 'power1':None, 'power2': None, 'quad': None}

    if a is not None:
        out['linear'] = {'a': a, 'a_err': None}
    if alpha is not None:
        out['power1'] = {'alpha': alpha, 'alpha_err': None}
    if a1 is not None and alpha1 is not None:
        out['power2'] = {'a1': a1, 'alpha1': alpha1, 'a1_err': None, 'alpha1_err': None}
    if u1 is not None and u2 is not None:
        out['quad'] = {'a': u1, 'b': u2, 'a_err': None, 'b_err': None}

    return out

def _datecode_to_iso(date_code):
    try:
        dt = datetime.strptime(date_code, "%Y%b%d")
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return "NA"

def parse_l2_filename(filename):
    base = os.path.basename(filename)
    m = re.match(
        r"^(MIRCX|MYSTIC|MIRC)_L2\.(\d{4}[A-Za-z]{3}\d{2})\.(.+?)\.(?:MIRCX|MYSTIC|MIRC)_IDL\.",
        base,
    )
    if not m:
        return None
    instrument = m.group(1).upper()
    if instrument == "MIRC":
        instrument = "MIRCX"
    date_code = m.group(2)
    target = m.group(3)
    return instrument, date_code, target

# ============ logging ============
def log_summary(oi, mircx_path, mystic_path, target, Teff, logg, FeH, log_dir, results):
    ud, ud_uncer, chi2_ud = results.ud.model, results.ud.uncertainty, results.ud.chi2
    sal, sal_uncer, chi2_sal = results.sal.model, results.sal.uncertainty, results.sal.chi2
    ll, ll_uncer, chi2_ll = results.ll.model, results.ll.uncertainty, results.ll.chi2
    pl, pl_uncer, chi2_pl = results.pl.model, results.pl.uncertainty, results.pl.chi2
    pl2, pl2_uncer, chi2_pl2 = results.pl2.model, results.pl2.uncertainty, results.pl2.chi2
    ql, ql_uncer, chi2_ql = results.ql.model, results.ql.uncertainty, results.ql.chi2
    exotic = results.exotic

    mircx_base = os.path.basename(mircx_path)
    mystic_base = os.path.basename(mystic_path)
    parsed = parse_l2_filename(mircx_base) or parse_l2_filename(mystic_base)
    mydate = _datecode_to_iso(parsed[1]) if parsed else "NA"
    instrument = "MIRCX+MYSTIC"

    # Combined H+K overplot is generated in fit_target_hk_pair() with selectable band_view.

    def _fmt(val):
        try:
            return f"{float(val):.3f}"
        except Exception:
            return ""

    def _mget(mapping, key, default=np.nan):
        return mapping.get(key, default) if isinstance(mapping, dict) else default

    row = {
        "oifits_h": mircx_base,
        "oifits_k": mystic_base,
        "target": target,
        "date": mydate,
        "instrument": instrument,
        "teff": _fmt(Teff),
        "logg": _fmt(logg),
        "feh": _fmt(FeH),
        "v2_0_SATLAS": _fmt(_mget(sal, 'V2_0')),
        "theta_CLV_SATLAS": _fmt(get_satlas_diam(sal)),
        "theta_CLV_SATLAS_err": _fmt(get_satlas_diam(sal_uncer)),
        "chi2_red_SATLAS": _fmt(chi2_sal),
        "v2_0_ud": _fmt(_mget(ud, 'V2_0')),
        "theta_ud_H": _fmt(_mget(ud, 'H,ud')),
        "theta_ud_H_err": _fmt(_mget(ud_uncer, 'H,ud')),
        "theta_ud_K": _fmt(_mget(ud, 'K,ud')),
        "theta_ud_K_err": _fmt(_mget(ud_uncer, 'K,ud')),
        "chi2_red_ud": _fmt(chi2_ud),
        "v2_0_PL": _fmt(_mget(pl, 'V2_0')),
        "theta_PL": _fmt(_mget(pl, 'diam')),
        "theta_PL_err": _fmt(_mget(pl_uncer, 'diam')),
        "alpha_chara_H": _fmt(_mget(pl, 'H,alpha')),
        "alpha_chara_H_err": _fmt(_mget(pl_uncer, 'H,alpha')),
        "alpha_chara_K": _fmt(_mget(pl, 'K,alpha')),
        "alpha_chara_K_err": _fmt(_mget(pl_uncer, 'K,alpha')),
        "chi2_red_PL": _fmt(chi2_pl),
        "v2_0_power2": _fmt(_mget(pl2, 'V2_0')),
        "theta_power2": _fmt(_mget(pl2, 'diam')),
        "theta_power2_err": _fmt(_mget(pl2_uncer, 'diam')),
        "alpha1_chara_H": _fmt(_mget(pl2, 'H,alpha1')),
        "alpha1_chara_H_err": _fmt(_mget(pl2_uncer, 'H,alpha1')),
        "alpha1_chara_K": _fmt(_mget(pl2, 'K,alpha1')),
        "alpha1_chara_K_err": _fmt(_mget(pl2_uncer, 'K,alpha1')),
        "p1_chara_H": _fmt(_mget(pl2, 'H,a1')),
        "p1_chara_H_err": _fmt(_mget(pl2_uncer, 'H,a1')),
        "p1_chara_K": _fmt(_mget(pl2, 'K,a1')),
        "p1_chara_K_err": _fmt(_mget(pl2_uncer, 'K,a1')),
        "chi2_red_power2": _fmt(chi2_pl2),
        "v2_0_linear": _fmt(_mget(ll, 'V2_0')),
        "theta_linear": _fmt(_mget(ll, 'diam')),
        "theta_linear_err": _fmt(_mget(ll_uncer, 'diam')),
        "a_chara_H": _fmt(_mget(ll, 'H,a')),
        "a_chara_H_err": _fmt(_mget(ll_uncer, 'H,a')),
        "a_chara_K": _fmt(_mget(ll, 'K,a')),
        "a_chara_K_err": _fmt(_mget(ll_uncer, 'K,a')),
        "chi2_red_linear": _fmt(chi2_ll),
        "v2_0_quadratic": _fmt(_mget(ql, 'V2_0')),
        "theta_quadratic": _fmt(_mget(ql, 'diam')),
        "theta_quadratic_err": _fmt(_mget(ql_uncer, 'diam')),
        "u1_chara_H": _fmt(_mget(ql, 'H,a')),
        "u1_chara_H_err": _fmt(_mget(ql_uncer, 'H,a')),
        "u1_chara_K": _fmt(_mget(ql, 'K,a')),
        "u1_chara_K_err": _fmt(_mget(ql_uncer, 'K,a')),
        "u2_chara_H": _fmt(_mget(ql, 'H,b')),
        "u2_chara_H_err": _fmt(_mget(ql_uncer, 'H,b')),
        "u2_chara_K": _fmt(_mget(ql, 'K,b')),
        "u2_chara_K_err": _fmt(_mget(ql_uncer, 'K,b')),
        "chi2_red_quadratic": _fmt(chi2_ql),
    }
    csv_path = os.path.join(log_dir, OUTPUT_FIT_RESULTS_CSV)
    append_csv_row(csv_path, row)

def fit_target_hk_pair(
    mircx_file,
    mystic_file,
    diam=3.0, Teff=5190.0, logg=1.86, mass=6.0, FeH=0.0,
    target='89_Her', log_dir=LOG_DIR_DEFAULT, fig_dir=FIG_DIR_DEFAULT, band_view='H'
):
    mircx_path = resolve_oifits_path(mircx_file, OIFITS_ROOT)
    mystic_path = resolve_oifits_path(mystic_file, OIFITS_ROOT)
    fit_label = f"{os.path.basename(mircx_path)}__{os.path.basename(mystic_path)}"

    ensure_dirs(log_dir, fig_dir)
    oi = pmoired.OI([mircx_path, mystic_path],  verbose=False)
    v2_err = 0.03
    wl=WAVELENGTH_TABLE

    print(f"\n{mircx_path}\n{mystic_path}\n  target={target}  mode=H+K combined")
    print(f"  seeds: diam={diam} Teff={Teff} logg={logg} [M/H]={FeH} mass={mass}\n")
    
    oi.setupFit({
        'obs': ['V2'],
        'min relative error': {'V2': v2_err},
        'max error': {'V2': 0.2},
        'wl ranges': wl,
    })

    # Use Stagger coefficients for priors/fixed-alpha comparison when they are
    # present in the input table. Plain target_metadata.csv does not include them.
    seeds = get_stagger_coeffs('H', target, TARGET_SAMPLE_CSV)
    seeds_k = get_stagger_coeffs('K', target, TARGET_SAMPLE_CSV)

    ud_param = {
        'H,ud': diam,
        'H,spectrum': '($WL<1.8)',
        'K,ud': diam,
        'K,spectrum': '($WL>1.8)',
    }

    ud, ud_uncer, chi2_ud = fit_visibility_model(oi, ud_param, fit_label, fig_dir, 'UD')
    diam_seed = float(np.nanmean([ud.get('H,ud', diam), ud.get('K,ud', diam)]))

    # SATLAS spherical
    filename = satlas.getClosestModel(Teff, logg, mass, verbose=True)
    model = build_combined_satlas_model(filename, diam_seed)
    sal, sal_uncer, chi2_sal = fit_visibility_model(oi, model, fit_label, fig_dir, 'SATLAS')

    V2_0_seed = float(sal.get('V2_0', 1.0))
    pri = build_priors_from_model(seeds, diam_seed, V2_0_seed)

    p_param = {
        'diam': diam_seed,
        'H,alpha': 0.2,
        'K,alpha': 0.2,
        'H,diam': '$diam',
        'H,profile': '$MU**$H,alpha',
        'H,spectrum': '($WL<1.8)',
        'K,diam': '$diam',
        'K,profile': '$MU**$K,alpha',
        'K,spectrum': '($WL>1.8)',
    }
    if seeds.get('power1') and seeds['power1'].get('alpha') is not None:
        p_param['H,alpha'] = float(seeds['power1']['alpha'])
        p_param['K,alpha'] = float(seeds['power1']['alpha'])

    pl, pl_uncer, chi2_pl = fit_visibility_model(
        oi,
        p_param,
        fit_label, fig_dir, 'Power Law',
        prior=pri.get('power1'),
    )

    pl_stagger = None
    chi2_pl_stagger = np.nan
    if seeds.get('power1') and seeds_k.get('power1'):
        pl_stagger_param = {
            'diam': float(pl.get('diam', diam_seed)),
            'H,alpha': float(seeds['power1']['alpha']),
            'K,alpha': float(seeds_k['power1']['alpha']),
            'H,diam': '$diam',
            'H,profile': '$MU**$H,alpha',
            'H,spectrum': '($WL<1.8)',
            'K,diam': '$diam',
            'K,profile': '$MU**$K,alpha',
            'K,spectrum': '($WL>1.8)',
        }
        pl_stagger, _, chi2_pl_stagger = fit_visibility_model(
            oi,
            pl_stagger_param,
            fit_label,
            fig_dir,
            'Stagger Power Law',
            log_scale_only=True,
            do_not_fit=['H,alpha', 'K,alpha'],
        )
    else:
        print("[stagger] INFO: skipping stagger power-law fit; missing H/K power1 coefficients.")

    ll_param = {
        'diam': diam_seed,
        'H,a': 0.2,
        'K,a': 0.2,
        'H,diam': '$diam',
        'H,profile': '1 - $H,a*(1 - $MU)',
        'H,spectrum': '($WL<1.8)',
        'K,diam': '$diam',
        'K,profile': '1 - $K,a*(1 - $MU)',
        'K,spectrum': '($WL>1.8)',
    }
    if seeds.get('linear'):
        ll_param['H,a'] = seeds['linear']['a']
        ll_param['K,a'] = seeds['linear']['a']

    ll, ll_uncer, chi2_ll = fit_visibility_model(
        oi, ll_param, fit_label, fig_dir, 'Linear Law',
        prior=pri.get('linear')
    )

    p2_param = {
        'diam': diam_seed,
        'H,a1': 0.5,
        'K,a1': 0.5,
        'H,alpha1': 1.0,
        'K,alpha1': 1.0,
        'H,diam': '$diam',
        'H,profile': '1-$H,a1*(1-($MU**$H,alpha1))',
        'H,spectrum': '($WL<1.8)',
        'K,diam': '$diam',
        'K,profile': '1-$K,a1*(1-($MU**$K,alpha1))',
        'K,spectrum': '($WL>1.8)',
    }
    if seeds.get('power2'):
        p2_param.update({'H,a1': seeds['power2']['a1'], 'H,alpha1': seeds['power2']['alpha1']})
        p2_param.update({'K,a1': seeds['power2']['a1'], 'K,alpha1': seeds['power2']['alpha1']})

    pl2, pl2_uncer, chi2_pl2 = fit_visibility_model(
        oi, p2_param, fit_label, fig_dir, 'Power-2 Law',
        prior=pri.get('power2')
    )

    q_param = {
        'diam': diam_seed,
        'H,a': 0.3,
        'K,a': 0.3,
        'H,b': 0.2,
        'K,b': 0.2,
        'H,diam': '$diam',
        'H,profile': '1 - $H,a*(1 - $MU) - $H,b*(1 - $MU)**2',
        'H,spectrum': '($WL<1.8)',
        'K,diam': '$diam',
        'K,profile': '1 - $K,a*(1 - $MU) - $K,b*(1 - $MU)**2',
        'K,spectrum': '($WL>1.8)',
    }
    if seeds.get('quad'):
        q_param.update({'H,a': seeds['quad']['a'], 'H,b': seeds['quad']['b']})
        q_param.update({'K,a': seeds['quad']['a'], 'K,b': seeds['quad']['b']})

    ql, ql_uncer, chi2_ql = fit_visibility_model(
        oi, q_param, fit_label, fig_dir, 'Quadratic Law',
        prior=pri.get('quad')
    )

    parsed = parse_l2_filename(os.path.basename(mircx_path)) or parse_l2_filename(os.path.basename(mystic_path))
    mydate = _datecode_to_iso(parsed[1]) if parsed else "NA"
    results = FitResults(
        ud=LawFit(ud, ud_uncer, chi2_ud),
        sal=LawFit(sal, sal_uncer, chi2_sal),
        ll=LawFit(ll, ll_uncer, chi2_ll),
        pl=LawFit(pl, pl_uncer, chi2_pl),
        pl_stagger=LawFit(pl_stagger, {}, chi2_pl_stagger),
        pl2=LawFit(pl2, pl2_uncer, chi2_pl2),
        ql=LawFit(ql, ql_uncer, chi2_ql),
        exotic=seeds,
    )
    plot_context = PlotContext(
        oi=oi,
        file_path=fit_label,
        target=target,
        instrument="MIRCX+MYSTIC",
        date=mydate,
    )
    plot_bundle = results.plot_bundle()
    for band in ('H', 'K'):
        overplot_grid_models(
            plot_context, plot_bundle, band_view=band, savefig=SAVEFIG
        )
        overplot_analytical_laws(
            plot_context, plot_bundle, band_view=band, savefig=SAVEFIG
        )

    log_summary(
        oi,
        mircx_path, mystic_path, target, Teff, logg, FeH, log_dir,
        results
    )

def run_fits(band_view='H', limit=None):
    runs = list(iter_target_runs(limit=limit, target_csv=TARGET_SAMPLE_CSV))
    print(f"[run_fits] selected pairs: {len(runs)}")
    for run in runs:
        print(
            f"[run_fits] {run.target}  {run.mircx_file} + {run.mystic_file} "
            f"seed_diam={run.seed_diam:.2f}"
        )
        fit_target_hk_pair(
            run.mircx_file,
            run.mystic_file,
            diam=run.seed_diam,
            Teff=run.teff,
            logg=run.logg,
            FeH=run.feh,
            mass=run.mass,
            target=run.target,
            log_dir=LOG_DIR_DEFAULT,
            fig_dir=FIG_DIR_DEFAULT,
            band_view=band_view,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit CHARA H+K OIFITS files with analytic limb-darkening laws.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N OIFITS pairs.")
    parser.add_argument("--band-view", default="H", choices=["H", "K"], help="Band view used for diagnostic plots.")
    args = parser.parse_args()
    run_fits(band_view=args.band_view, limit=args.limit)
