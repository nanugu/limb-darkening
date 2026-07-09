from dataclasses import dataclass
from pathlib import Path
import csv
import os
import re

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "csv"
TARGET_LIST_CSV = CSV_DIR / "chara_target_list.csv"


@dataclass(frozen=True)
class TargetPair:
    mircx_file: str
    mystic_file: str
    init_diameter: float


@dataclass(frozen=True)
class TargetRun:
    mircx_file: str
    mystic_file: str
    target: str
    date: str
    teff: float
    logg: float
    feh: float
    mass: float
    diam_guess: float
    seed_diam: float


def norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    s = re.sub(r"[^0-9A-Za-z]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_").lower()


def ensure_dirs(*dirs):
    for dirname in dirs:
        os.makedirs(dirname, exist_ok=True)


def sigma_or(default_val, *candidates):
    for candidate in candidates:
        try:
            candidate = float(candidate)
            if np.isfinite(candidate) and candidate > 0:
                return candidate
        except Exception:
            pass
    return default_val


def append_csv_row(csv_path: str | Path, row: dict) -> None:
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(csv_path).exists()
    with open(csv_path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def boxcar_throughput(wmin: float, wmax: float, dw: float = 10.0, margin: float = 200.0):
    w = np.arange(wmin - margin, wmax + margin + dw, dw, dtype=float)
    t = np.where((w >= wmin) & (w <= wmax), 1.0, 0.0)
    return w, t


def parse_l2_filename(file_name: str):
    parts = Path(file_name).name.split(".")
    if len(parts) >= 4:
        instrument = parts[0].upper()
        if instrument == "MIRC_L2":
            instrument = "MIRCX_L2"
        date = parts[1]
        end = None
        for idx in range(2, len(parts)):
            if parts[idx].upper() in {"MIRCX_IDL", "MYSTIC_IDL", "MIRC_IDL"}:
                end = idx
                break
        if end is None:
            end = 3
        target = ".".join(parts[2:end]).strip(".")
        return {"instrument": instrument, "date": date, "target": target}
    return None


def resolve_oifits_path(file_name, root: str | Path, *, require_exists=False, mirc_aliases=False):
    if not isinstance(file_name, str) or not file_name.strip():
        return None if require_exists else file_name

    file_name = file_name.strip()
    if os.path.isabs(file_name):
        if not require_exists or os.path.exists(file_name):
            return file_name
        base = os.path.basename(file_name)
        search_root = os.path.dirname(file_name)
    else:
        base = file_name
        search_root = str(root)

    candidates = [base]
    if mirc_aliases and base.startswith("MIRC_L2."):
        candidates.append("MIRCX_L2." + base[len("MIRC_L2."):])
    if mirc_aliases and base.startswith("MIRCX_L2."):
        candidates.append("MIRC_L2." + base[len("MIRCX_L2."):])

    for candidate in candidates:
        path = os.path.join(search_root, candidate)
        if not require_exists or os.path.exists(path):
            return path
    return None


def get_tf_params(oi_data, obs=None, with_v_slope=False):
    if isinstance(oi_data, list):
        res = {}
        for data_block in oi_data:
            res.update(get_tf_params(data_block, obs=obs, with_v_slope=with_v_slope))
        return res
    if obs is None and 'fit' in oi_data and 'obs' in oi_data['fit']:
        obs = oi_data['fit']['obs']
    res = {}
    ext = {'V2': 'OI_VIS2', '|V|': 'OI_VIS', 'T3PHI': 'OI_T3'}
    ext = {k: ext[k] for k in obs}
    for observable in ext:
        for name in oi_data[ext[observable]]:
            if 'VIS' in ext[observable]:
                res[f'#TF_{observable}_{name}_*'] = 1.0
                if with_v_slope:
                    res[f'#TF_{observable}_{name}_s'] = 0.01
            else:
                res[f'#TF_{observable}_{name}_+'] = 0.01
    return res


def add_transfer_function_params(oi, model_params, v2_0=0.8):
    params = model_params.copy()
    tf_params = get_tf_params(oi.data)
    params.update({k: '$V2_0' if '_V2_' in k else v for k, v in tf_params.items()})
    params['V2_0'] = v2_0
    return params


def load_target_pairs(target_list_csv: str | Path = TARGET_LIST_CSV):
    df = pd.read_csv(target_list_csv)
    required = {"mircx_file", "mystic_file", "init_diameter"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"{target_list_csv} is missing columns: {sorted(missing)}")

    pairs = []
    for row in df.itertuples(index=False):
        pairs.append(TargetPair(
            mircx_file=str(getattr(row, "mircx_file")),
            mystic_file=str(getattr(row, "mystic_file")),
            init_diameter=float(getattr(row, "init_diameter")),
        ))
    return pairs


def load_target_table(csv_path: str):
    df = pd.read_csv(csv_path)
    if "Target" not in df.columns and "target" in df.columns:
        df["Target"] = df["target"]
    if "Target" not in df.columns:
        raise KeyError(f"{csv_path} has no Target or target column")
    df["Target"] = df.apply(_fill_target_with_hd, axis=1)
    df["Target"] = df["Target"].astype(str).str.replace("_", " ").str.strip()
    return df


def _fill_target_with_hd(row):
    if pd.isna(row.get("Target")) or str(row["Target"]).strip() == "":
        return f"HD {int(row['HD'])}" if not pd.isna(row.get("HD")) else None
    return str(row["Target"])


def get_target_params(target_name: str, target_table, source_label: str = "target table"):
    tnorm = norm_name(target_name)
    exact = target_table["Target"].astype(str).apply(norm_name) == tnorm
    row = target_table[exact]
    if row.empty:
        contains = target_table["Target"].astype(str).apply(lambda s: tnorm in norm_name(s))
        row = target_table[contains]
    if row.empty:
        print(f"[WARN] Target '{target_name}' not found in {source_label}. Using defaults.")
        return 5190.0, 1.86, 0.0, 6.0, 2.5
    r = row.iloc[0]
    teff = r.get("Teff", r.get("teff", 5190.0))
    logg = r.get("logg", 1.86)
    feh = r.get("FeH", r.get("feh", 0.0))
    mass = r.get("Mass_value", r.get("mass", 6.0))
    theta = r.get("LDD_mas_bourges", r.get("theta_init", 3.0))
    return float(teff), float(logg), float(feh), float(mass), float(theta)


def iter_target_runs(
    limit=None,
    target_csv: str | None = None,
    target_table=None,
    target_list_csv: str | Path = TARGET_LIST_CSV,
):
    table = target_table if target_table is not None else load_target_table(target_csv)
    pairs = load_target_pairs(target_list_csv)
    if limit is not None:
        pairs = pairs[: int(limit)]

    for pair in pairs:
        info = parse_l2_filename(pair.mircx_file) or parse_l2_filename(pair.mystic_file) or {}
        target = info.get("target", "unknown_target")
        date = info.get("date", "")
        teff, logg, feh, mass, diam_guess = get_target_params(target, table, target_csv or "target table")
        seed_diam = pair.init_diameter if pd.notna(pair.init_diameter) else float(diam_guess)
        yield TargetRun(
            mircx_file=pair.mircx_file,
            mystic_file=pair.mystic_file,
            target=target,
            date=date,
            teff=teff,
            logg=logg,
            feh=feh,
            mass=mass,
            diam_guess=diam_guess,
            seed_diam=seed_diam,
        )
