#!/usr/bin/env python3
"""Shared public CSV schema helpers for the CHARA limb-darkening scripts."""

from __future__ import annotations

import math
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

GREEK_MAP = {
    "α": "alf", "β": "bet", "γ": "gam", "δ": "del", "ε": "eps", "ζ": "zet",
    "η": "eta", "θ": "the", "ι": "iot", "κ": "kap", "λ": "lam", "μ": "mu",
    "ν": "nu", "ξ": "ksi", "ο": "omi", "π": "pi", "ρ": "rho", "σ": "sig",
    "ς": "sig", "τ": "tau", "υ": "ups", "φ": "phi", "χ": "chi", "ψ": "psi",
    "ω": "ome",
}

GRID_PUBLIC = {
    "kurucz": "Kurucz",
    "stagger": "Stagger",
    "mps1": "MPS1",
    "mps2": "MPS2",
    "satlas": "SATLAS",
}


def normalize_target(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return ""
    text = "".join(GREEK_MAP.get(ch, ch) for ch in text)
    text = text.replace(".", " ")
    text = re.sub(r"[^0-9A-Za-z]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_").lower()
    return text


def clean_grid(value: object) -> str:
    key = normalize_target(value)
    return GRID_PUBLIC.get(key, str(value).strip())


def find_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    exact = {c: c for c in df.columns}
    lower = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in exact:
            return exact[name]
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def first_present(row: pd.Series | dict, names: Iterable[str], default=np.nan):
    for name in names:
        if name in row:
            val = row[name]
            if not (isinstance(val, float) and math.isnan(val)):
                if str(val).strip() != "":
                    return val
    return default


def to_float(value: object, default=np.nan) -> float:
    try:
        val = float(value)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def load_target_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "target_norm" not in df.columns:
        target_col = find_col(df, ["target", "Target", "Target_chara", "Name"])
        if target_col is None:
            raise ValueError(f"{path} has no target column")
        df["target_norm"] = df[target_col].map(normalize_target)
    return df


def base_metadata_from_row(row: pd.Series | dict) -> dict:
    target = first_present(row, ["target", "Target", "Target_chara", "TARGET"], "")
    target_norm = first_present(row, ["target_norm", "Target_norm"], "")
    target_norm = normalize_target(target_norm) or normalize_target(target)
    return {
        "target": target,
        "target_norm": target_norm,
        "hd": first_present(row, ["hd", "HD", "HD_number"], ""),
        "date": first_present(row, ["date", "Date", "DATE", "Date_first"], ""),
        "oifits_h": first_present(row, ["oifits_h", "MIRCX_file", "MIRCX_FILE", "MIRCX_file_first"], ""),
        "oifits_k": first_present(row, ["oifits_k", "MYSTIC_file", "MYSTIC_FILE", "MYSTIC_file_first"], ""),
        "teff": to_float(first_present(row, ["teff", "Teff", "TEFF"])),
        "logg": to_float(first_present(row, ["logg", "LOGG"])),
        "feh": to_float(first_present(row, ["feh", "FeH", "M_H", "M_H_baines"])),
        "theta_init": to_float(first_present(row, ["theta_init", "LDD_mas_bourges", "diam", "PL_diam"])),
        "mass": to_float(first_present(row, ["mass", "Mass_value", "MASS"])),
    }
