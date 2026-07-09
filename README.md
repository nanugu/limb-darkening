# CHARA H/K Limb-Darkening Analysis

This repository contains the frozen public analysis scripts, processed data
products, and generated figures associated with the CHARA H- and K-band
limb-darkening study. It is intended as the archive companion to the
manuscript and is structured so that a reader can trace the released CSV
products and figure materials back to the corresponding analysis branches used
in the paper.

## Archive scope

This archive includes:

- processed branch-level CSV products
- a merged per-target summary table
- generated figure products used in the paper workflow
- compact metadata tables needed for the public summary products
- the frozen source code used to generate the released processed products

This archive does not include:

- raw CHARA OIFITS files ([CHARA Observer Database](https://chara.gsu.edu/observers/database)) or calibrated CHARA OIFITS files ([JMMC OIDB](https://oidb.jmmc.fr/))
- local ExoTiC-LD stellar-grid cache files
- external PMOIRED resources installed outside this repository
- the author's full working directory structure

Several released CSV products retain OIFITS filenames or target-level
bookkeeping needed for provenance. These are included as metadata only; the
corresponding interferometric files are not bundled here.

## Repository layout

- `src/` contains the Python analysis scripts and shared helper modules.
- `csv/` contains the released processed CSV products and metadata tables.
- `figures/` contains the generated figure products archived with the release.

## Canonical public data products

The main released CSV products are:

- `csv/fit_visibility_laws.csv`
- `csv/compute_exotic_coefficients.csv`
- `csv/compute_svam_coefficients.csv`
- `csv/fit_diameters_with_direct_CLV.csv`
- `csv/merged_four_branches_wide.csv`

The recommended starting point for most downstream use is:

- `csv/merged_four_branches_wide.csv`

This merged table contains one row per target and combines the empirical CHARA
fits, direct intensity-domain model coefficients, synthetic-visibility (SVAM)
coefficients, and fixed-CLV diameter products used throughout the paper.

## Atmosphere-model families included

The public products compare five atmosphere-model families:

- Kurucz
- MPS1
- MPS2
- Stagger
- SATLAS

In the manuscript and in these released data products, `SATLAS` refers to the
Rosseland-radius coefficient convention adopted for the spherical SATLAS
comparison.

## Figure organization

The public `figures/` directory is organized to match the manuscript figure
numbering as closely as practical. The main subfolders currently include:

- `Figure2_CHARA_V2_ld_laws_fit`
- `Figure6_data_vs_model_comparison`
- `Figure7_closure_phase_companion_search`
- `Figure8_power_law_images_physical`
- `Figure15_chara_model_intensity_space_comparison`

This structure is intended to make it easy to map archived figure materials to
specific figures discussed in the paper.

## Main scripts

- `src/fit_visibility_laws.py`
  Fits the calibrated CHARA H+K visibilities with empirical analytic
  limb-darkening laws and writes `csv/fit_visibility_laws.csv`.
- `src/compute_exotic_coefficients.py`
  Computes model coefficients by fitting analytic laws directly to
  passband-integrated atmosphere-grid intensity profiles, `I(mu)`, and writes
  `csv/compute_exotic_coefficients_LONG.csv`.
- `src/compute_svam_coefficients.py`
  Computes SVAM coefficients by generating synthetic CHARA-sampled visibilities
  from model CLVs, refitting them with the same analytic laws, and writing
  `csv/compute_svam_coefficients.csv`.
- `src/fit_diameters_with_direct_CLV.py`
  Fits CHARA visibilities with fixed atmosphere-model CLV profiles and writes
  `csv/fit_diameters_with_direct_CLV.csv`.
- `src/merge_four_csvs.py`
  Merges the branch CSVs into `csv/merged_four_branches_wide.csv`.
- `src/All_APJ_figures.py`
  Generates the figures and prints the numerical summaries used in the paper.

## Shared helper modules

- `src/chara_fit_common.py`
- `src/public_schema.py`
- `src/pmoired_help_plot.py`
- `src/svam_recovery.py`

## How to use this archive

For most users, the intended use is to work from the released processed CSV
products rather than rerun the full raw-data pipeline.

Typical use cases are:

- inspect `csv/merged_four_branches_wide.csv` for the released per-target
  summary products
- use the branch CSV files to reproduce paper-level comparisons and summary
  statistics
- use `figures/` to access the archived figure products associated with the
  manuscript

A user working only from the released CSV products can reproduce most
paper-level figures and numerical summary tables without access to the raw
OIFITS files.

## Typical workflow from released products

The four branch CSVs can be regenerated independently if the required raw data
and model resources are available:

```bash
python src/fit_visibility_laws.py
python src/compute_exotic_coefficients.py
python src/compute_svam_coefficients.py
python src/fit_diameters_with_direct_CLV.py
```

Then merge the branch outputs:

```bash
python src/merge_four_csvs.py
```

Then regenerate the paper figures and console summaries:

```bash
python src/All_APJ_figures.py --csv csv/merged_four_branches_wide.csv --outdir figures
```

## Requirements for a full rerun from raw inputs

A user attempting to rerun the full fit pipeline from scratch will need:

- local calibrated CHARA OIFITS files
- a working PMOIRED installation
- a working ExoTiC-LD installation
- local ExoTiC-LD stellar-grid data
- any external SATLAS resources expected by the PMOIRED-based workflow

Several scripts support environment-based configuration, especially through:

- `OIFITS_DIR`
- `EXOTIC_LD_DATA_DIR`

## Public-release notes

This repository is intended as a frozen analysis-and-results archive associated
with the manuscript. It should be interpreted as the citable public release of
the exact scripts, figures, and processed CSV products used for the paper,
rather than as a fully containerized one-command reproduction package.


## Python dependencies

The scripts use standard scientific Python packages plus domain-specific
packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `pmoired`
- `exotic-ld`

For reproducibility, this archive was prepared against the following code
states of the external packages used in the workflow:

- ExoTiC-LD: [nanugu/ExoTiC-LD.git](https://github.com/nanugu/ExoTiC-LD.git)
  at commit `1bcce41c8c0059c85fb511d23248c6943ea64669` ("Add power law")
- PMOIRED: [amerand/PMOIRED](https://github.com/amerand/PMOIRED)
  at commit `aa2adee1783d836dadba7e46204043dca66f6a69` ("handling of FLUXDATA (finally!)")


