# CHARA H/K Limb-Darkening Analysis Scripts

This repository contains the public analysis scripts and processed CSV products
used for the CHARA H- and K-band limb-darkening paper. The workflow is organized
around four analysis branches that are merged into one wide table for plotting
and paper-level summaries.

## Repository layout

- `src/` contains the Python scripts and shared helper modules.
- `csv/` contains processed CSV products and small target metadata tables.

## What is included

### Main scripts

- `src/fit_visibility_laws.py` fits the calibrated CHARA H+K visibilities with
  empirical analytic limb-darkening laws and writes `csv/fit_visibility_laws.csv`.
- `src/compute_exotic_coefficients.py` computes model coefficients by fitting
  analytic laws directly to passband-integrated atmosphere-grid intensity
  profiles, I(mu), and writes `csv/compute_exotic_coefficients.csv`.
- `src/compute_svam_coefficients.py` computes SVAM coefficients by generating
  synthetic CHARA-sampled visibilities from model CLVs, refitting them with the
  same analytic laws, and writing `csv/compute_svam_coefficients.csv`.
- `src/fit_diameters_with_direct_CLV.py` fits CHARA visibilities with fixed
  atmosphere-model CLV profiles and writes
  `csv/fit_diameters_with_direct_CLV.csv`.
- `src/merge_four_csvs.py` merges the four branch CSVs into
  `csv/merged_four_branches_wide.csv`.
- `src/All_APJ_figures.py` generates the figures and prints the numerical
  summaries used in the paper.

### Shared helper modules

- `src/chara_fit_common.py` contains shared CHARA/OIFITS path handling, target-list
  utilities, throughput helpers, transfer-function parameters, and CSV writing
  helpers.
- `src/public_schema.py` contains shared public-output metadata and naming helpers.
- `src/pmoired_help_plot.py` contains plotting helpers used by the empirical
  visibility-fit branch.
- `src/svam_recovery.py` contains the synthetic-visibility recovery functions used
  by `src/compute_svam_coefficients.py`.

### Processed CSV products

- `csv/fit_visibility_laws.csv`
- `csv/fit_visibility_laws_Hlow.csv`
- `csv/fit_visibility_laws_Hhigh.csv`
- `csv/fit_visibility_laws_Klow.csv`
- `csv/fit_visibility_laws_Khigh.csv`
- `csv/compute_exotic_coefficients.csv`
- `csv/compute_svam_coefficients.csv`
- `csv/fit_diameters_with_direct_CLV.csv`
- `csv/merged_four_branches_wide.csv`

The merged table has one row per target and is the recommended starting point
for reproducing the paper figures and summary statistics.



## Typical workflow

The four branch CSVs can be regenerated independently:

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
python src/All_APJ_figures.py --csv csv/merged_four_branches_wide.csv --outdir Figures
```

The split-band wavelength diagnostic can be regenerated with:

```bash
python src/APJ_Fig_HK_plot_chara_H_vs_K_ldcs_same_night.py --csv-dir csv --outdir Figures
```

## Notes on inputs

Several scripts expect local CHARA OIFITS paths and target metadata. In the
author's working environment these are provided through:

- `csv/chara_target_list.csv`
- `csv/target_metadata.csv`
- local OIFITS files
- local ExoTiC-LD model data
- PMOIRED and ExoTiC-LD Python installations

For scripts that rerun fits from raw data, set `OIFITS_DIR` to the directory
containing the calibrated OIFITS files and `EXOTIC_LD_DATA_DIR` to the local
ExoTiC-LD model-data cache.

If only the processed CSVs are used, these local raw-data dependencies are not
needed for figure regeneration.

## Python dependencies

The scripts use standard scientific Python packages plus domain-specific
packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `pmoired`
- `exotic-ld`

