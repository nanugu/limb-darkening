# Power-law limb-darkened model images

This package contains model images generated from `csv/merged_for_plots_merged.csv`
using [src/make_images.py](../../src/make_images.py).

## Contents

- `png/`: quick-look raster figures suitable for browser preview.
- `pdf/`: vector-friendly figure exports better suited for archival reuse.
- `manifest.csv`: per-file metadata linking each figure to the fitted parameters.

## Model definition

Each image uses the PMOIRED profile

`I(mu) proportional to mu^alpha`

with:

- `PL_diam` or `theta_PL`: fitted power-law angular diameter in mas
- `H_PL_alpha` or `alpha_chara_H`: H-band power-law coefficient
- `K_PL_alpha` or `alpha_chara_K`: K-band power-law coefficient

## Wavelength ranges

- Left panel: H band, 1.50-1.72 um
- Right panel: K band, 2.00-2.37 um

## Rendering notes

- Display stretch: square-root via `imPow=0.5`
- Pixel scale: 0.02 mas
- Field of view: `max(5.5 mas, 1.5 x fitted diameter)`
- Axes are in mas

## Reproducibility

To regenerate this package from the repository root:

```bash
python src/make_images.py
```

Generated files: 31
Source CSV: `csv/merged_for_plots_merged.csv`
