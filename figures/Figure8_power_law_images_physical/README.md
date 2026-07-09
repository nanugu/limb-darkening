# Power-law limb-darkened model images on a physical scale

This package contains the same PMOIRED power-law model images as the angular
set, but re-rendered and relabeled on a physical-radius scale using the
`Rstar` values in `csv/merged_four_branches_wide.csv`.

## Contents

- `png/`: quick-look raster figures suitable for browser preview.
- `pdf/`: vector-friendly figure exports better suited for archival reuse.
- `manifest.csv`: per-file metadata including the adopted stellar radius and
  the dynamic physical field of view.

## Rendering notes

- The underlying PMOIRED rendering is still performed in mas and then relabeled
  to physical units using the fitted angular diameter and the tabulated stellar
  radius.
- The physical full field of view is set per target to
  `max(6, 3.0 x Rstar)`
  in units of $R_\odot$, so each star remains visually resolved while the axis
  labels preserve the true physical scale.

## Reproducibility

To regenerate this package from the repository root:

```bash
python src/make_images.py
```

Generated files: 31
Source CSV: `csv/merged_four_branches_wide.csv`
