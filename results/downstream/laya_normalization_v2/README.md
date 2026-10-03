# Frozen downstream analysis V2

This directory contains the clean-preservation, M1-severity,
cross-method-generalization, and detailed V1 error analyses produced with the
exact frozen V1 Laya and TNT checkpoints. No training was performed.

Start with:

- `analysis_summary.md` for interpretation;
- `global_summary.json` for the complete machine-readable result;
- `experiment_manifest.json` for hashes, commit and configuration;
- `severity/severity_curve.csv` and
  `cross_method/cross_method_summary.csv` for portfolio figures;
- `error_analysis/*review.csv` for qualitative review.

V1 files under `../laya_normalization_v1/` were not overwritten.
