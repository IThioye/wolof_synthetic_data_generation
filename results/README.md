# Result registry

The canonical reported benchmark is:

- run family: `m7_seq2seq/tnt_edit_gold_dev_v4/`;
- frozen-test outputs: `m7_seq2seq/tnt_edit_gold_dev_v4/frozen_test/`;
- report tables and figures: `tnt_edit_report_analysis/`.

These directories contain lightweight protocols, selected-checkpoint records,
predictions, metrics, plots, and analysis tables. Model weights are deliberately
kept under the ignored `checkpoints/` directory.

Other result families are development history. They document mT5, compact
autoregressive, copy-aware, relative-copy, and earlier TNT attempts, but they
must not be mixed with the final table. New experiments should use a new named
run directory rather than overwrite a frozen result.

The separate downstream experiment writes to
`downstream/laya_normalization_v1/`. Its final files are absent until the user
runs Laya fine-tuning and explicitly enables the frozen Injongo test benchmark.

The follow-up frozen analyses use `downstream/laya_normalization_v2/`. V2 never
rewrites V1 and performs no additional Laya or TNT training.

Rebuild the final report analysis with:

```bash
python scripts/reporting/build_final_benchmark_analysis.py
```
