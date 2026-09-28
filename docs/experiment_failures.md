# Superseded modeling experiments

The final benchmark uses the TNT-style, character-level, non-autoregressive
edit Transformer. Earlier notebooks were removed from the public tree after
their conclusions were documented here. Local copies may be retained under the
ignored `local_archive/experimental_notebooks/` directory.

## mT5 and LoRA route

The original M7 evaluator used mT5-small, later with LoRA and larger batches.
Even with adapters, the 300M-parameter backbone made iteration slow on the
available laptop GPU. The very small Gold split also produced weak or unstable
generations. Additional epochs could not be justified fairly when each
synthetic condition required the same expensive schedule. This route was
therefore unsuitable as the controlled evaluator for the final benchmark.

## Autoregressive compact Transformer

A small character Transformer trained much faster and could overfit short
sanity samples. It nevertheless became unreliable as sequences grew longer:
early errors shifted later decoding, malformed continuations accumulated, and
development outputs could collapse despite decreasing teacher-forced loss.
Autoregressive evaluation was also disproportionately slow because characters
were generated one step at a time.

## Copy-aware variants

The copy gate and positional copy priors were intended to preserve unchanged
characters. In practice, the model often learned the identity shortcut,
especially for short or single-token inputs. Absolute positional bias also
became incorrect after an insertion or deletion shifted the remaining target.
Relative-copy and edit-weighted variants reduced some symptoms but introduced
additional evaluator assumptions and did not provide a stable, method-neutral
benchmark.

## Synthetic-development checkpoint selection

Selecting the pretraining checkpoint on a held-out slice of each synthetic
dataset was investigated to avoid consulting Gold development during synthetic
pretraining. The generated evaluation loop was too slow for the autoregressive
models, and method-specific synthetic validation distributions would not be
directly comparable. The final fixed protocol instead uses the same Gold
development split for checkpoint selection across every condition and reports
the frozen Gold test only once.

## M6 hierarchical preview

An exploratory hierarchical M6 notebook tested more explicit generalization
from lexical mappings to POS/context character templates. It was a preview, not
the source of the reported M6 dataset. The final M6 implementation and its
provenance remain in `m6_gold_linguistic_engine.ipynb` and
`src/generation/gold_linguistic_engine.py`.

These failures are methodological evidence, but their notebook outputs are not
part of the final reported result table.
