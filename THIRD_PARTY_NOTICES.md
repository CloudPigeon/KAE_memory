# Third-party notices

## HotpotQA lexical scorer

`src/prism/hotpot_official.py` preserves the scoring functions already present in the pinned method source. Those functions were vendored from `GAM/research/eval/hotpotqa_test.py` and `E-mem/evaluation/hotpotqa/eval_hotpotqa.py`, whose function bodies matched in the local source audit.

The E-mem distribution uses Apache License 2.0. Its license and notice are reproduced in `third_party/E-mem/`. The upstream notice identifies the adapted GAM code and reproduces GAM's MIT permission notice for Copyright (c) 2025 VectorSpaceLab.

The release changes the scorer's module description and removes research-local path references. Function bodies retain the pinned implementation.

## LoCoMo metric profile

`src/prism/metrics.py` implements the audited E-mem normalization and token-multiset F1 formula using the independent arithmetic implementation in the pinned source's `scripts/gpt4omini_metrics.py`. Optional BLEU-1 uses NLTK's tokenization and sentence BLEU functions with the same source settings.

## Dependencies and datasets

Installed dependencies retain their own licenses. Dataset distribution and usage follow the dataset maintainers' terms.
