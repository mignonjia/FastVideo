---
date: 2026-03-25
experiment: wangame_1.3b_output/wangame_1.3b_478.err
category: infrastructure
severity: important
---

# Validation Dataset HF FileLock Exhausted Distributed Launches

## What Happened
`wangame_1.3b_478.err` failed during the first validation pass before training
steps began. Multiple ranks crashed in `ValidationDataset` with
`OSError: [Errno 37] No locks available`.

## Root Cause
`fastvideo/dataset/validation_dataset.py` loaded local validation files through
`datasets.load_dataset(...)`. In a 32-rank launch, each rank tried to create and
release HuggingFace dataset cache locks for the same local JSON file. On this
system, the lock backend exhausted available file locks and validation aborted.

## Fix / Workaround
Load validation `.csv`, `.json`, `.parquet`, and `.arrow` files directly from
disk inside `ValidationDataset` instead of routing through HuggingFace Datasets.
This keeps validation local, avoids cache/filelock contention, and preserves the
existing sample sharding logic.

## Prevention
For small local validation manifests, prefer direct parsers over dataset-cache
builders that introduce cross-rank filesystem coordination. When triaging
distributed launch failures, inspect rank-local validation setup before assuming
the model or training step failed.
