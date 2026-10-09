# Distillation Log

Started: 2026-10-08T06:09:53.190490
Seed: 42


## Filtering Summary
- Raw candidates: 1500
- Final kept: 1005
- Truncated dropped: -6
- Language dropped: -2
- Exact prompt dupes: -0
- Near-content dupes: -474
- Eval overlap dropped: -13
- Eval flagged for manual review: 0
- QC dropped: -0


## Sequence Length Analysis
- P95 Token Length: 1012
- Recommended max_seq_length: 1024
- Dataset Coverage: 95.2%


## Training and Evaluation Summary
- Model: `google/gemma-3-1b-it` (fp16 standard LoRA, r=16, alpha=32)
- Epochs: 2, Effective Batch Size: 16
- Training Loss: 1.7907
- Training Duration: 12.9 min
- Peak Training VRAM: 2.77 GB
- Peak Inference VRAM: Base=2.13 GB, Distilled=2.33 GB
- Predictions CSV: `data/predictions.csv` (60 rows)

