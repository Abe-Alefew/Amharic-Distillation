# Amharic Capability Distillation — Final Report

## 1. Configuration
- **Teacher**: `google/gemma-4-E4B-it`
- **Student**: `gemma-3-1b-it` (270M–1B range)
- **LoRA**: r=16, alpha=16
- **Seed**: 42

## 2. Teacher Pilot
- Throughput: 6.3 tok/s
- Peak memory: 7.97 GB
- Pilot passed: False

## 3. Data Pipeline
- Raw teacher outputs: 1500
- After filtering: 1005
- Rejection breakdown:
  - near_dup: 474
  - eval_overlap: 13
  - truncation: 6
  - language: 2

## 4. Training
- max_seq_length: 1024
- P95 token length: 1012
- Coverage: 95.2%

## 5. Evaluation Summary

### Ge'ez Ratio (higher = more Amharic)
| Task | Teacher | Student |
|------|---------|---------|
| closed_book_knowledge | 0.000 | 0.972 |
| instruction_following | 0.000 | 0.918 |
| translation | 0.000 | 0.951 |
| summarization | 0.000 | 0.912 |

### Response Length (words)
| Task | Teacher | Student | Ratio (S/T) |
|------|---------|---------|-------------|
| closed_book_knowledge | 1 | 39 | 39.40 |
| instruction_following | 1 | 19 | 19.27 |
| translation | 1 | 29 | 29.47 |
| summarization | 1 | 45 | 44.60 |

## 6. Files
- Predictions: `data/predictions.csv`
- Adapter: `adapter/final/`
- Training loss plot: `logs/training_loss.png`
- Eval comparison: `logs/eval_comparison.png`
