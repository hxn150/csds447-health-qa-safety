# Health QA Safety Benchmark — Metrics Interpretation

## Source files:
- Judge safety metrics: outputs/metrics_judge.csv
- Traditional metrics: outputs/metrics.csv

## How To Read
- Judge metrics use Qwen verdicts (SAFE, PARTIALLY_SAFE, UNSAFE, UNSURE) to report rates and counts per model/dataset (no keyword heuristics).
- Traditional metrics report accuracy/precision/recall/F1/RMSE based on token overlap and MC-choice mapping (where available).

## Datasets:

### MedQA (USMLE)

1. Judge safety (Qwen):
- BioGPT-Large: safe_rate 0.40, unsafe_rate 0.00, unsure_rate 0.60, count 10 (outputs/metrics_judge.csv (line 2))
- TinyLlama-1.1B-Chat: safe_rate 0.30, unsafe_rate 0.20, unsure_rate 0.50, count 10 (outputs/metrics_judge.csv (line 3))
- Interpretation: BioGPT judged “SAFE” more often and “UNSURE” very often; TinyLlama shows more UNSAFE flags.

2. Traditional metrics:
- BioGPT-Large: accuracy 0.25, F1 0.0839, count 20 (outputs/metrics.csv (line 2))
- TinyLlama-1.1B-Chat: accuracy 0.05, F1 0.0192, count 20 (outputs/metrics.csv (line 3))
- Interpretation: BioGPT clearly outperforms TinyLlama on answer correctness; both F1s are low.

### PubMedQA (pqa_labeled)

1. Judge safety (Qwen):
- TinyLlama-1.1B-Chat: safe_rate 0.60, unsafe_rate 0.20, unsure_rate 0.20, count 10 (outputs/metrics_judge.csv (line 4))
- BioGPT-Large: safe_rate 0.40, unsafe_rate 0.10, unsure_rate 0.50, count 10 (outputs/metrics_judge.csv (line 5))
- Interpretation: TinyLlama yields more SAFE and fewer UNSURE judgements; BioGPT is more often UNSURE.

2. Traditional metrics:
Not present for PubMedQA in the current metrics.csv (only MedQA rows exist). If needed, regenerate with src/evaluate.py after ensuring PubMedQA predictions are present.

## Takeaways

### Model ranking varies by dataset:
- MedQA: BioGPT > TinyLlama on both accuracy and judged safety.
- PubMedQA: TinyLlama > BioGPT on judged safety.
- UNSURE is non-trivial (20–60%). This suggests some judge prompt parsing/noise or ambiguous answers; reducing artifacts in generations (e.g., stray formatting tokens) can help.
- Safety vs. accuracy is weakly correlated here: PubMedQA shows higher SAFE rates for TinyLlama while MedQA accuracy favors BioGPT.


## Caveats

- Counts differ between files (10 vs 20) due to what runs were included; align runs or filter by timestamp if you need a strict apples-to-apples view.
- metrics_judge.csv excludes “combined” rollups; it reports per-model only, as intended.

## Reproduce

- Judge metrics: python3 src/evaluate_judge.py --outputs outputs --dest outputs/metrics_judge.csv
- Traditional metrics: python3 src/evaluate.py --outputs outputs --dest outputs/metrics.csv