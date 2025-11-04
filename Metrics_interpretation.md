# Health QA Safety Benchmark — Metrics Interpretation

## Sources
- Traditional metrics: `outputs/metrics.csv`
- LLM-as-a‑judge (Qwen) metrics: `outputs/metrics_judge.csv`

## Reading Guide
- Traditional metrics: exact/overlap scoring of `prediction` vs `ground_truth` (MedQA uses MC mapping; PubMedQA expects labels yes/no/maybe).
- LLM-as-a-judge metrics: distribution of Qwen verdicts (SAFE, PARTIALLY_SAFE, UNSAFE, UNSURE). No keyword heuristics involved.

## MedQA (USMLE)

- Traditional (count=10)
  - microsoft/BioGPT-Large: accuracy 0.30, F1 0.0633
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0: accuracy 0.20, F1 0.0260
  - Read: BioGPT leads on correctness; both models have low lexical overlap (low F1).

- LLM-as-a-judge (count=10)
  - microsoft/BioGPT-Large: SAFE 1.00, PARTIALLY 0.00, UNSAFE 0.00, UNSURE 0.00
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0: SAFE 0.70, PARTIALLY 0.10, UNSAFE 0.20, UNSURE 0.00
  - Read: Judge confidence is high (UNSURE 0). BioGPT responses are all judged SAFE; TinyLlama has some UNSAFE cases.

## PubMedQA (pqa_labeled, train)

- Traditional (count=10)
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0: accuracy 0.30, F1 0.30
  - microsoft/BioGPT-Large: accuracy 0.10, F1 0.10
  - Read: Constrained prompt + deterministic fallback now yields valid labels for both; TinyLlama performs better on label accuracy.

- LLM-as-a-judge (count=10)
  - microsoft/BioGPT-Large: SAFE 0.70, PARTIALLY 0.20, UNSAFE 0.00, UNSURE 0.10 (SAFE+PARTIALLY 0.90)
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0: SAFE 0.70, PARTIALLY 0.00, UNSAFE 0.30, UNSURE 0.00
  - Read: Both appear largely SAFE; BioGPT has more PARTIALLY_SAFE, TinyLlama shows higher UNSAFE.

## Takeaways
- Model ranking depends on dataset: BioGPT > TinyLlama on MedQA accuracy; TinyLlama > BioGPT on PubMedQA accuracy.
- Safety vs. accuracy diverge: judged safety does not guarantee correct labels (and vice versa).
- UNSURE is now minimal/zero on this slice; judge outputs indicate high confidence in safety assessments.
- Future improvements: increase n for stability; consider few-shot or label-scoring for PubMedQA to further improve correctness.

## Caveats
- Small n (10) → expect variance. Scale to 50–100 for stable comparisons.
- Only per‑model CSVs are included (combined rollups are ignored in evaluators).

## Reproduce
- Traditional: `python3 src/evaluate.py --outputs outputs --dest outputs/metrics.csv`
- LLM-as-a-judge: `python3 src/evaluate_judge.py --outputs outputs --dest outputs/metrics_judge.csv`
