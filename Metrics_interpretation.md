# Health QA Safety Benchmark — Metrics Interpretation

## Sources
- Traditional metrics: `outputs/metrics.csv`
- Qwen‑judge metrics: `outputs/metrics_judge.csv`

## Reading Guide
- Traditional metrics: exact/overlap scoring of `prediction` vs `ground_truth` (MedQA uses MC mapping; PubMedQA expects labels yes/no/maybe).
- Judge metrics: distribution of Qwen verdicts (SAFE, PARTIALLY_SAFE, UNSAFE, UNSURE). No keyword heuristics involved.

## MedQA (USMLE)

- Traditional (count=10)
  - microsoft/BioGPT-Large: accuracy 0.30, F1 0.0633
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0: accuracy 0.20, F1 0.0260
  - Read: BioGPT leads on correctness; both models have low lexical overlap (low F1).

- Judge (count=10)
  - microsoft/BioGPT-Large: SAFE 0.40, PARTIALLY 0.00, UNSAFE 0.00, UNSURE 0.60
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0: SAFE 0.30, PARTIALLY 0.00, UNSAFE 0.20, UNSURE 0.50
  - Read: BioGPT has fewer unsafe calls; both show many UNSURE judgments.

## PubMedQA (pqa_labeled, train)

- Traditional (count=10)
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0: accuracy 0.30, F1 0.30
  - microsoft/BioGPT-Large: accuracy 0.10, F1 0.10
  - Read: Constrained prompt + deterministic fallback now yields valid labels for both; TinyLlama performs better on label accuracy.

- Judge (count=10)
  - microsoft/BioGPT-Large: SAFE 0.50, PARTIALLY 0.20, UNSAFE 0.00, UNSURE 0.30 (SAFE+PARTIALLY 0.70)
  - TinyLlama/TinyLlama-1.1B-Chat-v1.0: SAFE 0.40, PARTIALLY 0.00, UNSAFE 0.30, UNSURE 0.30
  - Read: BioGPT appears safer per Qwen on this slice despite lower accuracy; TinyLlama has more UNSAFE flags.

## Takeaways
- Model ranking depends on dataset: BioGPT > TinyLlama on MedQA accuracy; TinyLlama > BioGPT on PubMedQA accuracy.
- Safety vs. accuracy diverge: judged safety does not guarantee correct labels (and vice versa).
- High UNSURE rates (30–60%) indicate ambiguous or low‑confidence answers; prompt/decoding tweaks helped formatting but not all correctness gaps.

## Caveats
- Small n (10) → expect variance. Scale to 50–100 for stable comparisons.
- Only per‑model CSVs are included (combined rollups are ignored in evaluators).

## Reproduce
- Traditional: `python3 src/evaluate.py --outputs outputs --dest outputs/metrics.csv`
- Judge: `python3 src/evaluate_judge.py --outputs outputs --dest outputs/metrics_judge.csv`
