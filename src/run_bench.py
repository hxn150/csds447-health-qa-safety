import argparse, json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from registry import get_model
from judge import evaluate as evaluate_with_judge
from config import MODEL, SYSTEM_PROMPT, USER_PROMPT

YES_NO_MAYBE = {"yes", "no", "maybe"}


def get_medqa_data(split="test", limit=None):
    dataset = load_dataset("GBaker/MedQA-USMLE-4-options", cache_dir="data/cache")[split]
    for index, row in enumerate(dataset):
        if limit and index >= limit: break
        yield row["question"], row.get("answer","")

def get_pubmedqa_data(split="train", limit=None):
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled", cache_dir="data/cache")[split]
    for index, row in enumerate(dataset):
        if limit and index >= limit: break
        ground_truth = row.get("final_decision") or row.get("long_answer","")
        yield row["question"], ground_truth

def _pubmedqa_prompt(question: str) -> str:
    constraint = (
        "Answer with one word only: yes, no, or maybe. "
        "Do not add punctuation or explanation."
    )
    return f"{SYSTEM_PROMPT}\n\n{USER_PROMPT.format(question=question)}\n{constraint}"


def _extract_pubmedqa_label(text: str) -> str:
    import re
    if not isinstance(text, str):
        return ""
    for m in re.finditer(r"\b(yes|no|maybe)\b", text.lower()):
        return m.group(1)
    # fallback: simple contains check
    lower = text.lower()
    if "yes" in lower: return "yes"
    if "no" in lower: return "no"
    if "maybe" in lower: return "maybe"
    return ""


def generate_model_outputs(model_name: str, qa_list, dataset_name: str = "medqa"):
    if model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0":
        model = get_model(model_name, lora_path=f"checkpoints/lora-tinyllama-{dataset_name}")
    elif model_name == "microsoft/BioGPT-Large":
        model = get_model(model_name, lora_path=f"checkpoints/lora-microsoftbiogpt-{dataset_name}")
    else:
        model = get_model(model_name)
    output = []
    for question, ground_truth in tqdm(qa_list, desc=f"{model_name}"):
        if dataset_name == "pubmedqa":
            prompt = _pubmedqa_prompt(question)
        else:
            prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT.format(question=question)}"
        gen = model.generate(prompt, max_new_tokens=128, temperature=0.0)
        print("Full generated text:", gen)
        answer = gen.split("Answer:")[-1].strip() if "Answer:" in gen else gen.strip()
        print("Extracted answer:", answer)
        m = re.search(r"\nQuestion:\s", answer)
        if m:
            answer = answer[:m.start()].strip()
        pred_value = answer
        record = {"model": model_name, "question": question, "prediction": pred_value, "ground_truth": ground_truth}
        if dataset_name == "pubmedqa":
            inferred = _extract_pubmedqa_label(answer)
            # Store raw for traceability
            record["raw_prediction"] = answer
            if inferred in YES_NO_MAYBE:
                record["prediction"] = inferred
            else:
                # Deterministic fallback: classify among yes/no/maybe using model log-likelihoods
                try:
                    label = model.classify_options(_pubmedqa_prompt(question), ["yes", "no", "maybe"])
                    if label in YES_NO_MAYBE:
                        record["prediction"] = label
                except Exception as e:
                    # Keep existing prediction if classification fails
                    pass
        output.append(record)
    return output

def main():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--models", nargs="+", required=True,
                    help="List of models: TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/BioGPT-Large, etc")
    argument_parser.add_argument("--dataset", choices=["medqa","pubmedqa"], default="medqa")
    argument_parser.add_argument("--split", default="test")
    argument_parser.add_argument("-n", type=int, default=50, help="limit per dataset")
    argument_parser.add_argument("--out_dir", default="outputs")
    argument_parser.add_argument("--max_workers", type=int, default=1, help="parallel models (threads) that run at the same time")
    argument_parser.add_argument("--judge_model", default=None,
                    help="Optional model name used as the judge. If omitted, no judging is done.")
    args = argument_parser.parse_args()

    Path(args.out_dir).mkdir(exist_ok=True)

    qa_list = list(get_medqa_data(args.split, args.n)) if args.dataset=="medqa" \
              else list(get_pubmedqa_data(args.split, args.n))

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(generate_model_outputs, model, qa_list, args.dataset): model for model in args.models}
        for fut in as_completed(futs):
            model = futs[fut]
            output = fut.result()
            if args.judge_model:
                output = evaluate_with_judge(output, judge_model_name=args.judge_model)
            dataframe = pd.DataFrame(output)
            per_path = Path(args.out_dir)/f"pred_{args.dataset}_{model.replace('/','_').replace(':','_')}.csv"
            dataframe.to_csv(per_path, index=False)
            print(f"[✓] Saved {model} predictions → {per_path}")
            results.append(dataframe)

    combined_results = pd.concat(results, ignore_index=True)
    combined_path = Path(args.out_dir)/f"pred_{args.dataset}_combined.csv"
    combined_results.to_csv(combined_path, index=False)
    print(f"[✓] Combined predictions → {combined_path}")

if __name__ == "__main__":
    main()
