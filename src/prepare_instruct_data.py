import json
from datasets import load_dataset
from pathlib import Path
from config import SYSTEM_PROMPT, USER_PROMPT

def make_example(question, answer, tags=None, source="medqa"):
    return {
        "instruction": SYSTEM_PROMPT,
        "input": USER_PROMPT.format(question=question),
        "output": answer,
        "tags": tags or [],
        "source": source
    }

def medqa_to_instruct(split="train", limit=None):
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", cache_dir="data/cache")[split]
    for i, row in enumerate(ds):
        if limit and i >= limit: break
        yield make_example(row["question"], row.get("answer", ""), source="medqa")

def pubmedqa_to_instruct(split="train", limit=None):
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", cache_dir="data/cache")[split]
    for i, row in enumerate(ds):
        if limit and i >= limit: break
        answer = row.get("final_decision", "") or row.get("long_answer", "")
        yield make_example(row["question"], answer, source="pubmedqa")

def main():
    out_dir = Path("data/instruct")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "medqa_train.jsonl", "w", encoding="utf-8") as f:
        for ex in medqa_to_instruct("train"):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out_dir / "medqa_test.jsonl", "w", encoding="utf-8") as f:
        for ex in medqa_to_instruct("test"):
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    pubmedqa_rows = list(pubmedqa_to_instruct("train"))
    with open(out_dir / "pubmedqa_train.jsonl", "w", encoding="utf-8") as f:
        for ex in pubmedqa_rows[:900]:  
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    with open(out_dir / "pubmedqa_test.jsonl", "w", encoding="utf-8") as f:
        for ex in pubmedqa_rows[900:]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print("Instruction datasets written to", out_dir)

if __name__ == "__main__":
    main()
