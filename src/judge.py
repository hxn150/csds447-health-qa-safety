from registry import get_model
import json
from tqdm import tqdm
from config import JUDGE_PROMPT, JUDGE_SYSTEM

def extract_json(text: str):
    start = text.find("{")
    while start != -1:
        end = text.find("}", start + 1)
        while end != -1:
            candidate = text[start:end+1]
            try:
                return json.loads(candidate)
            except Exception:
                end = text.find("}", end + 1)
        start = text.find("{", start + 1)
    return None

def evaluate(outputs, judge_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=256):
    judge = get_model(judge_model_name)
    judged = []
    for entry in tqdm(outputs, desc="judge"):
        prompt = JUDGE_SYSTEM + "\n\n" + JUDGE_PROMPT.format(
            question=entry.get("question",""),
            prediction=entry.get("prediction",""),
            ground_truth=entry.get("ground_truth","")
        )
        gen = judge.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1
        )
        print("Full judge output:", gen)
        parsed = extract_json(gen)
        print("Parsed judge output:", parsed)
        if parsed is None:
            entry.update({"verdict":"UNSURE", "reason": gen.strip().replace("\n"," "), "tags":[], "_raw_judge": gen})
        else:
            entry.update({
                "verdict": parsed.get("verdict","UNSURE"),
                "reason": parsed.get("reason",""),
                "tags": parsed.get("tags",[])
            })
        judged.append(entry)
    return judged