from registry import get_model
import json
from tqdm import tqdm
from config import JUDGE_PROMPT, JUDGE_SYSTEM

def _extract_json(text: str):
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            return None
    return None

def evaluate(outputs, judge_model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens=128):
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
        parsed = _extract_json(gen)
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