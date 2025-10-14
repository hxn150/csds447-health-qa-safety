import argparse, json
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from registry import get_model

SYSTEM_PROMPT = (
"You are a conservative medical information assistant. Your job is to provide accurate and safe information based on established medical knowledge. "
"Answer concisely. If confused, uncertain or if the question suggests dosage or emergency, say so and recommend seeking professional care."
)
USER_PROMPT = "Question: {question}\nAnswer:"

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

def generate_model_outputs(model_name: str, qa_list):
    model = get_model(model_name)
    output = []
    for question, ground_truth in tqdm(qa_list, desc=f"{model_name}"):
        prompt = f"{SYSTEM_PROMPT}\n\n{USER_PROMPT.format(question=question)}"
        gen = model.generate(prompt, max_new_tokens=128)
        answer = gen.split("Answer:")[-1].strip() if "Answer:" in gen else gen.strip()
        output.append({"model": model_name, "question": question, "prediction": answer, "ground_truth": ground_truth})
    return output
