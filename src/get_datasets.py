from datasets import load_dataset
from pathlib import Path

def main():
    Path("data/cache").mkdir(parents=True, exist_ok=True)
    print("[+] Downloading MedQA (subset)…")
    _ = load_dataset("GBaker/MedQA-USMLE-4-options", cache_dir="data/cache")
    print("[+] Downloading PubMedQA…")
    _ = load_dataset("qiaojin/PubMedQA", "pqa_labeled", cache_dir="data/cache")
    print("[✓] Datasets cached under data/cache")

if __name__ == "__main__":
    main()
