# csds447-health-qa-safety

Health QA Safety Benchmark for Medical Language Models

## Setup and Installation

### Virtual environment
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Running the Benchmark

Always include the Qwen/Qwen3-0.6B judge so every run collects safety verdicts.

**MedQA (USMLE)**
- TinyLlama/TinyLlama-1.1B-Chat-v1.0  
  ```bash
  python3 src/run_bench.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset medqa -n 20 --max_workers 2 --judge_model Qwen/Qwen3-0.6B
  ```
- microsoft/BioGPT-Large  
  ```bash
  python3 src/run_bench.py --models microsoft/BioGPT-Large --dataset medqa -n 20 --max_workers 2 --judge_model Qwen/Qwen3-0.6B
  ```

**PubMedQA (pqa_labeled, train split)**
- TinyLlama/TinyLlama-1.1B-Chat-v1.0  
  ```bash
  python3 src/run_bench.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 --dataset pubmedqa --split train -n 20 --max_workers 2 --judge_model Qwen/Qwen3-0.6B
  ```
- microsoft/BioGPT-Large  
  ```bash
  python3 src/run_bench.py --models microsoft/BioGPT-Large --dataset pubmedqa --split train -n 20 --max_workers 2 --judge_model Qwen/Qwen3-0.6B
  ```

You can benchmark several models at once by passing a space-separated list to `--models`.

### Evaluating Model Outputs

Summarise accuracy and safety rates (including the judge verdicts) with:
```bash
python3 src/evaluate.py --outputs outputs --dest outputs/metrics_qwen.csv
```

### Fine-tuning with LoRA
Prepare data: python src/prepare_instruct_data.py
Train: python src/train_lora.py --model model_name --dataset dataset_name

## Models

- [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [BioGPT-Large](https://huggingface.co/microsoft/BioGPT-Large)
- [LLaMA 3](https://huggingface.co/meta-llama/Meta-Llama-3-8B) (to be added)
- [Qwen](https://huggingface.co/Qwen/Qwen3-VL-4B-Thinking) (judge model)

## Datasets

- **MedQA**: [https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)
- **PubMedQA**: [https://huggingface.co/datasets/qiaojin/PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA)

## Citations

If you use this benchmark, please cite the following papers:
```bibtex
@article{jin2020disease,
  title={What Disease does this Patient Have? A Large-scale Open Domain Question Answering Dataset from Medical Exams},
  author={Jin, Di and Pan, Eileen and Oufattole, Nassim and Weng, Wei-Hung and Fang, Hanyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:2009.13081},
  year={2020}
}

@inproceedings{jin2019pubmedqa,
  title={PubMedQA: A Dataset for Biomedical Research Question Answering},
  author={Jin, Qiao and Dhingra, Bhuwan and Liu, Zhengping and Cohen, William and Lu, Xinghua},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={2567--2577},
  year={2019}
}
```
