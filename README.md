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

Run MedQA samples with two Hugging Face models:
```bash
python src/run_bench.py \
  --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/BioGPT-Large \
  --dataset medqa \
  -n 20 \
  --max_workers 1
```

## Models

- [TinyLlama-1.1B-Chat-v1.0](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [BioGPT-Large](https://huggingface.co/microsoft/BioGPT-Large)

## Datasets

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


