# csds447-health-qa-safety

setup and install env/reqs
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# MedQA samples with two HF models
python src/run_bench.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0 microsoft/BioGPT-Large --dataset medqa -n 20 --max_workers 1

# Models used:
tiny llama 1
bioGPT

# datasets:

PubMedQA: https://huggingface.co/datasets/qiaojin/PubMedQA

# dataset paper:
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


