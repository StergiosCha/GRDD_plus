# GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation

This repository contains the dataset and code for the paper **"GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation"** ([arXiv:2511.03772](https://arxiv.org/abs/2511.03772)).

## Abstract

We present an extended Greek Dialectal Dataset (GRDD+) that complements the existing GRDD dataset with more data from Cretan, Cypriot, Pontic and Northern Greek, while we add six new varieties: Greco-Corsican, Griko (Southern Italian Greek), Maniot, Heptanesian, Tsakonian, and Katharevusa Greek. The result is a dataset with total size 6,374,939 words and 10 varieties. This is the first dataset with such variation and size to date. We conduct a number of fine-tuning experiments to see the effect of good quality dialectal data on a number of LLMs. We fine-tune three model architectures (Llama-3-8B, Llama-3.1-8B, Krikri-8B) and compare the results to frontier models (Claude-3.7-Sonnet, Gemini-2.5, ChatGPT-5).

## Dataset

The dataset covers 10 Greek varieties. The raw text files are located in the `data/` directory:

| Variety | Filename |
| :--- | :--- |
| **Cretan** | `Cretan_final.txt` |
| **Cypriot** | `final_cypriot.txt` |
| **Pontic** | `Pontic_final.txt` |
| **Northern Greek** | `Northern_final.txt` |
| **Griko** (Southern Italian Greek) | `Griko_final.txt` |
| **Maniot** | `final_maniot.txt` |
| **Heptanesian** | `Eptanisian_final.txt` |
| **Tsakonian** | `final_tsakonian.txt` |
| **Katharevusa** | `final_katharevousa.txt` |
| **Greco-Corsican** | *(Included in dataset)* |

### Fine-tuning Data
A subset of the data used specifically for fine-tuning experiments is located in `data/fine-tuning/`.

## Repository Structure

```
GRDD_plus/
├── data/
│   ├── fine-tuning/        # Subsets for fine-tuning experiments
│   ├── Cretan_final.txt
│   ├── Eptanisian_final.txt
│   ├── ...                 # Other dialect files
├── src/
│   ├── clean.ipynb         # Data cleaning notebook
│   ├── train_llama3_8b.py  # Fine-tuning script for Llama-3-8B
│   ├── train_llama31_8b.py # Fine-tuning script for Llama-3.1-8B
│   ├── train_krikri.py     # Fine-tuning script for Krikri-8B
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## Models

The paper evaluates fine-tuning on the following architectures:
*   **Llama-3-8B**
*   **Llama-3.1-8B**
*   **Krikri-8B**

The training scripts in `src/` allow for reproducing these experiments.

## Citation

```bibtex
@misc{2511.03772,
Author = {Stergios Chatzikyriakidis},
Title = {GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation},
Year = {2025},
Eprint = {arXiv:2511.03772},
}
```
