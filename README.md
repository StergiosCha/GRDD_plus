# GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation

This repository contains the code and resources for the paper **"GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation"**.

## Abstract

We present an extended Greek Dialectal Dataset (GRDD+) that complements the existing GRDD dataset with more data from Cretan, Cypriot, Pontic and Northern Greek, while we add six new varieties: Greco-Corsican, Griko (Southern Italian Greek), Maniot, Heptanesian, Tsakonian, and Katharevusa Greek. The result is a dataset with total size 6,374,939 words and 10 varieties. This is the first dataset with such variation and size to date. We conduct a number of fine-tuning experiments to see the effect of good quality dialectal data on a number of LLMs. We fine-tune three model architectures (Llama-3-8B, Llama-3.1-8B, Krikri-8B) and compare the results to frontier models (Claude-3.7-Sonnet, Gemini-2.5, ChatGPT-5).

## Repository Structure

```
GRDD-Plus/
├── data/               # Data files
│   └── fine-tuning/    # Fine-tuning datasets
├── src/                # Source code
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

## Setup

1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Preparation
(Add instructions here)

### Training
(Add instructions here)

### Evaluation
(Add instructions here)

## Citation
(Add citation here)
