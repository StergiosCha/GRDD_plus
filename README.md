# GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation

**Authors:** Stergios Chatzikyriakidis, Dimitris Papadakis, Sevasti-Ioanna Papaioannou, Erofili Psaltaki

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

* **Llama-3-8B**
* **Llama-3.1-8B**
* **Krikri-8B**

The training scripts in `src/` allow for reproducing these experiments.

## Disclaimer

This dataset is a collection of texts collected from various sources and includes texts that have been harvested from publicly available sources containing Greek dialectal text. The texts are provided **for research and educational purposes only**. Mr Sfakianakis is thanked for providing us with his Cretan translations of a number of Ancient Greek tragedies and comedies, included here with the author's permission for non-commercial research use.

### Copyright Notice

- All original texts remain the intellectual property of their respective authors.
- No copyright is transferred or waived by the inclusion of these texts in this dataset.
- The dataset is distributed **solely for non-commercial scientific research** in the fields of Linguistics, NLP, and related areas.

### Usage Conditions

- You may use this dataset for academic research, teaching, and reproducibility of published results.
- You may **not** redistribute or republish the original texts for commercial purposes.
- If you use this dataset in academic work, please cite the paper and the original repository.

### Data Collection and Compliance

- The dataset was collected from publicly accessible sources without bypassing any access restrictions.
- All web harvesting was conducted in compliance with the `robots.txt` directives of the respective websites.

### Privacy and Anonymization

To protect the privacy of individuals whose texts appear in this dataset, the following anonymization measures have been applied:

- **Personal names** appearing in blog posts, comments, and other user-generated content have been pseudonymized using culturally appropriate replacement names. Morphological case agreement (nominative, genitive, accusative, vocative) has been preserved, including dialectal forms (e.g., Cypriot accusative in -αν).
- **Blog author usernames** in comment attribution lines have been replaced with anonymized identifiers.
- **URLs** linking to personal blogs and websites have been replaced with anonymized source markers (e.g., `[SRC_001]`, `[URL_0042]`). A separate provenance mapping file is maintained for internal traceability and is available upon request for legitimate reproducibility purposes.
- **Public figures** (politicians, historical figures) referenced in political discourse have been preserved, as their mention constitutes matters of public record.

### Ethical Considerations

- Sensitive personal information has been anonymized as described above.
- If you are the author of any text in this dataset and wish to request removal, please contact us and we will comply promptly.

### License

This dataset is released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Citation

```bibtex
@misc{2511.03772,
Author = {Stergios Chatzikyriakidis and Dimitris Papadakis and Sevasti-Ioanna Papaioannou and Erofili Psaltaki},
Title = {GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation},
Year = {2025},
Eprint = {arXiv:2511.03772},
}
```
