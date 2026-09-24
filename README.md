# GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation

**Authors:** Stergios Chatzikyriakidis, Dimitris Papadakis, Sevasti-Ioanna Papaioannou, Erofili Psaltaki

This repository contains the dataset and code for the paper **"GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation"** ([arXiv:2511.03772](https://arxiv.org/abs/2511.03772)).

## Abstract

We present an extended Greek Dialectal Dataset (GRDD+) that complements the existing GRDD dataset with more data from Cretan, Cypriot, Pontic and Northern Greek, while we add six new varieties: Greco-Corsican, Griko (Southern Italian Greek), Maniot, Heptanesian, Tsakonian, and Katharevusa Greek. The result is a dataset with total size 6,374,939 words and 10 varieties. We conduct a number of fine-tuning experiments to see the effect of good quality dialectal data on a number of LLMs. We fine-tune three model architectures (Llama-3-8B, Llama-3.1-8B, Krikri-8B) and compare the results to frontier models (Claude-3.7-Sonnet, Gemini-2.5, ChatGPT-5).

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

A subset of the data used specifically for fine-tuning experiments is located in `data/fine-tuning/`. It is drawn from the files above and is subject to the same terms.

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

## Sources and Rights

The texts in this dataset come from three kinds of sources, each with a different rights status.

**1. Public-domain texts.** Older literary and traditional texts whose authors died long ago, such as Vitsentzos Kornaros's *Erotokritos* (17th century). The works themselves are in the public domain. Where a text was taken from a modern edition, the edition is acknowledged, and any rights in the editorial work remain with the editor.

**2. Texts included with permission.** Cretan translations of Ancient Greek tragedies and comedies by Mr Sfakianakis, whom we thank. They are included with the author's permission for non-commercial research use only and remain his intellectual property.

**3. Texts harvested from the web.** The Cypriot data include texts from publicly accessible blogs, forums and websites. These texts remain the intellectual property of their authors. They were collected and are made available solely for scientific research, under the text and data mining exception for research purposes (Article 3 of Directive (EU) 2019/790, as transposed into Greek law 2121/1993 by Law 4996/2022). They have been pseudonymized as described below.

No copyright in any text is transferred, claimed or waived by the maintainers through its inclusion in this dataset.

## Data Collection

- All texts were collected from publicly accessible sources, without bypassing logins, paywalls or any other access restrictions. Private or restricted blogs were not collected.
- All web harvesting complied with the `robots.txt` directives of the respective websites.

## Privacy and Pseudonymization

To protect the privacy of individuals whose texts appear in the web-harvested data, the following measures have been applied:

- **Personal names** appearing in blog posts, comments and other user-generated content have been replaced with culturally appropriate pseudonyms. Morphological case agreement (nominative, genitive, accusative, vocative) has been preserved, including dialectal forms (e.g. the Cypriot accusative in -αν).
- **Blog author usernames** in comment attribution lines have been replaced with anonymized identifiers.
- **URLs** linking to personal blogs and websites have been replaced with anonymized source markers (e.g. `[SRC_001]`, `[URL_0042]`).
- **Public figures** (politicians, historical figures) referenced in public discourse have been kept, as their mention concerns matters of public record.

The provenance mapping between source markers and original URLs is **not publicly released**. It is kept in secure storage by the maintainers for verification of published results and for handling removal requests. Access may be granted only to researchers, for verification purposes, under a written agreement that prohibits redistribution and re-identification.

## Terms of Use

By using this dataset you agree to the following:

- The dataset may be used only for **non-commercial** academic research, teaching and the reproduction of published results in linguistics, NLP and related fields.
- The original texts may not be republished or redistributed for commercial purposes, and may not be used to train or improve models offered commercially.
- You may not attempt to re-identify any person whose text or name appears in the dataset, or to link pseudonymized texts back to their authors or original sources.
- Quotations from the texts in publications should be short and limited to what the research requires.
- If you use this dataset in academic work, please cite the papers listed below and acknowledge the original authors of the texts.

## Removal Requests

If you are the author of a text included in this dataset, or a person named in it, and you want it removed or corrected, please write to **kafouroutsos@gmail.com**. We will remove or correct the material in the next update of the repository, and in any case within 30 days of the request. No justification is required.

## License

- **Annotations, metadata, cleaning code and training scripts** are released under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.
- **Public-domain texts** (category 1) carry no copyright restrictions on the works themselves.
- **All other texts** (categories 2 and 3) are not licensed by the maintainers. They remain the property of their authors and are provided only for non-commercial research under the terms above.

## Disclaimer

The dataset is provided "as is", without warranty of any kind. The maintainers make no claim of ownership over texts they did not author and accept no liability for uses of the dataset that breach these terms.

## Citation

If you use GRDD+, please cite:

```bibtex
@misc{chatzikyriakidis2025grddplus,
  author = {Stergios Chatzikyriakidis and Dimitris Papadakis and Sevasti-Ioanna Papaioannou and Erofili Psaltaki},
  title  = {GRDD+: An Extended Greek Dialectal Dataset with Cross-Architecture Fine-tuning Evaluation},
  year   = {2025},
  eprint = {2511.03772},
  archivePrefix = {arXiv}
}
```

and the original GRDD dataset:

```bibtex
@misc{chatzikyriakidis2023grdd,
  author = {Stergios Chatzikyriakidis and Chatrine Qwaider and Ilias Kolokousis and Christina Koula and Dimitris Papadakis and Efthymia Sakellariou},
  title  = {GRDD: A Dataset for Greek Dialectal NLP},
  year   = {2023},
  eprint = {2308.00802},
  archivePrefix = {arXiv}
}
```
