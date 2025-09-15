# Available models

## B-Cell Receptor (BCR) Models

| Model       | Command     | Embedding Dimension | Trained on | Reference                                                                        |
| ----------- | ----------- | ------------------- | ---------- | -------------------------------------------------------------------------------- |
| AbLang      | ablang      | 768                 | H, L, H+L  | [doi.org/10.1101/2022.01.20.477061](https://doi.org/10.1101/2022.01.20.477061)   |
| AntiBERTa2  | antiberta2  | 1024                | H, L, H+L  | [doi:10.1016/j.patter.2022.100513](https://doi.org/10.1016/j.patter.2022.100513) |
| AntiBERTy   | antiberty   | 512                 | H, L, H+L  | [doi:10.48550/arXiv.2112.07782](https://doi.org/10.48550/arXiv.2112.07782)       |
| BALM-paired | balm-paired | 1024                | HL, LH     | [doi:10.1016/j.patter.2024.100967](https://doi.org/10.1016/j.patter.2024.100967) |

### AntiBERTa2

RoFormer model pre-trained on 1.54 billion unpaired and 2.9 million paired human antibody sequences (H, L, H+L chains). This model was trained on individual Heavy/Light chains separately and cannot understand paired sequences.

Reference:
Leem J, Mitchell LS, Farmery JHR, Barton J, Galson JD. Deciphering the language of antibodies using self-supervised learning. Patterns. 2022;3: 100513. [doi:10.1016/j.patter.2022.100513](https://doi.org/10.1016/j.patter.2022.100513).

### AntiBERTy

Lightweight BERT model pre-trained on 588 million Observed Antibody Space (OAS) heavy and light antibody sequences from multiple species (H, L, H+L chains). This model was trained on individual Heavy/Light chains separately and cannot understand paired sequences.

Reference:
Ruffolo JA, Gray JJ, Sulam J. Deciphering antibody affinity maturation with language models and weakly supervised learning. arXiv. 2021; 2112.07782. [doi:10.48550/arXiv.2112.07782](https://doi.org/10.48550/arXiv.2112.07782).

### AbLang

Antibody language model with separate models for heavy and light chains (H, L, H+L chains). Trained on antibody sequences in the OAS database, demonstrating power in restoring missing residues. This model was trained on individual Heavy/Light chains separately and cannot understand paired sequences.

Reference:
Olsen TH, Boyles F, Deane CM. AbLang: an antibody language model for completing antibody sequences. Bioinformatics Advances. 2022;2: vbac046. [doi:10.1093/bioadv/vbac046](https://doi.org/10.1093/bioadv/vbac046).

### BALM-paired

RoBERTa-based large model pre-trained on 1.34 million concatenated heavy and light chain human antibody sequences (HL, LH chains). This model requires paired input and cannot process individual chains. It was trained on concatenated Heavy-Light sequences and understands chain pairing structure.

Reference:
Burbach SM, Briney B. Improving antibody language models with native pairing. Patterns. 2024;5. [doi:10.1016/j.patter.2024.100967](https://doi.org/10.1016/j.patter.2024.100967).

## T-Cell Receptor (TCR) Models

| Model    | Command  | Embedding Dimension | Trained on        | Reference                                                                                    |
| -------- | -------- | ------------------- | ----------------- | -------------------------------------------------------------------------------------------- |
| TCR-BERT | tcr-bert | 768                 | H, L, HL, LH, H+L | [doi:10.1101/2021.11.18.469186](https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1) |
| TCRT5    | tcrt5    | 256                 | H only            | [doi.org/10.1101/2024.11.11.623124](https://doi.org/10.1101/2024.11.11.623124)               |

### TCR-BERT

BERT model pre-trained on 88,403 human TCR alpha and beta sequences (TRA/TRB) from VDJdb and PIRD databases (H, L, HL, LH, H+L chains). Specialized for alpha/beta T-Cell Receptor analysis. This model supports all chain formats and understands chain relationships.

Reference:
Lu T, Zhang Z, Zhu J, et al. Deep learning-based prediction of the T cell receptor–antigen binding specificity. bioRxiv. 2021. [doi:10.1101/2021.11.18.469186](https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1).

### TCRT5

T5-based model pre-trained on masked span reconstruction using ~14M CDR3 β sequences from TCRdb and ~780k peptide-pseudosequence pairs from IEDB (H chains only). This model is specialized for TCR beta chains only and supports H chain embedding exclusively. Here we included only the model trained on the TCR sequence only, not the TCR-peptide pairs.

Reference:
Deng K, Guan R, Liu Z, et al. TCRT5: T-cell receptor sequence modeling with T5. bioRxiv. 2024. [doi:10.1101/2024.11.11.623124](https://doi.org/10.1101/2024.11.11.623124).

## General Protein Models

| Model                 | Command | Embedding Dimension | Trained on       | Reference                                                                                              |
| --------------------- | ------- | ------------------- | ---------------- | ------------------------------------------------------------------------------------------------------ |
| ESM2 (650M parameter) | esm2    | 1280                | H, L, H+L, HL/LH | [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)                                 |
| ProtT5                | prott5  | 1024                | H, L, H+L, HL/LH | [doi:10.1101/2020.07.12.199554](https://doi.org/10.1101/2020.07.12.199554)                             |
| Custom models         | custom  | Configurable        | H, L, H+L, HL/LH | User-provided fine-tuned or custom models (requires --model-path, --embedding-dimension, --max-length) |

### ESM2

General protein language model pre-trained on 216 million UniRef50 protein sequences (H, L, H+L chains). Supports both BCR and TCR sequences. This is a general protein language model without paired chain mechanisms, so HL/LH paired chains will be processed but accuracy may be reduced as the model cannot understand chain pairing structure.

Reference:
Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379: 1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574).

### ProtT5

Protein language model pre-trained on UniRef50 sequences (H, L, H+L chains). Supports both BCR and TCR data with unified chain mapping (H=Heavy/Beta, L=Light/Alpha). This is a general protein language model without paired chain mechanisms, so HL/LH paired chains will be processed but accuracy may be reduced as the model cannot understand chain pairing structure.

Reference:
Elnaggar A, Heinzinger M, Dallago C, et al. ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing. bioRxiv. 2020. [doi:10.1101/2020.07.12.199554](https://doi.org/10.1101/2020.07.12.199554).

## Immune Receptor Specific Models (BCR & TCR)

| Model      | Command    | Embedding Dimension | Trained on                 | Reference                                                                                                                |
| ---------- | ---------- | ------------------- | -------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Immune2Vec | immune2vec | 100 (configurable)  | H, L, H+L + (warning)HL/LH | [doi:10.3389/fimmu.2021.680687](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.680687/full) |

### Immune2vec

Protein language model trained on immune receptor sequences (H, L, H+L chains). This is a general protein language model without paired chain mechanisms, so HL/LH paired chains will be processed but accuracy may be reduced as the model cannot understand chain pairing structure.

Reference:
Beshnova D, Ye J, Onabolu O, et al. De novo prediction of cancer-associated T cell receptors for noninvasive cancer detection. Science Translational Medicine. 2020;12: eaaz3738. [doi:10.1126/scitranslmed.aaz3738](https://doi.org/10.1126/scitranslmed.aaz3738).
