# AMULETY

Amulety stands for Adaptive imMUne receptor Language model Embedding Tool.
It is a Python command line tool to embed B-cell receptor (antibody) and T-cell receptor (TCR) amino acid sequences using pre-trained protein or antibody language models. The package supports both BCR and TCR embeddings, with specialized TCR processing for alpha-beta chain concatenation. The package also has functionality to translate nucleotide sequences to amino acids with IgBlast.

Here is the list of currently supported embeddings:

## BCR (B-Cell Receptor) Models

| Model       | Command     | Embedding Dimension | Reference                                                                        |
| ----------- | ----------- | ------------------- | -------------------------------------------------------------------------------- |
| AntiBERTa2  | antiberta2  | 1024                | [doi:10.1016/j.patter.2022.100513](https://doi.org/10.1016/j.patter.2022.100513) |
| AntiBERTy   | antiberty   | 512                 | [doi:10.48550/arXiv.2112.07782](https://doi.org/10.48550/arXiv.2112.07782)       |
| BALM-paired | balm-paired | 1024                | [doi:10.1016/j.patter.2024.100967](https://doi.org/10.1016/j.patter.2024.100967) |

## TCR (T-Cell Receptor) Models

| Model    | Command  | Embedding Dimension | TCR Type Support | Reference                                                                                            |
| -------- | -------- | ------------------- | ---------------- | ---------------------------------------------------------------------------------------------------- |
| TCR-BERT | tcr-bert | 768                 | α/β only         | [doi:10.1101/2021.11.18.469186](https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1)         |
| DeepTCR  | deep-tcr | 256                 | mainly α/β       | [doi:10.1038/s41467-021-21879-w](https://www.nature.com/articles/s41467-021-21879-w)                 |
| Trex     | trex     | 768                 | α/β only         | [PMID:39164479](https://pubmed.ncbi.nlm.nih.gov/39164479/)                                           |
| TCREMP   | tcremp   | 512                 | mainly α/β       | [doi:10.1016/j.jmb.2025.168712](https://www.sciencedirect.com/science/article/pii/S0022283625002712) |

## General Protein Models (BCR & TCR)

| Model                 | Command | Embedding Dimension | TCR Type Support | Reference                                                                  |
| --------------------- | ------- | ------------------- | ---------------- | -------------------------------------------------------------------------- |
| ESM2 (650M parameter) | esm2    | 1280                | α/β + γ/δ        | [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)     |
| ProtT5                | prott5  | 1024                | α/β + γ/δ        | [doi:10.1101/2020.07.12.199554](https://doi.org/10.1101/2020.07.12.199554) |
| User-specified model  | custom  | Configurable        | depends on model | Custom model support                                                       |

## Chain Parameter Interface

AMULETY uses a unified chain parameter interface for both BCR and TCR sequences:

- **For BCR**:

  - `H` = Heavy chain
  - `L` = Light chain
  - `HL` = Heavy-Light chain pairs

- **For TCR**:
  - `H` = Beta/Delta chains (TRB/TRD)
  - `L` = Alpha/Gamma chains (TRA/TRG)
  - `HL` = Beta-Alpha/Delta-Gamma chain pairs

This unified interface supports both alpha/beta and gamma/delta TCR types when the embedding models allow it.

## Installation

You can install AMULETY using pip:

```bash
pip install amulety
```

## Usage

To print the usage help for the AMULETY package then type:

```bash
amulety --help
```

The full usage documentation can also be found on the readthedocs [usage page](https://amulety.readthedocs.io/en/latest/usage.html).

## Contact

If you need help or have any questions, please contact the [Immcantation Group](mailto:immcantation@googlegroups.com).

If you have discovered a bug or have a feature request, you can open an issue using the [issue tracker](https://github.com/immcantation/amulety/issues).

To receive alerts about Immcantation releases, news, events, and tutorials, join the [Immcantation News](https://groups.google.com/g/immcantation-news) Google Group. [Membership settings](https://groups.google.com/g/immcantation-news/membership) can be adjusted to change the frequency of email updates.

## Authors

[Mamie Wang](https://github.com/mamie) (aut,cre)
[Gisela Gabernet](https://github.com/ggabernet) (aut,cre)
[Wengyao Jiang](https://github.com/wenggyaoo) (cont)
[Steven Kleinstein](mailto:steven.kleinstein@yale.edu) (aut,cph)

## Citing

If you use this package, please cite the pre-print:

> AMULETY: A Python package to embed adaptive immune receptor sequences.
> Meng Wang, Yuval Kluger, Steven H. Kleinstein, Gisela Gabernet.
> BioRXiv 2025. DOI: [https://doi.org/10.1101/2025.03.21.644583](https://doi.org/10.1101/2025.03.21.644583)

To cite the paper comparing the embedding methods on BCR sequences, please cite:

> Supervised fine-tuning of pre-trained antibody language models improves antigen specificity prediction.
> Meng Wang, Jonathan Patsenker, Henry Li, Yuval Kluger, Steven H. Kleinstein.
> BioRXiv 2024. DOI: [https://doi.org/10.1101/2024.05.13.593807](https://doi.org/10.1101/2024.05.13.593807).

## License

This project is licensed under the terms of the GPL v3 license. See the LICENSE file for details.
