# AMULETY

Amulety stands for Adaptive imMUne receptor Language model Embedding Tool.
It is a Python command line tool to embed B-cell receptor (antibody) and T-cell Receptor amino acid sequences using pre-trained protein or antibody language models. So far only BCR embeddings are supported but TCR support is planned for future releases. The package also has functionality to translate nucleotide sequences to amino acids with IgBlast.

Here is the list of currently supported embeddings:

| Model                 | Command     | Embedding Dimension | Reference                                                                        |
| --------------------- | ----------- | ------------------- | -------------------------------------------------------------------------------- |
| AntiBERTa2            | antiberta2  | 1024                | [doi:10.1016/j.patter.2022.100513](https://doi.org/10.1016/j.patter.2022.100513) |
| AntiBERTy             | antiberty   | 512                 | [doi:10.48550/arXiv.2112.07782](https://doi.org/10.48550/arXiv.2112.07782)       |
| BALM-paired           | balm_paired | 1024                | [doi:10.1016/j.patter.2024.100967](https://doi.org/10.1016/j.patter.2024.100967) |
| ESM2 (650M parameter) | esm2        | 1280                | [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)           |
| User-specified model  | custommodel | Configurable        |                                                                                  |

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

For help and questions please contact the [Immcantation Group](mailto:immcantation@googlegroups.com).

## Authors

[Mamie Wang](https://github.com/mamie) (aut,cre)
[Gisela Gabernet](https://github.com/ggabernet) (aut,cre)
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
