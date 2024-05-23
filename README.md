# AMULETY

Amulety stands for Adaptive imMUne receptor Language model Embedding Tool.
It is a Python command line tool to embed B-cell receptor (antibody) and T-cell Receptor amino acid sequences using pre-trained protein or antibody language models. So far only BCR embeddings are supported but TCR support is planned for future releases. The package also has functionality to translate nucleotide sequences to amino acids wiht IgBlast to make sure that they are in-frame.

Integrated embedding models are:

- antiBERTy
- antiBERTa2
- ESM2
- Custom models

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

The full documentation can also be found on the readthedocs page.

## Contact

For help and questions please contact the Immcantation Group.

## Authors

[Mamie Wang](https://github.com/mamie) (aut,cre)
[Gisela Gabernet](https://github.com/ggabernet) (aut,cre)
[Steven Kleinstein](mailto:steven.kleinstein@yale.edu) (aut,cph)

## Citing

This package is not yet published.

To cite the paper comparing the embedding methods on BCR sequences, please cite:

> Supervised fine-tuning of pre-trained antibody language models improves antigen specificity prediction.
> Meng Wang, Jonathan Patsenker, Henry Li, Yuval Kluger, Steven H. Kleinstein.
> BioRXiv 2024. DOI: [https://doi.org/10.1101/2024.05.13.593807](https://doi.org/10.1101/2024.05.13.593807).

## License

This project is licensed under the terms of the GPL v3 license. See the LICENSE file for details.
