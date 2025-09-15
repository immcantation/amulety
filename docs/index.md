# Amulety

Amulety stands for Adaptive imMUne receptor Language model Embedding tool for TCR and antibodY.
Amulety is a Python command line tool to embed B-cell receptor (BCR), also termed antibodies in their secreted form, and T-cell receptor (TCR) amino acid sequences using pre-trained general protein or specific immune receptor language models. The package supports both BCR and TCR embeddings. The package also has functionality to translate nucleotide sequences to amino acids with IgBlast.

## Contact

If you need help or have any questions, please contact the
[Immcantation Group](mailto:immcantation@googlegroups.com).

If you have discovered a bug or have a feature request, you can open an issue using the [issue tracker](https://github.com/immcantation/amulety/issues).

To receive alerts about Immcantation releases, news, events, and tutorials, join the [Immcantation News](https://groups.google.com/g/immcantation-news) Google Group. [Membership settings](https://groups.google.com/g/immcantation-news/membership) can be adjusted to change the frequency of email updates.

## Authors

[Mamie Wang](https://github.com/mamie) (aut,cre)
[Gisela Gabernet](https://github.com/ggabernet) (aut,cre)
[Wengyao Jiang](https://github.com/wenggyaoo) (aut,cre)
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

This project is licensed under the terms of the GPL v3 license.

```{toctree}
:hidden:
:maxdepth: 2

Immcantation Portal <https://immcantation.readthedocs.io>
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: About

Introduction <self>
installation
Included models <included_models>
Release Notes <history>
contributing
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: Tutorials

tutorials/amulety_cli
tutorials/ML_tutorial
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: CLI reference

cli
```

```{toctree}
:hidden:
:maxdepth: 2
:caption: API reference

api
```
