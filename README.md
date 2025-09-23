# AMULETY

AMULETY stands for Adaptive imMUne receptor Language model Embedding tool for TCR and antibodY.
AMULETY is a Python command line tool to embed B-cell receptor (BCR), also termed antibodies in
their secreted form, and T-cell receptor (TCR) amino acid sequences using pre-trained general
protein or specific immune receptor language models. The package supports both BCR and TCR embeddings.
The package also has functionality to translate nucleotide sequences to amino acids with IgBlast.

AMULETY is part of the [Immcantation](http://immcantation.readthedocs.io)
analysis framework for Adaptive Immune Receptor Repertoire sequencing
(AIRR-seq) data analysis.

## Quick start

You can install AMULETY using conda (it requires python 3.8 or higher):

```bash
conda install amulety
```

The conda installation will also install the necessary IgBlast dependency.
You can also install AMULETY via pip, this will though require previously
installing IgBlast if translations are desired.

```bash
pip install amulety
```

Or install from source:

```bash
git clone https://github.com/immcantation/amulety.git
cd amulety
pip install -e .
```

The full usage documentation can also be found on the readthedocs
[usage page](https://amulety.readthedocs.io/en/latest/usage.html).

To print the usage help for the AMULETY package type:

```bash
amulety --help
```

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
