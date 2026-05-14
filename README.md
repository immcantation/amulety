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

The full AMULETY usage documentation can be found on the
[readthedocs page](https://amulety.readthedocs.io/).

You can install AMULETY using conda (it requires python 3.8 or higher):

```bash
conda install -c conda-forge -c bioconda amulety --strict-channel-priority
```

The conda installation will also install the necessary IgBlast dependency. Installing `amulety` through conda should also install Python, however if the error `amulety: command not found` is encountered after this installation, please ensure that Python was correctly installed or add Python to the installation command:

```bash
conda install -c conda-forge -c bioconda python amulety --strict-channel-priority
```

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

To print the usage help for the AMULETY package type:

```bash
amulety --help
```

### Python version compatibility and AbLang support

AMULETY supports Python versions `>=3.8,<4`. However, the optional **AbLang** model (used for some antibody embeddings) depends on `numba`, which currently does **not** support Python 3.14 and above. Beginning with Python 3.14, AbLang is automatically excluded from installation and is unavailable at runtime.

If you run AMULETY on Python 3.14+, attempting to use `--model ablang` (or programmatic use of the AbLang embedding function) will raise a clear `ImportError` explaining the limitation. To use AbLang, please install AMULETY under a Python version `<3.14`:

```bash
conda create -n amulety-ablang python=3.13
conda activate amulety-ablang
pip install amulety ablang
```

The build configuration uses an environment marker to skip installing AbLang on Python 3.14+ so normal installation and other models continue to work without failure.

## Using the docker container

The docker container is available under `immcantation/amulety`. Please refer to the [docker documentation](https://docs.docker.com/engine/install/) to install docker first on your system.

To use amulety from within the container run:

```
docker run -itv `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) immcantation/amulety amulety embed --input-airr tests/AIRR_rearrangement_translated_mixed.tsv --chain H --model immune2vec --output-file-path test_fixed.tsv --cache-dir /tmp/cache
```

You can also create an alias so that you don't need to type all of this each time you call amulety:

```
alias amulety="docker run -itv `pwd`:`pwd` -w `pwd` -u $(id -u):$(id -g) immcantation/amulety amulety"
```

Once applied you can just use the amulety command instead:

```
amulety embed --input-airr AIRR_translated.tsv --chain H --model antiberta2 --output-file-path antiberta2_embeddings.tsv
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

This project is licensed under the terms of the MIT license.
