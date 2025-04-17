# AMULETY

Amulety stands for Adaptive imMUne receptor Language model Embedding Tool.
It is a Python command line tool to embed B-cell receptor (antibody) and T-cell Receptor amino acid sequences using pre-trained protein or antibody language models. So far only BCR embeddings are supported but TCR support is planned for future releases. The package also has functionality to translate nucleotide sequences to amino acids with IgBlast.

Here is the list of currently supported embeddings:

| Model                 | Command     | Embedding Dimension | Max Length   | Notes                                                                                                                                                                                         | Reference                                                                                                                                                                                                                                    |
| --------------------- | ----------- | ------------------- | ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| AntiBERTy             | antiberty   | 512                 | 510          | Lightweight BERT model pre-trained on 588 million Observed Antibody Space (OAS) heavy and light antibody sequences from multiple species. Suitable with low computational resources.          | Ruffolo JA, Gray JJ, Sulam J. Deciphering antibody affinity maturation with language models and weakly supervised learning. arXiv. 2021; 2112.07782. [doi:10.48550/arXiv.2112.07782](https://doi.org/10.48550/arXiv.2112.07782)              |
| AntiBERTa2            | antiberta2  | 1024                | 256          | RoFormer model pre-trained on 1.54 billion unpaired and 2.9 million paired human antibody sequences. Suitable for users needing a balance between computational cost and embedding dimension. | Leem J, Mitchell LS, Farmery JHR, Barton J, Galson JD. Deciphering the language of antibodies using self-supervised learning. Patterns. 2022;3: 100513. [doi:10.1016/j.patter.2022.100513](https://doi.org/10.1016/j.patter.2022.100513)     |
| ESM2 (650M parameter) | esm2        | 1280                | 512          | General protein language model pre-trained on 216 million UniRef50 protein sequences. Ideal for complex tasks and requires heavy computational resources.                                     | Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379: 1123–1130. [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574) |
| BALM-paired           | balm_paired | 1024                | 510          | RoBERTa-based large model pre-trained on 1.34 million concatenated heavy and light chain human antibody sequences. Specialized model for paired chain embeddings.                             | Burbach SM, Briney B. Improving antibody language models with native pairing. Patterns. 2024;5. [doi:10.1016/j.patter.2024.100967](https://doi.org/10.1016/j.patter.2024.100967)                                                             |
| User-specified model  | custommodel | Configurable        | Configurable | Offers flexibility for users with pre-trained custom models.                                                                                                                                  |                                                                                                                                                                                                                                              |

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
