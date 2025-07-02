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

| Model                 | Command | Embedding Dimension | TCR Type Support | Reference                                                                                      |
| --------------------- | ------- | ------------------- | ---------------- | ---------------------------------------------------------------------------------------------- |
| ESM2 (650M parameter) | esm2    | 1280                | α/β + γ/δ        | [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)                         |
| Fine-tuned models     | custom  | Configurable        | depends on model | Fine-tuned ESM2 and other custom models (requires model_path, embedding_dimension, max_length) |
| ProtT5                | prott5  | 1024                | α/β + γ/δ        | [doi:10.1101/2020.07.12.199554](https://doi.org/10.1101/2020.07.12.199554)                     |
| User-specified model  | custom  | Configurable        | depends on model | Custom model support                                                                           |

## Immune-Specific Models (BCR & TCR)

| Model      | Command    | Embedding Dimension | TCR Type Support | Reference                                                                                                                |
| ---------- | ---------- | ------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Immune2Vec | immune2vec | 100 (configurable)  | α/β + γ/δ        | [doi:10.3389/fimmu.2021.680687](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.680687/full) |

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

### Using Fine-tuned and Custom Models

To use a fine-tuned ESM2 model or other custom models, use the `custom` command with the required parameters:

```bash
# Using a fine-tuned ESM2 model from HuggingFace
amulety embed --chain HL --model custom --model-path "your-username/esm2-bcr-finetuned" --embedding-dimension 1280 --max-length 512 --output-file-path embeddings.pt input.tsv

# Using a local fine-tuned model
amulety embed --chain HL --model custom --model-path "/path/to/local/model" --embedding-dimension 1280 --max-length 512 --output-file-path embeddings.pt input.tsv
```

**Important Requirements for Custom Models:**

1. **Architecture Compatibility**: For fine-tuned ESM2 models, must be based on ESM2 architecture (facebook/esm2_t33_650M_UR50D)
2. **Tokenizer Compatibility**: Should use the same tokenizer as base ESM2
3. **Output Dimensions**: Typically 1280-dimensional embeddings (will auto-detect if different)
4. **HuggingFace Format**: Must be compatible with `transformers.AutoModelForMaskedLM`

**Supported Model Sources:**

- HuggingFace Hub models (e.g., `username/model-name`)
- Local model directories
- Any ESM2-compatible fine-tuned model

**Note**: AMULETY will attempt to load any model you specify, but compatibility is not guaranteed for non-ESM2 models.

### Custom Light Chain Selection

When using paired chains (`--chain HL`), AMULETY automatically selects the best light chain when multiple light chains exist for the same cell. By default, it uses the `duplicate_count` column (sequence read count), but you can specify a custom numeric column:

```bash
# Default behavior: use duplicate_count
amulety embed --chain HL --model antiberta2 --output-file-path embeddings.pt input.tsv

# Custom selection: use a quality score column
amulety embed --chain HL --model antiberta2 --selection-col quality_score --output-file-path embeddings.pt input.tsv

# Custom selection: use UMI count
amulety embed --chain HL --model antiberta2 --selection-col umi_count --output-file-path embeddings.pt input.tsv
```

**Requirements for Custom Selection Columns:**

1. **Numeric Type**: The column must contain numeric values (integers or floats)
2. **Higher is Better**: AMULETY selects the chain with the highest value
3. **User-Defined**: You need to add these columns to your AIRR data file yourself

**Common Custom Columns:**

- `quality_score`: Sequence quality metrics
- `umi_count`: Unique molecular identifier counts
- `expression_level`: Gene expression levels
- `confidence_score`: Assembly confidence scores

**Example: Adding a Custom Column**

```python
import pandas as pd

# Read your AIRR data
data = pd.read_csv('input.tsv', sep='\t')

# Add a custom quality score (example calculation)
data['quality_score'] = data['duplicate_count'] * data['sequence_length'] / 100

# Save enhanced data
data.to_csv('enhanced_input.tsv', sep='\t', index=False)
```

Then use with AMULETY:

```bash
amulety embed --chain HL --model antiberta2 --selection-col quality_score --output-file-path embeddings.pt enhanced_input.tsv
```

### Using Immune2Vec

Immune2Vec requires cloning the repository. To use it:

1. **Clone the Immune2Vec repository**:

```bash
git clone https://github.com/edelarosilva/immune2vec.git
cd immune2vec
```

2. **Add to Python path** (in your script):

```python
import sys
sys.path.append('/path/to/immune2vec')
```

3. **Use with AMULETY**:

```bash
amulety embed --chain HL --model immune2vec --output-file-path embeddings.pt input.tsv
```

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
