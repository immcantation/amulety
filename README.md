# AMULETY

Amulety stands for Adaptive imMUne receptor Language model Embedding Tool.
It is a Python command line tool to embed B-cell receptor (antibody) and T-cell receptor (TCR) amino acid sequences using pre-trained protein or antibody language models. The package supports both BCR and TCR embeddings, with specialized TCR processing for alpha-beta chain concatenation. The package also has functionality to translate nucleotide sequences to amino acids with IgBlast.

Here is the list of currently supported embeddings:

## Chain Type Classification

AMULETY supports different chain input formats based on model architecture and training data:

- **H**: Heavy chains (BCR) or Beta/Delta chains (TCR) - individual chain embedding
- **L**: Light chains (BCR) or Alpha/Gamma chains (TCR) - individual chain embedding
- **HL**: Paired chains - concatenated Heavy-Light (BCR) or Beta-Alpha/Delta-Gamma (TCR) sequences
- **LH**: Reverse paired chains - concatenated Light-Heavy (BCR) or Alpha-Beta/Gamma-Delta (TCR) sequences
- **H+L**: Both chains separately - processes H and L chains individually without pairing

## BCR (B-Cell Receptor) Models

| Model       | Command     | Embedding Dimension | Chain Support | Reference                                                                        |
| ----------- | ----------- | ------------------- | ------------- | -------------------------------------------------------------------------------- |
| AbLang      | ablang      | 768                 | H, L, H+L     | [doi.org/10.1101/2022.01.20.477061](https://doi.org/10.1101/2022.01.20.477061)   |
| AntiBERTa2  | antiberta2  | 1024                | H, L, H+L     | [doi:10.1016/j.patter.2022.100513](https://doi.org/10.1016/j.patter.2022.100513) |
| AntiBERTy   | antiberty   | 512                 | H, L, H+L     | [doi:10.48550/arXiv.2112.07782](https://doi.org/10.48550/arXiv.2112.07782)       |
| BALM-paired | balm-paired | 1024                | HL, LH        | [doi:10.1016/j.patter.2024.100967](https://doi.org/10.1016/j.patter.2024.100967) |

## TCR (T-Cell Receptor) Models

| Model    | Command  | Embedding Dimension | Chain Support     | Reference                                                                                    |
| -------- | -------- | ------------------- | ----------------- | -------------------------------------------------------------------------------------------- |
| TCR-BERT | tcr-bert | 768                 | H, L, HL, LH, H+L | [doi:10.1101/2021.11.18.469186](https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1) |
| TCRT5    | tcrt5    | 256                 | H only            | [doi.org/10.1101/2024.11.11.623124](https://doi.org/10.1101/2024.11.11.623124)               |

## General Protein Models

| Model                 | Command | Embedding Dimension | Chain Support    | Reference                                                                                              |
| --------------------- | ------- | ------------------- | ---------------- | ------------------------------------------------------------------------------------------------------ |
| ESM2 (650M parameter) | esm2    | 1280                | H, L, H+L, HL/LH | [doi:10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)                                 |
| ProtT5                | prott5  | 1024                | H, L, H+L, HL/LH | [doi:10.1101/2020.07.12.199554](https://doi.org/10.1101/2020.07.12.199554)                             |
| Custom models         | custom  | Configurable        | H, L, H+L, HL/LH | User-provided fine-tuned or custom models (requires --model-path, --embedding-dimension, --max-length) |

## Immune Receptor Specific Models (BCR & TCR)

| Model      | Command    | Embedding Dimension | Chain Support              | TCR Type Support | Reference                                                                                                                |
| ---------- | ---------- | ------------------- | -------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Immune2Vec | immune2vec | 100 (configurable)  | H, L, H+L + (warning)HL/LH | α/β + γ/δ        | [doi:10.3389/fimmu.2021.680687](https://www.frontiersin.org/journals/immunology/articles/10.3389/fimmu.2021.680687/full) |

## Installation

### Requirements

- **Python 3.8 or higher** (Python 3.11+ required only for external TCREMP tool)

### Install AMULETY

You can install AMULETY using pip:

```bash
pip install amulety
```

Or install from source:

```bash
git clone https://github.com/immcantation/amulety.git
cd amulety
pip install -e .
```

### Optional Dependencies

Most models work out-of-the-box and do not require further tool installations. Some models require additional dependencies that are not installable with PyPI:

**Protein Language Models:**

- **Immune2Vec** - Protein language model for immune receptor sequences. Requires specific setup:

  **Prerequisites:**

  ```bash
  # Install required dependencies
  python3 -m pip install gensim==3.8.3
  pip3 install ray
  ```

  **Setup:**

  ```bash
  # Clone the Immune2Vec repository
  git clone https://bitbucket.org/yaarilab/immune2vec_model.git
  ```

  **Usage:**

  ```bash
  # Use with custom path parameter
  amulety embed --model immune2vec --immune2vec-path /path/to/immune2vec_model --input-airr data.tsv --chain H --output-file-path output.pt
  ```

  **Python API:**

  ```python
  from amulety.protein_embeddings import immune2vec
  embeddings = immune2vec(sequences, immune2vec_path='/path/to/immune2vec_model')
  ```

  **Note:** Immune2Vec requires gensim version 3.8.3 specifically. If installation fails due to compilation issues, ensure you have compatible Python version (3.8-3.9 recommended).

### Custom Light Chain Selection

When using paired chains (`--chain HL`), AMULETY automatically selects the most abundant light chain when multiple light chains exist for the same cell. By default, it uses the `duplicate_count` column (sequence read count), but you can specify a custom numeric column:

```bash
# Default behavior: use duplicate_count
amulety embed --chain HL --model antiberta2 --output-file-path embeddings.pt input.tsv

# Custom selection: use a quality score column
amulety embed --chain HL --model antiberta2 --duplicate-col quality_score --output-file-path embeddings.pt input.tsv

# Custom selection: use UMI count
amulety embed --chain HL --model antiberta2 --duplicate-col umi_count --output-file-path embeddings.pt input.tsv
```

**Requirements for Custom Selection Columns:**

The column must contain numeric values (integers or floats), AMULETY selects the chain with the highest value in that column.

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
amulety embed --chain HL --model antiberta2 --duplicate-col quality_score --output-file-path embeddings.pt enhanced_input.tsv
```

### Troubleshooting Models

If you encounter errors about missing packages, AMULETY will provide detailed installation instructions. To use the actual models:

1. **Check which packages are missing:**

```bash
amulety check-deps
```

Or in Python:

```python
from amulety.utils import check_dependencies
check_dependencies()
```

2. **Install missing packages** following the instructions above

3. **Verify installation** by running the check again

## Usage

To print the usage help for the AMULETY package type:

```bash
amulety --help
```

The full usage documentation can also be found on the readthedocs [usage page](https://amulety.readthedocs.io/en/latest/usage.html).

### Using Custom Models

To use any fine-tuned or custom model, use the `custom` command with the required parameters:

```bash
# Using a custom model from HuggingFace
amulety embed --chain HL --model custom --model-path "your-username/your-custom-model" --embedding-dimension 1280 --max-length 512 --output-file-path embeddings.pt input.tsv

# Using a local custom model
amulety embed --chain HL --model custom --model-path "/path/to/local/model" --embedding-dimension 768 --max-length 256 --output-file-path embeddings.pt input.tsv
```

**Important Requirements for Custom Models:**

1. **HuggingFace Compatibility**: Must be compatible with `transformers.AutoModelForMaskedLM` or similar interfaces
2. **Tokenizer Compatibility**: Should use a compatible tokenizer (ESM2, BERT, or similar)
3. **Output Dimensions**: You must specify the correct embedding dimension with `--embedding-dimension`
4. **Sequence Length**: You must specify the maximum sequence length with `--max-length`
5. **Model Architecture**: Works best with transformer-based protein language models

**Supported Model Sources:**

- HuggingFace Hub models (e.g., `username/model-name`)
- Local model directories
- Any transformer-based protein language model
- Fine-tuned versions of ESM2, ProtBERT, ProtT5, or similar models

**Note**: AMULETY will attempt to load any model you specify, but compatibility depends on the model architecture and tokenizer. Transformer-based protein language models work best.

## Contact

If you need help or have any questions, please contact the [Immcantation Group](mailto:immcantation@googlegroups.com).

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

This project is licensed under the terms of the GPL v3 license. See the LICENSE file for details.
