# Installation

## Stable release

You can install AMULETY using conda or pip. To install AMULETY using conda, run this command:

```console
conda install -c bioconda amulety
```

This is the preferred method to install AMULETY, as it will install all the needed dependencies.

To install amulety with pip, you can run this command:

```console
pip install amulety
```

If you don't have [pip](https://pip.pypa.io) installed, this [Python installation guide](http://docs.python-guide.org/en/latest/starting/installation/) can guide you through the process.

## From sources

The sources for AMULETY can be downloaded from the [Github repo](https://github.com/immcantation/amulety).

You can either clone the public repository:

```console
git clone https://github.com/immcantation/amulety
```

Or download the [tarball](https://github.com/immcantation/amulety/tarball/master):

```console
curl -OJL https://github.com/immcantation/amulety/tarball/master
```

Once you have a copy of the source, you can install it with:

```console
pip install .
```

## Using the docker container

The docker container is available under `immcantation/amulety`. Please refer to the [docker documentation]() to install docker first on your system.

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

## Model-specific Dependencies

Most models work out-of-the-box and do not require further tool installations. Some models require additional dependencies that are not installable with PyPI:

### Immune2vec

The **Immune2vec** language model for immune receptor sequences requires specific setup:

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

**CLI Usage:**

```bash
# Use with custom path parameter
amulety embed --model immune2vec --installation-path /path/to/immune2vec_model --input-airr data.tsv --chain H --output-file-path output.pt
```

**Python API:**

```python
from amulety.protein_embeddings import immune2vec
embeddings = immune2vec(sequences, installation_path='/path/to/immune2vec_model')
```

**Note:** Immune2Vec requires gensim version 3.8.3 specifically. If installation fails due to compilation issues, ensure you have compatible Python version (3.8-3.9 recommended).

## Troubleshooting Models

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

## Using Custom Models

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
