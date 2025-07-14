"""
Protein sequence embedding functions using various models.
Please order alphabetically by function name.
"""
# ruff: noqa: N806

import logging
import math
import time
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from amulety.utils import batch_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def custommodel(
    sequences: pd.Series,
    model_path: str,
    embedding_dimension: int,
    max_seq_length: int,
    cache_dir: Optional[str] = "/tmp/amulety",
    batch_size: Optional[int] = 50,
):
    """
    Embeds sequences using a custom model specified by the user. The maximum length of the sequences to be embedded is specified by the user.
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("Model size: %sM", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, embedding_dimension))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        print(f"Batch {i}/{n_batches}\n")
        x = torch.tensor(
            [
                tokenizer.encode(
                    seq,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_special_tokens_mask=True,
                )
                for seq in batch
            ]
        ).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
            outputs = outputs.hidden_states[-1]
            outputs = list(outputs.detach())

        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1, :].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))

    return embeddings


def esm2(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
):
    """
    Embeds sequences using the ESM2 model. The maximum length of the sequences to be embedded is 512. The embedding dimension is 1280.

    Args:
        sequences: Input protein sequences
        cache_dir: Directory to cache model files
        batch_size: Number of sequences to process in each batch
        model_name: HuggingFace model name or path to fine-tuned model
    """
    max_seq_length = 512
    dim = 1280
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("ESM2 650M model size: %s M", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("Batch %s/%s.", i, n_batches)
        x = torch.tensor(
            [
                tokenizer.encode(
                    seq,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_special_tokens_mask=True,
                )
                for seq in batch
            ]
        ).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
            outputs = outputs.hidden_states[-1]
            outputs = list(outputs.detach())

        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1, :].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))

    return embeddings


def prott5(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Embeds BCR or TCR sequences using the ProtT5-XL protein language model (Rostlab/prot_t5_xl_uniref50). The maximum sequence length to embed is 1024 amino acids, and the generated embeddings have a dimension of 1024.
    """
    max_seq_length = 1024  # ProtT5 can't handle longer sequences
    dim = 1024

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # bcr_chain = {"A": "L", "B": "H", "AB": "HL"}.get(chain, chain)

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])

    # ProtT5 expects space-separated amino acids
    X = X.apply(lambda seq: " ".join(list(seq.replace("<cls><cls>", " <cls> <cls> "))))
    sequences = X.values

    logger.info("Loading ProtT5 model for protein sequence embedding...")

    try:
        from transformers import T5EncoderModel, T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=cache_dir, do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=cache_dir)
        logger.info("Successfully loaded ProtT5 XL encoder model")
        model_type = "encoder"
    except Exception as e:
        logger.warning("ProtT5 XL encoder failed, trying base model: %s", str(e))
        try:
            tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_base_uniref50", cache_dir=cache_dir, do_lower_case=False
            )
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_base_uniref50", cache_dir=cache_dir)
            logger.info("Successfully loaded ProtT5 base encoder model")
            model_type = "encoder"
        except Exception as e2:
            logger.error("Failed to load ProtT5 models: %s", str(e2))

            if "SentencePiece" in str(e2):
                raise RuntimeError(
                    "ProtT5 requires the SentencePiece library. Please install it with:\n"
                    "pip install sentencepiece\n"
                    "Then restart your environment and try again.\n"
                    "If you want to use ESM2 instead, please use the 'esm2' command."
                ) from e2
            else:
                raise RuntimeError(
                    f"Failed to load ProtT5 model: {str(e2)}\n"
                    "Please check your internet connection and try again.\n"
                    "If you want to use ESM2 instead, please use the 'esm2' command."
                ) from e2
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("ProtT5 model loaded. Size: %s M", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)

    embeddings = torch.empty((n_seqs, dim))
    logger.info("Initialized embeddings tensor with ProtT5 dimension: %s", dim)

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("ProtT5 Batch %s/%s.", i, n_batches)

        x = torch.tensor(
            [
                tokenizer.encode(
                    seq,
                    padding="max_length",
                    truncation=True,
                    max_length=max_seq_length,
                    return_special_tokens_mask=True,
                )
                for seq in batch
            ]
        ).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)

        with torch.no_grad():
            if model_type == "encoder":
                outputs = model(input_ids=x, attention_mask=attention_mask)
                outputs = outputs.last_hidden_state
            else:
                outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
                outputs = outputs.hidden_states[-1]
            outputs = list(outputs.detach())

        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1, :].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("ProtT5 embedding took %s seconds", round(end_time - start_time, 2))

    return embeddings


def immune2vec(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
    n_dim: int = 100,
    n_gram: int = 3,
    pretrained_model_path: Optional[str] = None,
    data_fraction: float = 1.0,
    window: int = 25,
    min_count: int = 1,
    workers: int = 3,
    random_seed: int = 42,
):
    """
    Embeds sequences using Immune2Vec model.

    Immune2Vec is a Word2Vec-based embedding method specifically designed for
    immune receptor sequences (both BCR and TCR). It uses n-gram decomposition
    of amino acid sequences to learn vector representations.


    Args:
        sequences: Input protein sequences
        cache_dir: Directory to cache model files
        batch_size: Number of sequences to process (not used for Immune2Vec but kept for consistency)
        n_dim: Embedding dimension (default: 100)
        n_gram: N-gram size for sequence decomposition (default: 3)
        pretrained_model_path: Path to a pre-trained Immune2Vec model (optional)
        data_fraction: Fraction of data to use for training (default: 1.0)
        window: Context window size for Word2Vec (default: 25)
        min_count: Minimum count for words to be included (default: 1)
        workers: Number of worker threads (default: 3)
        random_seed: Random seed for reproducibility (default: 42)

    Returns:
        torch.Tensor: Embeddings of shape (n_sequences, n_dim)
    """
    import os

    import numpy as np

    start_time = time.time()

    try:
        import gensim

        logger.info("Gensim available (version: %s)", gensim.__version__)

        # try to import immune2vec since this might need to be installed separately
        try:
            from embedding import sequence_modeling

            logger.info("Immune2Vec package successfully imported")
        except ImportError as immune2vec_error:
            logger.warning("Immune2Vec package not found: %s", str(immune2vec_error))
            logger.warning("Returning placeholder embeddings. To use actual Immune2Vec:")
            logger.info("1. Install gensim: pip install gensim>=3.8.3")
            logger.info("2. Clone repository: git clone https://bitbucket.org/yaarilab/immune2vec_model.git")
            logger.info("3. Add to Python path: sys.path.append('/path/to/immune2vec_model')")

            # Return placeholder embeddings
            n_seqs = len(sequences)
            placeholder_embeddings = torch.randn(n_seqs, n_dim)
            logger.info(f"Immune2Vec placeholder embedding completed. Shape: {placeholder_embeddings.shape}")
            return placeholder_embeddings

    except ImportError as gensim_error:
        if "gensim" in str(gensim_error):
            logger.warning("Gensim not found: %s", str(gensim_error))
            logger.warning("Returning placeholder embeddings. To use actual Immune2Vec:")
            logger.info("Install Gensim: pip install gensim>=3.8.3")

            # Return placeholder embeddings
            n_seqs = len(sequences)
            placeholder_embeddings = torch.randn(n_seqs, n_dim)
            logger.info(f"Immune2Vec placeholder embedding completed. Shape: {placeholder_embeddings.shape}")
            return placeholder_embeddings
        else:
            logger.error("Unexpected import error: %s", str(gensim_error))
            raise

    logger.info("Starting Immune2Vec embedding...")
    logger.info("Parameters: n_dim=%d, n_gram=%d, window=%d", n_dim, n_gram, window)

    # Sequences are already cleaned by process_airr function
    if len(sequences) == 0:
        raise ValueError("No valid sequences found for embedding")

    logger.info("Processing %d sequences", len(sequences))

    # Create cache directory if specified
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    # Helper functions
    def train_immune2vec(sequences_train, out_corpus_fname):
        """Train Immune2Vec model following reference implementation."""
        return sequence_modeling.ProtVec(
            data=sequences_train,
            n=n_gram,
            reading_frame=None,
            trim=None,
            size=n_dim,
            out=out_corpus_fname,
            sg=1,  # Skip-gram
            window=window,
            min_count=min_count,
            workers=workers,
            sample_fraction=data_fraction,
            random_seed=random_seed,
        )

    def embed_data_helper(word):
        """Helper function to embed individual sequences with error handling."""
        try:
            return model.to_vecs(word, n_read_frames=None)
        except Exception:
            return np.nan

    try:
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logger.info("Loading pre-trained Immune2Vec model from: %s", pretrained_model_path)
            model = sequence_modeling.load_protvec(pretrained_model_path)
        else:
            logger.info("Training new Immune2Vec model...")

            corpus_filename = "immune2vec_temp_corpus"
            if cache_dir:
                corpus_filename = os.path.join(cache_dir, corpus_filename)

            model = train_immune2vec(sequences, corpus_filename)

            # save model to cache if cache_dir is provided
            if cache_dir:
                model_path = os.path.join(cache_dir, f"immune2vec_{n_gram}mer_{n_dim}dim.model")
                model.save(model_path)
                logger.info("Saved trained model to: %s", model_path)

        logger.info("Generating embeddings...")

        # Apply embedding function to all sequences (following reference implementation)
        embed_vectors = sequences.apply(embed_data_helper)

        # Convert to list and handle NaN values
        embeddings_list = []
        for i, embedding in enumerate(embed_vectors):
            if embedding is not np.nan and embedding is not None:
                try:
                    if isinstance(embedding, np.ndarray):
                        embeddings_list.append(embedding)
                    else:
                        embeddings_list.append(np.array(embedding))
                except Exception:
                    logger.warning("Could not process embedding for sequence %d, using zero vector", i)
                    embeddings_list.append(np.zeros(n_dim))
            else:
                logger.warning("Could not embed sequence %d, using zero vector", i)
                embeddings_list.append(np.zeros(n_dim))

        # Convert to torch tensor
        embeddings = torch.tensor(np.array(embeddings_list), dtype=torch.float32)

        end_time = time.time()
        logger.info("Immune2Vec embedding completed in %.2f seconds", end_time - start_time)
        logger.info("Generated embeddings shape: %s", embeddings.shape)

        return embeddings

    except Exception as e:
        logger.error("Failed to generate Immune2Vec embeddings: %s", str(e))
        raise RuntimeError(f"Immune2Vec embedding failed: {str(e)}") from e
