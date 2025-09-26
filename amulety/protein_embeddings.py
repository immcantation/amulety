"""
Protein sequence embedding functions using various models.
"""
# Please order alphabetically by function name.
# ruff: noqa: N806

import logging
import math
import time
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from amulety.utils import batch_loader

logger = logging.getLogger(__name__)

# Immune2Vec installation instructions (used in multiple places)
IMMUNE2VEC_INSTALLATION_INSTRUCTIONS = (
    "Immune2Vec package not available. Please follow these installation instructions:\n\n"
    "STEP 1: Install gensim dependency (if not already installed)\n"
    "   pip install gensim>=3.8.3\n\n"
    "STEP 2: Clone the Immune2Vec repository\n"
    "   git clone https://bitbucket.org/yaarilab/immune2vec_model.git\n\n"
    "STEP 3: Either add to Python path or specify custom path\n"
    "   Option A: import sys; sys.path.append('/path/to/immune2vec_model')\n"
    "   Option B: Use installation_path parameter: immune2vec(..., installation_path='/path/to/immune2vec_model')\n\n"
    "STEP 4: Verify installation\n"
    "   python -c 'from embedding import sequence_modeling; print(\"Immune2Vec installed successfully\")'\n\n"
    "Reference: https://bitbucket.org/yaarilab/immune2vec_model/src/master/"
)


def custommodel(
    sequences,
    model_path: str,
    embedding_dimension: int,
    max_seq_length: int,
    cache_dir: Optional[str] = "/tmp/amulety",
    residue_level: bool = False,
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

    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_path, cache_dir=cache_dir)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("Model size: %sM", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)
    if residue_level:
        embeddings = torch.empty((n_seqs, max_seq_length, embedding_dimension))
    else:
        embeddings = torch.empty((n_seqs, embedding_dimension))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info(f"Batch {i}/{n_batches}\n")
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

        if not residue_level:
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
    sequences,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    residue_level: bool = False,
):
    """
    Embeds sequences using the ESM2 model. The maximum length of the sequences to be embedded is 512. The embedding dimension is 1280.

    Args:
        sequences: Input protein sequences (pd.Series for single chain or pd.DataFrame for H+L mode)
        cache_dir: Directory to cache model files
        batch_size: Number of sequences to process in each batch
        model_name: HuggingFace model name or path to fine-tuned model
    """
    max_seq_length = 512

    # Handle both Series (single chain) and DataFrame (H+L) inputs
    if isinstance(sequences, pd.DataFrame):
        # H+L mode: DataFrame contains rows with sequence data
        if "chain" in sequences.columns:
            # Look for common sequence column names
            sequence_col_candidates = ["sequence_vdj_aa", "sequence_aa", "sequence"]
            sequence_col = None
            for col in sequence_col_candidates:
                if col in sequences.columns:
                    sequence_col = col
                    break

            if sequence_col is None:
                raise ValueError(
                    f"No recognized sequence column found in DataFrame. Expected one of: {sequence_col_candidates}"
                )

            X = sequences[sequence_col].apply(lambda a: str(a)[:max_seq_length])
            sequences_array = X.values
        else:
            raise ValueError("DataFrame input must contain 'chain' column for H+L mode")
    else:
        # Single chain mode
        X = sequences.apply(lambda a: str(a)[:max_seq_length])
        sequences_array = X.values
    dim = 1280
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # sequences_array is already processed above
    sequences = sequences_array

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_name, cache_dir=cache_dir)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("ESM2 650M model size: %s M", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)
    if residue_level:
        embeddings = torch.empty((n_seqs, max_seq_length, dim))
    else:
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

        if not residue_level:
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
    sequences,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    residue_level: bool = False,
):
    """
    Embeds BCR or TCR sequences using the ProtT5-XL protein language model (Rostlab/prot_t5_xl_uniref50).
    The maximum sequence length to embed is 1024 amino acids, and the generated embeddings have a dimension of 1024.

    Args:
        sequences: Input protein sequences (pd.Series for single chain or pd.DataFrame for H+L mode)
        cache_dir: Directory to cache model files
        batch_size: Number of sequences to process in each batch
    """
    max_seq_length = 1024  # ProtT5 can't handle longer sequences
    dim = 1024

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Handle both Series (single chain) and DataFrame (H+L) inputs
    if isinstance(sequences, pd.DataFrame):
        # H+L mode: DataFrame contains rows with sequence data
        if "chain" in sequences.columns:
            # Look for common sequence column names
            sequence_col_candidates = ["sequence_vdj_aa", "sequence_aa", "sequence"]
            sequence_col = None
            for col in sequence_col_candidates:
                if col in sequences.columns:
                    sequence_col = col
                    break

            if sequence_col is None:
                raise ValueError(
                    f"No recognized sequence column found in DataFrame. Expected one of: {sequence_col_candidates}"
                )

            X = sequences[sequence_col].apply(lambda a: str(a)[:max_seq_length])
        else:
            raise ValueError("DataFrame input must contain 'chain' column for H+L mode")
    else:
        # Single chain mode
        X = sequences.apply(lambda a: str(a)[:max_seq_length])
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

    if residue_level:
        embeddings = torch.empty((n_seqs, max_seq_length, dim))
    else:
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

        if not residue_level:
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
    sequences,
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
    installation_path: Optional[str] = None,
):
    """
    Embeds sequences using Immune2Vec model.

    Immune2Vec is a Word2Vec-based embedding method specifically designed for
    immune receptor sequences (both BCR and TCR). It uses n-gram decomposition
    of amino acid sequences to learn vector representations.

    Args:
        sequences: Input protein sequences (pd.Series for single chain or pd.DataFrame for H+L mode)
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
        installation_path: Custom path to Immune2Vec installation directory (required if not in PYTHONPATH)

    Returns:
        torch.Tensor: Embeddings of shape (n_sequences, n_dim)
    """
    import os

    import numpy as np

    start_time = time.time()

    try:
        import gensim

        logger.info("Gensim available (version: %s)", gensim.__version__)

    except ImportError as gensim_error:
        detailed_instructions = (
            "Gensim library is required for Immune2Vec but not installed.\n\n"
            "Please install gensim:\n"
            "   pip install gensim>=3.8.3\n\n"
            "Then follow the Immune2Vec installation instructions:\n"
            "1. Clone repository: git clone https://bitbucket.org/yaarilab/immune2vec_model.git\n"
            "2. Add to Python path: sys.path.append('/path/to/immune2vec_model')\n"
            "3. Verify: python -c 'from embedding import sequence_modeling'"
        )
        raise ImportError(detailed_instructions) from gensim_error

    # If user provided a specific path, validate it first
    if "IMMUNE2VEC_PATH" in os.environ and not installation_path:
        installation_path = os.environ["IMMUNE2VEC_PATH"]
    if installation_path:
        if not os.path.exists(installation_path) or not os.path.isdir(installation_path):
            raise ImportError(
                f"Invalid installation_path provided: '{installation_path}' does not exist or is not a directory.\n\n{IMMUNE2VEC_INSTALLATION_INSTRUCTIONS}"
            )

        # Try to import from the specified path
        import sys

        original_path = sys.path.copy()
        sys.path.insert(0, installation_path)
        try:
            from embedding import sequence_modeling

            logger.info("Immune2Vec package successfully imported from user-provided path: %s", installation_path)
        except ImportError as path_error:
            # Restore original path
            sys.path[:] = original_path
            raise ImportError(
                f"Cannot import Immune2Vec from provided path '{installation_path}'. Please verify the path contains a valid Immune2Vec installation.\n\n{IMMUNE2VEC_INSTALLATION_INSTRUCTIONS}"
            ) from path_error
    else:
        # try to import immune2vec since this might need to be installed separately
        try:
            from embedding import sequence_modeling

            logger.info("Immune2Vec package successfully imported")
        except ImportError as immune2vec_error:
            # Try to find and add immune2vec_model directory to path
            import sys

            # Build list of paths to search (only default locations since no specific path was provided)
            possible_paths = [
                "immune2vec_model",  # Current directory (CI environment)
                "../immune2vec_model",  # Parent directory
                os.path.expanduser("~/immune2vec_model"),  # Home directory
                "/tmp/immune2vec_model",  # Temporary directory
            ]

            immune2vec_found = False
            for path in possible_paths:
                if os.path.exists(path) and os.path.isdir(path):
                    logger.info("Found Immune2Vec repository at: %s", path)
                    sys.path.insert(0, path)
                    try:
                        from embedding import sequence_modeling

                        logger.info("Immune2Vec package successfully imported from: %s", path)
                        immune2vec_found = True
                        break
                    except ImportError:
                        logger.debug("Failed to import from: %s", path)
                        continue

            if not immune2vec_found:
                raise ImportError(
                    f"Immune2Vec package is required but not installed.\n\n{IMMUNE2VEC_INSTALLATION_INSTRUCTIONS}"
                ) from immune2vec_error

    logger.info("Starting Immune2Vec embedding...")
    logger.info("Parameters: n_dim=%d, n_gram=%d, window=%d", n_dim, n_gram, window)

    # Handle both Series (single chain) and DataFrame (H+L) inputs
    if isinstance(sequences, pd.DataFrame):
        # H+L mode: DataFrame contains rows with sequence data
        if "chain" in sequences.columns:
            # Look for common sequence column names
            sequence_col_candidates = ["sequence_vdj_aa", "sequence_aa", "sequence"]
            sequence_col = None
            for col in sequence_col_candidates:
                if col in sequences.columns:
                    sequence_col = col
                    break

            if sequence_col is None:
                raise ValueError(
                    f"No recognized sequence column found in DataFrame. Expected one of: {sequence_col_candidates}"
                )

            sequences = sequences[sequence_col]
        else:
            raise ValueError("DataFrame input must contain 'chain' column for H+L mode")

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
        logger.info("Training Immune2Vec with %d sequences", len(sequences_train))
        logger.info(
            "Training parameters: n_gram=%d, n_dim=%d, window=%d, min_count=%d", n_gram, n_dim, window, min_count
        )

        # Convert pandas Series to list if needed
        if hasattr(sequences_train, "tolist"):
            sequences_list = sequences_train.tolist()
        else:
            sequences_list = list(sequences_train)

        # Check for empty sequences
        valid_sequences = [seq for seq in sequences_list if seq and len(seq.strip()) > 0]
        if len(valid_sequences) != len(sequences_list):
            logger.warning("Filtered out %d empty sequences", len(sequences_list) - len(valid_sequences))

        if len(valid_sequences) == 0:
            raise ValueError("No valid sequences found for training")

        logger.info("Training with %d valid sequences", len(valid_sequences))

        try:
            model = sequence_modeling.ProtVec(
                data=valid_sequences,
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
            logger.info("Immune2Vec model training completed successfully")
            return model
        except Exception as e:
            logger.error("Failed to train Immune2Vec model: %s", str(e))
            raise

    def embed_data_helper(word):
        """Helper function to embed individual sequences with error handling."""
        try:
            result = model.to_vecs(word, n_read_frames=None)
            if result is None:
                logger.warning("Model returned None for sequence: %s", word[:20] + "..." if len(word) > 20 else word)
                return None
            return result
        except Exception as e:
            logger.warning("Failed to embed sequence '%s': %s", word[:20] + "..." if len(word) > 20 else word, str(e))
            return None

    try:
        if pretrained_model_path and os.path.exists(pretrained_model_path):
            logger.info("Loading pre-trained Immune2Vec model from: %s", pretrained_model_path)
            model = sequence_modeling.load_protvec(pretrained_model_path)
            logger.info("Successfully loaded pre-trained model")
        else:
            logger.info("Training new Immune2Vec model...")

            corpus_filename = "immune2vec_temp_corpus"
            if cache_dir:
                corpus_filename = os.path.join(cache_dir, corpus_filename)

            model = train_immune2vec(sequences, corpus_filename)

            # save model to cache if cache_dir is provided
            if cache_dir:
                model_path = os.path.join(cache_dir, f"immune2vec_{n_gram}mer_{n_dim}dim.model")
                try:
                    model.save(model_path)
                    logger.info("Saved trained model to: %s", model_path)
                except Exception as save_error:
                    logger.warning("Failed to save model: %s", str(save_error))

        # Validate model
        if not hasattr(model, "to_vecs"):
            raise AttributeError("Loaded model does not have 'to_vecs' method")

        # Test model with a simple sequence
        test_seq = sequences.iloc[0] if len(sequences) > 0 else "ACDEFGHIKLMNPQRSTVWY"
        try:
            test_result = model.to_vecs(test_seq, n_read_frames=None)
            if test_result is None:
                logger.warning("Model test returned None - this may indicate training issues")
            else:
                logger.info(
                    "Model validation successful, test embedding shape: %s",
                    np.array(test_result).shape if hasattr(test_result, "shape") else type(test_result),
                )
        except Exception as test_error:
            logger.warning("Model validation failed: %s", str(test_error))

        logger.info("Generating embeddings...")

        # Apply embedding function to all sequences (following reference implementation)
        embed_vectors = sequences.apply(embed_data_helper)

        # Convert to list and handle None/invalid values
        embeddings_list = []
        failed_count = 0
        for i, embedding in enumerate(embed_vectors):
            if embedding is not None:
                try:
                    if isinstance(embedding, np.ndarray):
                        # Check for NaN values in the embedding
                        if np.isnan(embedding).any():
                            logger.warning("Embedding for sequence %d contains NaN values, using zero vector", i)
                            embeddings_list.append(np.zeros(n_dim))
                            failed_count += 1
                        else:
                            embeddings_list.append(embedding)
                    else:
                        # Try to convert to numpy array
                        embedding_array = np.array(embedding)
                        if np.isnan(embedding_array).any():
                            logger.warning("Embedding for sequence %d contains NaN values, using zero vector", i)
                            embeddings_list.append(np.zeros(n_dim))
                            failed_count += 1
                        else:
                            embeddings_list.append(embedding_array)
                except Exception as e:
                    logger.warning("Could not process embedding for sequence %d: %s, using zero vector", i, str(e))
                    embeddings_list.append(np.zeros(n_dim))
                    failed_count += 1
            else:
                logger.warning("Could not embed sequence %d, using zero vector", i)
                embeddings_list.append(np.zeros(n_dim))
                failed_count += 1

        if failed_count > 0:
            logger.warning(
                "Failed to embed %d out of %d sequences (%.1f%%)",
                failed_count,
                len(sequences),
                (failed_count / len(sequences)) * 100,
            )

        # Convert to torch tensor
        try:
            embeddings_array = np.array(embeddings_list)
            logger.info("Embeddings array shape: %s", embeddings_array.shape)

            # Final check for NaN values
            if np.isnan(embeddings_array).any():
                nan_count = np.isnan(embeddings_array).sum()
                logger.warning("Found %d NaN values in final embeddings array", nan_count)
                # Replace NaN with zeros
                embeddings_array = np.nan_to_num(embeddings_array, nan=0.0)

            embeddings = torch.tensor(embeddings_array, dtype=torch.float32)
        except Exception as tensor_error:
            logger.error("Failed to convert embeddings to tensor: %s", str(tensor_error))
            raise RuntimeError(f"Failed to create embeddings tensor: {str(tensor_error)}") from tensor_error

        end_time = time.time()
        logger.info("Immune2Vec embedding completed in %.2f seconds", end_time - start_time)
        logger.info("Generated embeddings shape: %s", embeddings.shape)

        # Final validation
        if embeddings.shape[0] != len(sequences):
            logger.error("Embedding count mismatch: expected %d, got %d", len(sequences), embeddings.shape[0])
            raise RuntimeError(f"Embedding count mismatch: expected {len(sequences)}, got {embeddings.shape[0]}")

        if embeddings.shape[1] != n_dim:
            logger.error("Embedding dimension mismatch: expected %d, got %d", n_dim, embeddings.shape[1])
            raise RuntimeError(f"Embedding dimension mismatch: expected {n_dim}, got {embeddings.shape[1]}")

        return embeddings

    except Exception as e:
        logger.error("Failed to generate Immune2Vec embeddings: %s", str(e))
        logger.error("This may be due to:")
        logger.error("1. Insufficient training data (try with more sequences)")
        logger.error("2. Invalid model parameters (try different n_gram, n_dim, or window values)")
        logger.error("3. Sequence format issues (ensure sequences contain valid amino acids)")
        logger.error("4. Immune2Vec installation issues (verify the package is properly installed)")
        raise RuntimeError(f"Immune2Vec embedding failed: {str(e)}") from e
