"""
TCR embedding functions using various models.
"""
# ruff: noqa: N806
# Please order alphabetically by function name.

import logging
import math
import time
from typing import Optional

import pandas as pd
import torch

from amulety.utils import batch_loader

# Optional imports for TCR models are handled within individual functions

logger = logging.getLogger(__name__)


def check_tcr_dependencies():
    """Check if optional TCR embedding dependencies are installed and provide installation instructions."""

    missing_deps = []
    available_models = []

    # Check TCR-BERT (requires transformers)
    try:
        from transformers import BertModel, BertTokenizer  # noqa: F401

        available_models.append("TCR-BERT")
    except ImportError:
        missing_deps.append(("TCR-BERT", "pip install transformers"))

    # Check TCRT5 (requires transformers)
    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer  # noqa: F401

        available_models.append("TCRT5")
    except ImportError:
        missing_deps.append(("TCRT5", "pip install transformers"))

    # Check Immune2Vec (in protein_embeddings but used for TCR too)
    try:
        import gensim  # noqa: F401
        from embedding import sequence_modeling  # noqa: F401

        available_models.append("Immune2Vec")
    except ImportError as e:
        if "gensim" in str(e):
            missing_deps.append(
                (
                    "Immune2Vec",
                    "pip install gensim>=3.8.3 && git clone https://bitbucket.org/yaarilab/immune2vec_model.git",
                )
            )
        else:
            missing_deps.append(
                ("Immune2Vec", "git clone https://bitbucket.org/yaarilab/immune2vec_model.git && add to Python path")
            )

    # Report results
    if available_models:
        logger.info("Available TCR models: %s", ", ".join(available_models))

    if missing_deps:
        logger.warning("Missing TCR model dependencies: %s", ", ".join([dep[0] for dep in missing_deps]))
    else:
        logger.info("All TCR embedding dependencies are available!")

    return missing_deps


def tcr_bert(
    sequences,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    residue_level: bool = False,
):
    """
    Embeds T-Cell Receptor (TCR) sequences using the TCR-BERT model.

    Args:
        sequences: Input TCR sequences (pd.Series for single chain or pd.DataFrame for H+L mode)
        cache_dir: Directory to cache model files
        batch_size: Number of sequences to process in each batch

    Note:\n
    Pretrained on 88,403 human TRA/TRB sequences from VDJdb and PIRD.
    Non-fine-tuned version focused on human TCR data only. The maximum length of the sequences to be embedded is 64.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length = 64
    dim = 768

    # Handle both Series (single chain) and DataFrame (H+L) inputs
    if isinstance(sequences, pd.DataFrame):
        # DataFrame mode: Extract sequence data from the 'chain' column
        if "chain" in sequences.columns:
            X = sequences["chain"]
        else:
            # Fallback: look for common sequence column names
            sequence_col_candidates = ["sequence_vdj_aa", "sequence_aa", "sequence"]
            sequence_col = None
            for col in sequence_col_candidates:
                if col in sequences.columns:
                    sequence_col = col
                    break
            if sequence_col is None:
                raise ValueError(
                    f"No sequence column found in DataFrame. Expected 'chain' or one of {sequence_col_candidates}"
                )
            X = sequences[sequence_col]
    else:
        # Series mode: direct sequence input
        X = sequences

    X = X.apply(lambda a: a[:max_seq_length])

    # TCR-BERT expects space-separated amino acid sequences
    # Convert "CASSLAPGATNEKLFF" to "C A S S L A P G A T N E K L F F"
    X = X.apply(lambda seq: seq.replace("<cls><cls>", " "))  # Remove any existing special tokens
    X = X.apply(lambda seq: " ".join(list(seq)))  # Space-separate amino acids
    sequences = X.values

    logger.info("Loading TCR-BERT model for TCR embedding...")

    try:
        from transformers import BertModel, BertTokenizer

        # non-fine-tuned TCR-BERT model pre-trained on human data only
        model_name = "wukevin/tcr-bert"

        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, do_lower_case=False)
        model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        logger.info("Successfully loaded TCR-BERT model")

    except Exception as e:
        logger.warning("TCR-BERT model not available, using BERT-base as fallback: %s", str(e))
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
            model = BertModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
            logger.info("Using BERT-base as fallback for TCR-BERT")
        except Exception as e2:
            logger.error("Failed to load any BERT model: %s", str(e2))
            raise RuntimeError("Could not load TCR-BERT or fallback model") from e2

    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("TCR-BERT model loaded. Size: %s M", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)

    if residue_level:
        embeddings = torch.empty((n_seqs, max_seq_length, dim))
    else:
        embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("TCR-BERT Batch %s/%s.", i, n_batches)

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
            outputs = model(input_ids=x, attention_mask=attention_mask)
            outputs = outputs.last_hidden_state
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
    logger.info("TCR-BERT embedding took %s seconds", round(end_time - start_time, 2))

    return embeddings


def tcrt5(
    sequences,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    residue_level: bool = False,
):
    """
    Embeds T-Cell Receptor (TCR) sequences using the TCRT5 model.

    Args:
        sequences: Input TCR sequences (pd.Series for single chain or pd.DataFrame for H+L mode)
        cache_dir: Directory to cache model files
        batch_size: Number of sequences to process in each batch

    Note:\n
    TCRT5 was pre-trained on masked span reconstruction using ~14M CDR3 β sequences from TCRdb
    and ~780k peptide-pseudosequence pairs from IEDB. This model only supports beta chains (H chains for TCR).
    Maximum sequence length: 20 amino acids.
    Embedding dimension: 256.

    Reference: https://huggingface.co/dkarthikeyan1/tcrt5_pre_tcrdb
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length = 20
    dim = 256

    # Handle both Series (single chain) and DataFrame (H+L) inputs
    if isinstance(sequences, pd.DataFrame):
        # DataFrame mode: Extract sequence data from the 'chain' column
        if "chain" in sequences.columns:
            X = sequences["chain"]
        else:
            # Fallback: look for common sequence column names
            sequence_col_candidates = ["sequence_vdj_aa", "sequence_aa", "sequence"]
            sequence_col = None
            for col in sequence_col_candidates:
                if col in sequences.columns:
                    sequence_col = col
                    break
            if sequence_col is None:
                raise ValueError(
                    f"No sequence column found in DataFrame. Expected 'chain' or one of {sequence_col_candidates}"
                )
            X = sequences[sequence_col]
    else:
        # Series mode: direct sequence input
        X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    sequences = X.values

    logger.info("Loading TCRT5 model for TCR embedding...")
    start_time = time.time()

    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        model_name = "dkarthikeyan1/tcrt5_pre_tcrdb"
        tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=cache_dir)
        model.to(device)
        model.eval()

    except ImportError as e:
        logger.error("transformers library not found: %s", str(e))
        raise ImportError("transformers library is required for TCRT5. Install with: pip install transformers") from e
    except Exception as e:
        logger.error("Failed to load TCRT5 model: %s", str(e))
        raise RuntimeError("Could not load TCRT5 model") from e

    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)
    if residue_level:
        embeddings = torch.zeros((n_seqs, max_seq_length, dim))
    else:
        embeddings = torch.zeros((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("TCRT5 Batch %s/%s.", i, n_batches)

        # Process each sequence in the batch
        batch_embeddings = []
        for seq in batch:
            # Format sequence for TCRT5 (just the sequence without PMHC format for CDR3 embedding)
            encoded_seq = tokenizer(seq, return_tensors="pt", truncation=True, max_length=max_seq_length).to(device)

            with torch.no_grad():
                # Use encoder outputs for embedding representation
                enc_outputs = model.encoder(**encoded_seq)
                if not residue_level:
                    # Use mean pooling over sequence length to get fixed-size embedding
                    sequence_embedding = enc_outputs.last_hidden_state.mean(dim=1).squeeze()
                else:
                    sequence_embedding = enc_outputs.last_hidden_state.squeeze()
                    if sequence_embedding.shape[0] < max_seq_length:
                        # Pad with zeros if needed
                        padding = torch.zeros((max_seq_length - sequence_embedding.shape[0], dim))
                        sequence_embedding = torch.cat((sequence_embedding, padding), dim=0)
                batch_embeddings.append(sequence_embedding.cpu())

        embeddings[start:end] = torch.stack(batch_embeddings)
        i += 1

    end_time = time.time()
    logger.info("TCRT5 embedding took %s seconds", round(end_time - start_time, 2))

    return embeddings
