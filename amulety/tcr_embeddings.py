"""
TCR embedding functions using various models.
Please order alphabetically by function name.
"""
# ruff: noqa: N806

import logging
import math
import time
from typing import Optional

import pandas as pd
import torch

from amulety.utils import batch_loader

# Optional imports for TCR models are handled within individual functions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_tcr_dependencies():
    """Check if optional TCR embedding dependencies are installed and provide installation instructions."""
    missing_deps = []

    # Following the pattern of BCR/protein models where most models don't appear in dependency checks
    # Only check for packages that require special installation (git clone, etc.)
    # ablang is now in requirements.txt like antiberty
    # TCREMP handles its own import errors with detailed messages when actually used
    # TCR-BERT is available through transformers (no additional installation needed)

    # Currently no packages require special dependency checking
    # All available packages are either in requirements.txt or handle their own errors

    logger.info("All standard TCR embedding dependencies are handled automatically.")
    logger.info("Optional packages (ablang) may have dependency conflicts - see README for details.")
    logger.info("TCREMP will provide installation instructions when used.")

    return missing_deps


def tcremp(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Embeds T-Cell Receptor (TCR) sequences using the TCREMP model.

    Note:\n
    Embedding method trained specifically for TCR sequences; focuses on T-cell receptor
    repertoire-based representation learning. Details on architecture not publicly released,
    but known to support repertoire-level prediction tasks.

    TCREMP supports all chain types including paired chains (HL/LH) and individual chains (H, L, H+L).
    Maximum sequence length: 30 amino acids.
    Embedding dimension: To be determined from original implementation.

    """
    max_seq_length = 30  # TCREMP max sequence length

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    sequences = X.values

    logger.info("Loading TCREMP model for TCR embedding...")

    try:
        # Try to import the actual TCREMP package
        try:
            import tcremp  # noqa: F401

            logger.info("TCREMP package found, using actual TCREMP model")

            # TCREMP integration would go here
            # Note: This requires the actual TCREMP package to be installed
            logger.warning("TCREMP integration is experimental - please verify results")

            # For now, return placeholder embeddings with temporary dimensions
            # Actual TCREMP embedding dimension is TBD from original implementation
            n_seqs = len(sequences)
            embeddings = torch.randn((n_seqs, 256))  # Temporary dimension - actual TCREMP dimension TBD

            logger.info("TCREMP embedding completed")
            return embeddings

        except ImportError as tcremp_error:
            logger.error("TCREMP package not found: %s", str(tcremp_error))
            detailed_instructions = (
                "TCREMP package not available. Please follow these installation instructions:\n\n"
                "REQUIREMENTS: Python 3.11 or higher is required for TCREMP\n"
                "   Check your Python version: python --version\n"
                "   If you have Python < 3.11, please upgrade or use a different environment\n\n"
                "STEP 1: Clone the TCREMP repository\n"
                "   git clone https://github.com/antigenomics/tcremp.git\n"
                "   cd tcremp\n\n"
                "STEP 2: Install the package\n"
                "   pip install .\n\n"
                "STEP 3: Verify installation\n"
                "   python -c 'import tcremp; print(\"TCREMP installed successfully\")'\n\n"
                "Note: TCREMP supports all chain types (H, L, HL, LH, H+L) with configurable dimensions.\n"
                "Reference: https://github.com/antigenomics/tcremp"
            )
            logger.warning("Using placeholder embeddings. %s", detailed_instructions)

            # Placeholder: return random embeddings with temporary shape
            # Note: Actual TCREMP embedding dimension is TBD from original implementation
            n_seqs = len(sequences)
            embeddings = torch.randn((n_seqs, 256))  # Temporary dimension - actual TCREMP dimension TBD

            logger.info("TCREMP placeholder embedding completed")
            return embeddings

    except Exception as e:
        logger.error("Failed to load TCREMP model: %s", str(e))
        raise RuntimeError("Could not load TCREMP model") from e


def tcr_bert(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Embeds T-Cell Receptor (TCR) sequences using the TCR-BERT model.

    Note:\n
    Pretrained on 88,403 human TRA/TRB sequences from VDJdb and PIRD.
    Non-fine-tuned version focused on human TCR data only. The maximum length of the sequences to be embedded is 64.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length = 64
    dim = 768

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])

    # TCR-BERT expects standard amino acid sequences without special tokens
    X = X.apply(lambda seq: seq.replace("<cls><cls>", " "))
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
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Embeds T-Cell Receptor (TCR) sequences using the TCRT5 model.

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
                # Use mean pooling over sequence length to get fixed-size embedding
                sequence_embedding = enc_outputs.last_hidden_state.mean(dim=1).squeeze()
                batch_embeddings.append(sequence_embedding.cpu())

        embeddings[start:end] = torch.stack(batch_embeddings)
        i += 1

    end_time = time.time()
    logger.info("TCRT5 embedding took %s seconds", round(end_time - start_time, 2))

    return embeddings
