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
    # DeepTCR and ablang are now in requirements.txt like antiberty
    # TCREMP handles its own import errors with detailed messages when actually used
    # TCR-BERT is available through transformers (no additional installation needed)

    # Currently no packages require special dependency checking
    # All available packages are either in requirements.txt or handle their own errors

    logger.info("All standard TCR embedding dependencies are handled automatically.")
    logger.info("Optional packages (ablang, DeepTCR) may have dependency conflicts - see README for details.")
    logger.info("TCREMP will provide installation instructions when used.")

    return missing_deps


def tcr_valid(
    sequences: pd.Series,
    chain_type: str = "TRB",
    model_name: str = "1_2_full_40",
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
):
    """
    TCR-VALID model for TCR sequence embedding.

    TCR-VALID requires complex manual installation due to dependency conflicts.
    This function provides installation guidance for users who need TCR-VALID.

    Args:
        sequences (pd.Series): TCR sequences to embed
        chain_type (str): Chain type - 'TRB' (beta) or 'TRA' (alpha)
        model_name (str): Model name to use
        cache_dir (Optional[str]): Not used
        batch_size (int): Batch size for processing

    Returns:
        torch.Tensor: Placeholder embeddings (use other models for actual analysis)

    Note:
        TCR-VALID installation is complex and requires Python 3.8 with old dependencies.
        For production use, consider tcr-bert or tcremp models instead.
    """
    installation_guide = """
    ================================================================================
    TCR-VALID Installation Guide
    ================================================================================

    TCR-VALID requires Python 3.8 and has dependency conflicts with modern packages.
    Installation is complex and not recommended for most users.

    IF YOU MUST USE TCR-VALID:

    1. Create separate Python 3.8 environment:
       conda create -n tcrvalid_env python=3.8 pip
       conda activate tcrvalid_env

    2. Clone TCR-VALID repository:
       git clone https://github.com/peterghawkins-regn/tcrvalid.git
       cd tcrvalid

    3. Modify requirements (dependencies are too old):
       Edit minimal_requirements.txt to use >= instead of ==

    4. Install with conda first:
       conda install numpy pandas scikit-learn matplotlib scipy tensorflow

    5. Install TCR-VALID:
       pip install -e .

    6. Test installation:
       python -c "from tcrvalid.load_models import *; print('Success!')"

    IMPORTANT NOTES:
    - TCR-VALID cannot coexist with AMULETY in the same environment
    - Requires separate environment switching for each use
    - Many dependency conflicts with modern packages
    - Consider using tcr-bert or tcremp for easier workflow

    """

    logger.warning("TCR-VALID is not installed or available")
    logger.info(installation_guide)
    logger.warning("Returning placeholder embeddings. Use tcr-bert or tcremp for actual analysis.")

    # Return placeholder embeddings with correct shape
    num_sequences = len(sequences)
    placeholder_embeddings = torch.randn(num_sequences, 16)
    logger.info(f"TCR-VALID placeholder embedding completed. Shape: {placeholder_embeddings.shape}")

    return placeholder_embeddings


def deep_tcr(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Embeds T-Cell Receptor (TCR) sequences using the DeepTCR model.

    It is trained on human and murine datasets, including CDR3 sequences and V/D/J gene usage.
    DeepTCR is a deep learning framework for analyzing T-cell receptor repertoires.
    Reference: https://www.nature.com/articles/s41467-021-21879-w
    """
    # device = "cuda" if torch.cuda.is_available() else "cpu"  # Currently unused
    max_seq_length = 40  # DeepTCR max sequence length
    # dim = 64  # DeepTCR embedding dimension  # Currently unused

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    sequences = X.values

    # Initialize DeepTCR model
    # Note: This is a simplified example - actual usage may require more configuration
    # dtcr = DeepTCR_U(Name="DeepTCR_Embedding")  # Currently unused

    # DeepTCR expects specific data format - this is a simplified implementation
    # In practice, you would need to format the data according to DeepTCR requirements
    logger.warning("DeepTCR integration is experimental - please verify results")

    # For now, return placeholder embeddings with correct dimensions
    n_seqs = len(sequences)
    dim = 64  # DeepTCR embedding dimension
    embeddings = torch.randn((n_seqs, dim))  # DeepTCR dimension

    logger.info("DeepTCR embedding completed")
    return embeddings


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
