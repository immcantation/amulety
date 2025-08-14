"""
TCR embedding functions using various models.
Please order alphabetically by function name.
"""
# ruff: noqa: N806

import logging
import math
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch

from amulety.utils import batch_loader

# Optional imports for TCR models are handled within individual functions

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_tcr_dependencies():
    """Check if optional TCR embedding dependencies are installed and provide installation instructions."""
    import subprocess

    missing_deps = []
    available_models = []

    # Check TCREMP (command-line tool)
    try:
        result = subprocess.run(["tcremp-run", "-h"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            available_models.append("TCREMP")
        else:
            missing_deps.append(
                (
                    "TCREMP",
                    "git clone https://github.com/antigenomics/tcremp.git && cd tcremp && pip install . (requires Python 3.11+)",
                )
            )
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        missing_deps.append(
            (
                "TCREMP",
                "git clone https://github.com/antigenomics/tcremp.git && cd tcremp && pip install . (requires Python 3.11+)",
            )
        )

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


def tcremp(
    sequences,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
    chain: str = "H",
    skip_clustering: bool = True,
):
    """
    Embeds T-Cell Receptor (TCR) sequences using the TCREMP model.

    Args:
        sequences: Input TCR sequences (pd.Series)
        cache_dir: Directory to cache model files (not used for TCREMP)
        batch_size: Number of sequences to process in each batch (not used for TCREMP)
        chain: Chain type specification ("H", "L", "HL", "LH", "H+L")
        skip_clustering: Whether to skip clustering step (default: True to avoid errors)

    Note:\n
    TCREMP is a command-line tool for TCR sequence embedding via prototypes.
    It focuses on T-cell receptor repertoire-based representation learning using prototype-based
    similarity calculations. TCREMP is trained on TCR sequences only (CDR3 + V/J genes).

    TCREMP supports all chain types including paired chains (HL/LH) and individual chains (H, L, H+L).
    Maximum sequence length: 30 amino acids.
    Embedding dimension: Variable, depends on number of prototypes used (default: 3000 prototypes).

    Reference: https://github.com/antigenomics/tcremp
    """
    import os
    import subprocess
    import tempfile

    max_seq_length = 30  # TCREMP max sequence length

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    sequences = X.values

    logger.info("Loading TCREMP model for TCR embedding...")

    # Check if tcremp-run command is available
    try:
        result = subprocess.run(["tcremp-run", "-h"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            raise FileNotFoundError("tcremp-run command not found")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        detailed_instructions = (
            "TCREMP command-line tool is not installed or not accessible.\n\n"
            "INSTALLATION:\n"
            "1. Ensure Python 3.11+ is installed: python --version\n"
            "2. Clone the repository: git clone https://github.com/antigenomics/tcremp.git\n"
            "3. Install: cd tcremp && pip install .\n"
            "4. Verify installation: tcremp-run -h\n\n"
            "TCREMP generates distance-based embeddings using prototype similarity.\n"
            "For direct TCR sequence embedding, consider using:\n"
            "- tcr-bert: Transformer-based TCR embedding\n"
            "- tcrt5: T5-based TCR embedding\n"
            "- esm2 or prott5: General protein language models\n\n"
            "Reference: https://github.com/antigenomics/tcremp"
        )
        raise ImportError(f"TCREMP command-line tool is required but not installed.\n\n{detailed_instructions}")

    # Create temporary files for TCREMP input and output
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = os.path.join(temp_dir, "input.txt")
        output_dir = os.path.join(temp_dir, "output")

        # Prepare input data in TCREMP format
        # TCREMP expects AIRR format with specific columns
        input_data = []
        for i, seq in enumerate(sequences):
            # Create minimal AIRR-like format for TCREMP
            # Note: This is a simplified format - real usage would need proper V/J gene annotations
            input_data.append(
                {
                    "clone_id": f"seq_{i}",
                    "junction_aa": seq,
                    "v_call": "TRBV1*01",  # Placeholder - would need real V gene annotation
                    "j_call": "TRBJ1-1*01",  # Placeholder - would need real J gene annotation
                    "locus": "beta",  # Assuming beta chain for H, alpha for L
                }
            )

        # Write input file
        input_df = pd.DataFrame(input_data)
        input_df.to_csv(input_file, sep="\t", index=False)

        try:
            # Run TCREMP command
            # Adjust parameters based on sample size to avoid dimension issues
            n_samples = len(sequences)
            # For small datasets, use minimal PCA components to avoid sklearn errors
            pca_components = min(3, max(1, n_samples - 1))  # Use minimal PCA components
            k_neighbors = min(3, max(1, n_samples - 1))  # Use minimal k-neighbors

            # Map amulety chain parameter to TCREMP chain parameter
            # H → TRB (beta), L → TRA (alpha), HL → TRA_TRB (paired)
            chain_mapping = {
                "H": "TRB",  # Heavy chain -> Beta chain
                "L": "TRA",  # Light chain -> Alpha chain
                "HL": "TRA_TRB",  # Paired chains -> Both chains
                "LH": "TRA_TRB",  # Paired chains -> Both chains
                "H+L": "TRA_TRB",  # Paired chains -> Both chains
            }
            tcremp_chain = chain_mapping.get(chain, "TRB")
            logger.info(f"Mapping amulety chain '{chain}' to TCREMP chain '{tcremp_chain}'")

            cmd = [
                "tcremp-run",
                "-i",
                input_file,
                "-c",
                tcremp_chain,  # Mapped chain parameter
                "-o",
                output_dir,
                "--prefix",
                "tcremp_output",
                "-cl",
                "False" if skip_clustering else "True",  # Control clustering based on parameter
                "-d",
                "True",  # Enable distance calculation to get embeddings
                "-npc",
                str(pca_components),  # Use sample-size adjusted PCA components
                "-kn",
                str(k_neighbors),  # Use sample-size adjusted k-neighbors
                "-n",
                "100",  # Use fewer prototypes for speed
            ]

            logger.info("Running TCREMP command-line tool...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

            # Check if TCREMP generated output files even if it failed (e.g., clustering failed but embedding succeeded)
            output_files = [f for f in os.listdir(output_dir) if f.endswith(".tsv") or f.endswith(".parquet")]

            if result.returncode != 0:
                logger.warning("TCREMP command returned non-zero exit code: %s", result.stderr)
                if not output_files:
                    logger.error("TCREMP command failed and no output files generated: %s", result.stderr)
                    raise RuntimeError(f"TCREMP execution failed: {result.stderr}")
                else:
                    logger.info(
                        "TCREMP generated output files despite error (likely clustering failed but embedding succeeded)"
                    )

            # Read TCREMP output
            # TCREMP outputs distance matrices - we need to convert to embeddings
            if not output_files:
                raise RuntimeError("TCREMP did not generate expected output files")

            output_file = os.path.join(output_dir, output_files[0])
            print(f"DEBUG: Output file: {output_file}")
            print(f"DEBUG: File exists: {os.path.exists(output_file)}")
            print(f"DEBUG: File size: {os.path.getsize(output_file) if os.path.exists(output_file) else 'N/A'}")
            with open(output_file, "rb") as debug_f:
                header = debug_f.read(20)
                print(f"DEBUG: File header: {header}")
            tcremp_output = pd.read_parquet(output_file)  # Fixed: TCREMP outputs parquet

            # Extract distance features as embeddings
            # TCREMP outputs distance columns - use these as embedding features
            distance_cols = [col for col in tcremp_output.columns if "_v" in col or "_j" in col or "_cdr3" in col]

            if not distance_cols:
                logger.warning("No distance columns found in TCREMP output, using all numeric columns")
                distance_cols = tcremp_output.select_dtypes(include=[np.number]).columns.tolist()

            embeddings_array = tcremp_output[distance_cols].values

            # Scale TCREMP distance features to reasonable range (preserve relative relationships)
            # Use min-max scaling to [0, 1] range, then center around 0
            min_val = embeddings_array.min()
            max_val = embeddings_array.max()
            embeddings_array = (embeddings_array - min_val) / (max_val - min_val)  # Scale to [0, 1]
            embeddings_array = embeddings_array - 0.5  # Center around 0, range [-0.5, 0.5]
            embeddings = torch.tensor(embeddings_array, dtype=torch.float32)

            logger.info("TCREMP embedding completed with dimension %d", embeddings.shape[1])
            return embeddings

        except subprocess.TimeoutExpired:
            raise RuntimeError("TCREMP execution timed out (>5 minutes)")
        except Exception as e:
            logger.error("TCREMP execution failed: %s", str(e))
            raise RuntimeError(f"TCREMP execution failed: {str(e)}") from e


def tcr_bert(
    sequences,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
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
    sequences,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
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
