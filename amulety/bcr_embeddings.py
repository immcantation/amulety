"""
BCR embedding functions using various models.
"""
# Please order alphabetically by function name.
# ruff: noqa: N806

import logging
import math
import os
import subprocess
import time
from typing import Optional

import pandas as pd
import torch
from torch.nn.functional import pad

from amulety.protein_embeddings import custommodel
from amulety.utils import batch_loader, insert_space_every_other_except_cls

logger = logging.getLogger(__name__)


def _patch_antiberty_for_new_transformers():
    """
    Monkey-patch AntiBERTy for compatibility with newer transformers versions.

    Two version-gated fixes are applied:

    Fix 1 (transformers >= 4.52): Replace self.init_weights() with self.post_init()
    in AntiBERTy.__init__.  Without this, newer transformers raises:
        AttributeError: 'AntiBERTy' object has no attribute 'all_tied_weights_keys'
    See: https://github.com/jeffreyruffolo/AntiBERTy/issues/4

    Fix 2 (transformers >= 4.36): Force attn_implementation="eager" in
    AntiBERTyRunner.  SDPA became the default attention backend for BERT in
    4.36 and produces numerically different embeddings; eager must be
    requested explicitly to preserve the original outputs.
    """
    import transformers
    from packaging.version import Version

    tv = Version(transformers.__version__)
    needs_post_init_fix = tv >= Version("4.52")
    needs_eager_fix = tv >= Version("4.36")

    if not needs_post_init_fix and not needs_eager_fix:
        return  # Nothing to patch for this transformers version

    try:
        from antiberty.AntiBERTyRunner import CHECKPOINT_PATH, VOCAB_FILE, AntiBERTyRunner
        from antiberty.model.AntiBERTy import AntiBERTy, AntiBERTyHeads
    except ImportError:
        return  # antiberty not installed; the caller will raise a clearer error

    if getattr(AntiBERTy, "_amulety_patched", False):
        return  # Already patched in this process

    if needs_post_init_fix:
        # Fix 1: Replace init_weights() with post_init() in AntiBERTy.__init__.
        def _patched_antiberty_init(self, config):
            super(AntiBERTy, self).__init__(config)
            from transformers.models.bert.modeling_bert import BertModel

            self.bert = BertModel(config)
            self.cls = AntiBERTyHeads(config)
            self.num_species = 6
            self.num_chains = 2
            self.num_grafts = 2
            self.post_init()

        AntiBERTy.__init__ = _patched_antiberty_init

    AntiBERTy._amulety_patched = True

    if needs_eager_fix:
        # Fix 2: Force eager attention in AntiBERTyRunner.__init__.
        def _patched_runner_init(self):
            import torch as _torch
            import transformers as _transformers

            self.device = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
            self.model = AntiBERTy.from_pretrained(CHECKPOINT_PATH, attn_implementation="eager").to(self.device)
            self.model.eval()
            self.tokenizer = _transformers.BertTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)

        AntiBERTyRunner.__init__ = _patched_runner_init


def antiberty(
    sequences,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
    residue_level: bool = False,
):
    """
    Embeds sequences using the AntiBERTy model.
    The maximum length of the sequences to be embedded is 510.

    Parameters:
        sequences: pd.Series for single chain or pd.DataFrame for H+L mode
    """
    from antiberty import AntiBERTyRunner

    _patch_antiberty_for_new_transformers()

    max_seq_length = 510

    # Handle both Series (single chain) and DataFrame (H+L) inputs
    if isinstance(sequences, pd.DataFrame):
        # H+L mode: DataFrame contains rows with 'chain' column indicating H or L
        if "chain" in sequences.columns:
            # Use the sequence column that the user provides
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
        # Single chain mode - sequences is a Series
        X = sequences.apply(lambda a: str(a)[:max_seq_length])

    X = X.str.replace("<cls><cls>", "[CLS][CLS]")
    X = X.apply(insert_space_every_other_except_cls)
    sequences_processed = X.str.replace("  ", " ")

    antiberty_runner = AntiBERTyRunner()
    model_size = sum(p.numel() for p in antiberty_runner.model.parameters())
    logger.info("AntiBERTy loaded. Size: %s M", round(model_size / 1e6, 2))
    start_time = time.time()
    n_seqs = len(sequences_processed)
    dim = max_seq_length + 2

    n_batches = math.ceil(n_seqs / batch_size)

    if residue_level:
        embeddings = torch.empty((n_seqs, max_seq_length, dim))
    else:
        embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences_processed, batch_size):
        logger.info("Batch %s/%s", i, n_batches)
        x = antiberty_runner.embed(batch)
        if not residue_level:
            x_keep = [a.mean(axis=0) for a in x]
        if residue_level:
            x_keep = []
            for a in x:
                if a.shape[0] < max_seq_length:
                    a_pad = pad(a.clone().detach(), (0, 0, 0, max_seq_length - a.shape[0]))
                    x_keep.append(a_pad)
        embeddings[start:end] = torch.stack(x_keep)
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))
    return embeddings


def ablang(
    sequences,
    batch_size: int = 50,
    residue_level: bool = False,
):
    """
    Embeds antibody sequences using the AbLang model.

    Note:\n
    AbLang consists of two models: one for heavy chains and one for light chains.
    Each AbLang model has two parts: AbRep (creates representations) and AbHead (predicts amino acids).
    Trained on antibody sequences in the OAS database, demonstrating power in restoring missing residues.
    This is a key capability for B-cell receptor repertoire sequencing data.
    Maximum sequence length: 160 amino acids.
    Reference: https://github.com/oxpig/AbLang

    Parameters:
        sequences: pd.Series for single chain or pd.DataFrame for H+L mode
        batch_size: int: Number of sequences to process in each batch.
        residue_level: bool: If True, returns residue-level embeddings.
    """
    import sys

    # Guard for Python version incompatibility (numba dependency unsupported on >=3.14)
    if sys.version_info >= (3, 14):
        raise ImportError(
            "AbLang is unavailable on Python 3.14+ due to numba version constraints. "
            "Please reinstall amulety under Python <3.14 to include the AbLang model."
        )
    try:
        import ablang  # noqa: F401
    except ImportError as e:
        raise ImportError("AbLang is not installed. Install with: pip install ablang (requires Python <3.14).") from e

    max_seq_length = 160

    # Handle both Series (single chain) and DataFrame (H+L) inputs
    if isinstance(sequences, pd.DataFrame):
        # H+L mode: DataFrame contains rows with 'chain' column indicating H or L
        if "chain" in sequences.columns:
            # Use the sequence column that the user provides
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

            # Process sequences and track chain types
            sequences_data = []
            for _, row in sequences.iterrows():
                seq = str(row[sequence_col])[:max_seq_length].upper()  # Convert to uppercase for AbLang
                seq = seq.replace("<CLS><CLS>", "")  # Remove CLS tokens that AbLang doesn't understand
                chain_type = row["chain"]
                sequences_data.append((seq, chain_type))
        else:
            raise ValueError("DataFrame input must contain 'chain' column for H+L mode")
    else:
        # Single chain mode - assume heavy chain as default
        sequences_data = [
            (str(seq)[:max_seq_length].upper().replace("<CLS><CLS>", ""), "H") for seq in sequences
        ]  # Convert to uppercase and remove CLS tokens for AbLang

    # Initialize AbLang models
    heavy_ablang = None
    light_ablang = None

    # Load models based on chain types present
    chain_types = set(chain_type for _, chain_type in sequences_data)

    if "H" in chain_types:
        heavy_ablang = ablang.pretrained("heavy")
        heavy_ablang.freeze()
        logger.info("AbLang heavy chain model loaded")

    if "L" in chain_types:
        light_ablang = ablang.pretrained("light")
        light_ablang.freeze()
        logger.info("AbLang light chain model loaded")

    # Generate embeddings using appropriate models
    embeddings_list = []
    start_time = time.time()
    n_seqs = len(sequences_data)
    n_batches = math.ceil(n_seqs / batch_size)

    # Process in batches for efficiency
    i = 1
    for start_idx in range(0, n_seqs, batch_size):
        end_idx = min(start_idx + batch_size, n_seqs)
        batch_data = sequences_data[start_idx:end_idx]

        logger.info("Batch %s/%s", i, n_batches)

        # Group sequences by chain type for batch processing
        heavy_seqs = []
        light_seqs = []
        heavy_indices = []
        light_indices = []

        for idx, (seq, chain_type) in enumerate(batch_data):
            if chain_type == "H":
                heavy_seqs.append(seq)
                heavy_indices.append(idx)
            else:  # "L" or other chain types
                light_seqs.append(seq)
                light_indices.append(idx)

        # Initialize batch embeddings list with None placeholders
        batch_embeddings = [None] * len(batch_data)

        # Helper function to process embeddings with padding
        def process_embeddings(embeddings, indices, apply_padding):
            for idx, seq_embedding in zip(indices, embeddings):
                seq_tensor = torch.tensor(seq_embedding)
                if apply_padding and seq_tensor.shape[0] < max_seq_length + 2:
                    pad_length = max_seq_length + 2 - seq_tensor.shape[0]
                    seq_tensor = pad(seq_tensor, (0, 0, 0, pad_length))
                batch_embeddings[idx] = seq_tensor

        # Process heavy chain sequences in batch
        if heavy_seqs and heavy_ablang is not None:
            mode = "rescoding" if residue_level else "seqcoding"
            heavy_embeddings = heavy_ablang(heavy_seqs, mode=mode)
            process_embeddings(heavy_embeddings, heavy_indices, residue_level)

        # Process light chain sequences in batch
        if light_seqs and light_ablang is not None:
            mode = "rescoding" if residue_level else "seqcoding"
            light_embeddings = light_ablang(light_seqs, mode=mode)
            process_embeddings(light_embeddings, light_indices, residue_level)
        # Handle fallback case: if light sequences exist but no light model, use heavy model
        elif light_seqs and heavy_ablang is not None:
            mode = "rescoding" if residue_level else "seqcoding"
            fallback_embeddings = heavy_ablang(light_seqs, mode=mode)
            process_embeddings(fallback_embeddings, light_indices, residue_level)

        embeddings_list.extend(batch_embeddings)
        i += 1

    embeddings = torch.stack(embeddings_list)
    end_time = time.time()
    logger.info("AbLang embedding completed. Took %s seconds", round(end_time - start_time, 2))
    return embeddings


def antiberta2(
    sequences,
    cache_dir: Optional[str] = None,
    residue_level: bool = False,
    batch_size: int = 50,
):
    """
    Embeds sequences using the antiBERTa2 RoFormer model.
    The maximum length of the sequences to be embedded is 256.

    Parameters:
        sequences: pd.Series for single chain or pd.DataFrame for H+L mode
        cache_dir: Optional[str]: Directory to cache the model files.
        residue_level: bool: If True, returns residue-level embeddings.
        batch_size: int: Number of sequences to process in each batch.
    """

    from transformers import RoFormerForMaskedLM, RoFormerTokenizer

    max_seq_length = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Handle both Series (single chain) and DataFrame (H+L) inputs
    if isinstance(sequences, pd.DataFrame):
        # H+L mode: DataFrame contains rows with 'chain' column indicating H or L
        if "chain" in sequences.columns:
            # Use the sequence column that the user provides
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
        # Single chain mode - sequences is a Series
        X = sequences.apply(lambda a: str(a)[:max_seq_length])

    X = X.str.replace("<cls><cls>", "[CLS][CLS]")
    X = X.apply(insert_space_every_other_except_cls)
    X = X.str.replace("  ", " ")
    sequences_array = X.values

    tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2", cache_dir=cache_dir)
    model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2", cache_dir=cache_dir)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("AntiBERTa2 loaded. Size: %s M", model_size / 1e6)

    start_time = time.time()
    n_seqs = len(sequences_array)
    dim = 1024
    n_batches = math.ceil(n_seqs / batch_size)
    if residue_level:
        embeddings = torch.empty((n_seqs, max_seq_length, dim))
    else:
        embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences_array, batch_size):
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

        # aggregate across the residuals, ignore the padded bases
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


def balm_paired(
    sequences,
    cache_dir: str = "/tmp/amulety",
    residue_level: bool = False,
    batch_size: int = 50,
):
    """
    Embeds sequences using the BALM-paired model.
    The maximum length of the sequences to be embedded is 1024.
    The embedding dimension is 1024.

    Parameters:
        sequences: pd.Series for single chain or pd.DataFrame for H+L mode
        cache_dir: Optional[str]: Directory to cache the model files.
        residue_level: bool: If True, returns residue-level embeddings.
        batch_size: int: Number of sequences to process in each batch.
    """

    os.makedirs(cache_dir, exist_ok=True)

    model_name = "BALM-paired_LC-coherence_90-5-5-split_122222"
    model_path = os.path.join(cache_dir, model_name)
    embedding_dimension = 1024
    max_seq_length = 510

    if not os.path.exists(model_path):
        try:
            command = f"""
                curl -L -o {os.path.join(cache_dir, "BALM-paired.tar.gz")} https://zenodo.org/records/8237396/files/BALM-paired.tar.gz
                tar -xzf {os.path.join(cache_dir, "BALM-paired.tar.gz")} -C {cache_dir}
                rm {os.path.join(cache_dir, "BALM-paired.tar.gz")}
            """
            subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error downloading or extracting BALM-paired model: {e}")

    embeddings = custommodel(
        sequences=sequences,
        model_path=model_path,
        embedding_dimension=embedding_dimension,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        cache_dir=cache_dir,
        residue_level=residue_level,
    )
    return embeddings
