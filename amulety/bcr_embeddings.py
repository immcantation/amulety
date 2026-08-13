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

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import MDS
from sklearn.utils.validation import check_is_fitted
from torch.nn.functional import pad

from amulety.protein_embeddings import custommodel
from amulety.utils import batch_loader, insert_space_every_other_except_cls

logger = logging.getLogger(__name__)


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


# implement vecBCRdist from vecTCRdist:
# Kept from clustcrdist.constants.hashing:
# the canonical amino-acid order and the BLOSUM62-derived tcrdist substitution-cost matrix.
AALPHABET = "ARNDCQEGHILKMFPSTWYV"
GAPCHAR = "-"
_BLOSUM_62_PATH = os.path.join(os.path.dirname(__file__), "matrices", "blosum_62.csv")
BLOSUM_62 = pd.read_csv(_BLOSUM_62_PATH, index_col=0).loc[list(AALPHABET), list(AALPHABET)].to_numpy(dtype=np.float64)
TCRDIST_DM = np.maximum(0.0, np.minimum(4.0, 4.0 - BLOSUM_62))


def setup_bcr_cdr_strings(
    bcrs: pd.DataFrame,
    cdr1_col: str = "cdr1_aa",
    cdr2_col: str = "cdr2_aa",
    cdr3_col: str = "cdr3_aa",
):
    """Replace V-gene lookup with observed BCR CDR1/2/3 strings."""
    cdr_columns = (cdr1_col, cdr2_col, cdr3_col)
    missing_columns = [column for column in cdr_columns if column not in bcrs.columns]
    if missing_columns:
        raise ValueError(
            f"Missing BCR CDR column(s): {missing_columns}. Make sure you run `amulety translate-igblast` with the option `--keep-regions` to obtain the CDR amino acid sequences."
        )

    allowed = set(AALPHABET)
    cdrs = bcrs.loc[:, cdr_columns].copy()
    for column in cdr_columns:
        if cdrs[column].isna().any():
            missing_rows = cdrs.index[cdrs[column].isna()].tolist()
            raise ValueError(f"Column '{column}' contains missing values at rows {missing_rows}")
        cdrs[column] = cdrs[column].astype(str).str.upper()
        for row_index, sequence in cdrs[column].items():
            invalid = sorted(set(sequence) - allowed)
            if not sequence:
                raise ValueError(f"Column '{column}' contains an empty sequence at row {row_index}")
            if invalid:
                raise ValueError(f"Column '{column}' contains unsupported amino acid(s) {invalid} at row {row_index}")
    short_cdr3_rows = cdrs.index[cdrs[cdr3_col].str.len().lt(5)].tolist()
    if short_cdr3_rows:
        raise ValueError(
            f"Column '{cdr3_col}' contains CDR3 sequence(s) of length < 5 at rows {short_cdr3_rows}; "
            "vecBCRdist requires length >= 5 because it trims three N-terminal and two C-terminal residues."
        )
    return list(cdrs.itertuples(index=False, name=None))


class BCRDistEncoder(BaseEstimator, TransformerMixin):
    """Minimal BCR adaptation of clustcrdist.encoding.TCRDistEncoder.
    Kept from clustcrdist.encoding.TCRDistEncoder:
    1. The vecTCRdist MDS, amino-acid encoding, trimming, gap placement, and square-root weighting are retained.
    2. The TCR V-gene dictionary path is replaced by observed AIRR ``cdr1_aa``, ``cdr2_aa``, and ``cdr3_aa``.
    """

    def __init__(
        self,
        distance_matrix: np.ndarray = TCRDIST_DM,
        aa_dim: int = 8,
        mds_eps: float = 1e-5,
        cdr1_num_pos: int = 16,
        cdr2_num_pos: int = 16,
        num_pos: int = 32,
        n_trim: int = 3,
        c_trim: int = 2,
        cdr3_weight: float = 3,
        cdr1_weight: float = 1,
        cdr2_weight: float = 1,
        gap_penalty: float = 4,
        cdr1_col: str = "cdr1_aa",
        cdr2_col: str = "cdr2_aa",
        cdr3_col: str = "cdr3_aa",
    ):
        self.distance_matrix = distance_matrix
        self.aa_dim = aa_dim
        self.mds_eps = mds_eps
        self.cdr1_num_pos = cdr1_num_pos
        self.cdr2_num_pos = cdr2_num_pos
        self.num_pos = num_pos
        self.n_trim = n_trim
        self.c_trim = c_trim
        self.cdr3_weight = cdr3_weight
        self.cdr1_weight = cdr1_weight
        self.cdr2_weight = cdr2_weight
        self.gap_penalty = gap_penalty
        self.cdr1_col = cdr1_col
        self.cdr2_col = cdr2_col
        self.cdr3_col = cdr3_col

    def _calc_mds_vecs(self):
        """Kept from TCRDistEncoder: embed the AA-plus-gap distance matrix."""
        mds = MDS(
            n_components=self.aa_dim,
            eps=self.mds_eps,
            dissimilarity="precomputed",
            random_state=11,
            n_init=4,
            normalized_stress=False,
        )
        return mds.fit_transform(self.dm)

    def _calc_tcrdist_aa_vectors(self, sqrt: bool = True):
        """Kept from TCRDistEncoder, with the BCRdist gap penalty."""
        self.dm = np.zeros((21, 21))
        self.dm[:20, :20] = self.distance_matrix
        self.dm[:20, 20] = self.gap_penalty
        self.dm[20, :20] = self.gap_penalty
        if sqrt:
            self.dm = np.sqrt(self.dm)
        vectors = self._calc_mds_vecs()
        return {amino_acid: vector for amino_acid, vector in zip(AALPHABET + GAPCHAR, vectors)}

    @staticmethod
    def _trim_and_gap(cdr: str, num_pos: int, n_trim: int = 0, c_trim: int = 0) -> str:
        """Generalize vecTCRdist's _trim_and_gap_cdr3 to an observed BCR CDR."""
        if len(cdr) < n_trim + c_trim:
            raise ValueError(f"Sequence '{cdr}' is too short for n_trim={n_trim} and c_trim={c_trim}")
        gappos = min(6, 3 + (len(cdr) - 5) // 2) - n_trim
        end = -c_trim if c_trim else len(cdr)
        sequence = cdr[n_trim:end]
        gappos = max(0, min(gappos, len(sequence), num_pos))
        afterlen = min(num_pos - gappos, len(sequence) - gappos)
        numgaps = max(0, num_pos - len(sequence))
        suffix = sequence[-afterlen:] if afterlen else ""
        fullseq = sequence[:gappos] + GAPCHAR * numgaps + suffix
        if len(fullseq) != num_pos:
            raise ValueError(f"Encoded CDR length is {len(fullseq)}, expected {num_pos}")
        return fullseq

    def _trim_and_gap_cdr3(self, cdr3: str) -> str:
        """Kept vecTCRdist CDR3 interface and BCRdist trim parameters."""
        return self._trim_and_gap(cdr3, self.num_pos, self.n_trim, self.c_trim)

    def _encode_sequence(self, sequence: str) -> np.ndarray:
        """Kept from TCRDistEncoder: concatenate amino-acid MDS vectors."""
        return np.concatenate([self.aa_vectors_[amino_acid] for amino_acid in sequence])

    def _gapped_encode_cdr(self, cdr: str, num_pos: int, n_trim: int = 0, c_trim: int = 0) -> np.ndarray:
        return self._encode_sequence(self._trim_and_gap(cdr, num_pos, n_trim, c_trim))

    def _gapped_encode_cdr3(self, cdr3: str) -> np.ndarray:
        """Kept vecTCRdist CDR3 interface."""
        return self._encode_sequence(self._trim_and_gap_cdr3(cdr3))

    def _gapped_encoder_bcr_cdrs(self, bcr: tuple) -> np.ndarray:
        """Replace vecTCRdist's V-gene-plus-CDR3 encoder with observed CDR1/2/3."""
        cdr1, cdr2, cdr3 = bcr
        cdr1_vec = np.sqrt(self.cdr1_weight) * self._gapped_encode_cdr(cdr1, self.cdr1_num_pos)
        cdr2_vec = np.sqrt(self.cdr2_weight) * self._gapped_encode_cdr(cdr2, self.cdr2_num_pos)
        cdr3_vec = np.sqrt(self.cdr3_weight) * self._gapped_encode_cdr3(cdr3)
        return np.concatenate([cdr1_vec, cdr2_vec, cdr3_vec])

    def _gapped_encode_bcr_cdrs(self, bcrs: pd.DataFrame) -> np.ndarray:
        """Replace _gapped_encode_tcr_chains with observed BCR CDR input."""
        bcr_tuples = setup_bcr_cdr_strings(bcrs, self.cdr1_col, self.cdr2_col, self.cdr3_col)
        vectors = np.array([self._gapped_encoder_bcr_cdrs(bcr) for bcr in bcr_tuples], dtype=np.float32)
        if vectors.shape != (len(bcr_tuples), self.m):
            raise RuntimeError("Unexpected vecBCRdist embedding shape")
        return vectors

    def fit(self, x=None, y=None):
        """Keep vecTCRdist fitting, but remove setup_gene_cdr_strings/combo_xcr.tsv."""
        del y
        self.aa_vectors_ = self._calc_tcrdist_aa_vectors()
        self.m = self.aa_dim * (self.cdr1_num_pos + self.cdr2_num_pos + self.num_pos)
        if x is not None:
            setup_bcr_cdr_strings(x, self.cdr1_col, self.cdr2_col, self.cdr3_col)
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        """Replace vecTCRdist's V-gene/junction DataFrame transform branch."""
        check_is_fitted(self, ("aa_vectors_", "m"))
        if not isinstance(x, pd.DataFrame):
            raise TypeError("BCRDistEncoder expects an AIRR pandas DataFrame")
        return self._gapped_encode_bcr_cdrs(x)


def vec_bcrdist(cdr_sequences: pd.DataFrame, aa_dim: int = 8):
    """Embed observed BCR CDR1/2/3 using the BCR adaptation of vecTCRdist."""
    encoder = BCRDistEncoder(aa_dim=aa_dim)
    embeddings = encoder.fit_transform(cdr_sequences)
    return torch.from_numpy(embeddings)
