"""Main module."""
import logging
import os
from typing import Iterable

import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def batch_loader(data: Iterable, batch_size: int):
    """
    This function generates batches from the provided data.

    Parameters:
    data (Iterable): The data to be batched.
    batch_size (int): The size of each batch.

    Yields:
    tuple: A tuple containing the start index, end index, and the batch of data.
    """
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]


def insert_space_every_other_except_cls(input_string: str):
    """
    This function inserts a space after every character in the input string, except for the '[CLS]' token.

    Parameters:
    input_string (str): The input string where spaces are to be inserted.

    Returns:
    str: The modified string with spaces inserted.
    """
    parts = input_string.split("[CLS]")
    modified_parts = ["".join([char + " " for char in part]).strip() for part in parts]
    result = " [CLS] ".join(modified_parts)
    return result


def check_output_file_type(outpath: str):
    """
    Checks if the output file type specified in the given file path is one of the allowed types.

    Parameters:
    outpath (str): The file path of the output file whose type needs to be checked.
    The output suffix should be one of the following:
    - 'pt': PyTorch binary format
    - 'tsv': Tab-separated values format
    - 'csv': Comma-separated values format

    Returns:
    str: The file extension of the output file if it is one of the allowed types.

    Raises:
    ValueError: If the file extension is not one of the allowed types ('tsv', 'csv', 'pt').
    """
    out_format = os.path.splitext(outpath)[-1][1:]
    allowed_outputs = ["tsv", "csv", "pt"]
    if out_format not in allowed_outputs:
        raise ValueError(f"Output suffix must be one of {allowed_outputs}")
    return out_format


def save_embedding(dat, embedding, outpath, outformat, cell_id_col):
    """
    Saves the embedding data to a specified file path in the desired format.

    Args:
        dat (DataFrame): The original DataFrame containing index columns and possibly other data.
        embedding (Tensor): The embedding data to be saved.
        outpath (str): The file path where the embedding data will be saved.
        cell_id_col (str): The name of the column containing the single-cell barcode.

    Raises:
        ValueError: If the output format is not supported.

    Note:
        Index columns from the original DataFrame 'dat' will be included in the saved output.

    Example:
        save_embedding(dat, embeddings, "embedding.tsv", "cell_id")
    """
    allowed_index_cols = ["sequence_id", cell_id_col]
    index_cols = [col for col in dat.columns if col in allowed_index_cols]
    if outformat == "pt":
        torch.save(embedding, outpath)
    elif outformat in ["tsv", "csv"]:
        embedding_df = pd.DataFrame(embedding.numpy())
        result_df = pd.concat([dat.loc[:, index_cols].reset_index(drop=True), embedding_df], axis=1)
        sep = "\t" if outformat == "tsv" else ","
        result_df.to_csv(outpath, sep=sep, index=False)


def process_airr(inpath: str, chain: str, sequence_col: str = "sequence_vdj_aa", cell_id_col: str = "cell_id"):
    """
    Processes AIRR-seq data from the input file path and returns a pandas DataFrame containing the sequence to embed.
    It will drop cells with missing heavy or light chain if operating in single-cell only mode (no cell IDs missing) and log the number of missing chains.\n
    If the data is bulk only, it will raise an error if chain = "HL".\n
    If the data is mixed bulk and single-cell, and the mode is HL it will concatenate heavy and light chains per cell and drop cells with missing chains.\n

    Parameters:
        inpath (str): The file path to the input data.
        chain (str): The input chain, which can be one of ["H", "L", "HL"].
        sequence_col (str): The name of the column containing the amino acid sequences to embed.
        cell_id_col (str): The name of the column containing the single-cell barcode.

    Returns:
        pandas.DataFrame: Dataframe with formatted sequences.

    Raises:
        ValueError: If chain is not one of ["H", "L", "HL"].
    """
    allowed_sequence_input = ["H", "L", "HL"]
    if chain not in allowed_sequence_input:
        raise ValueError(f"Input x must be one of {allowed_sequence_input}")

    data = pd.read_table(inpath)
    if "locus" not in data.columns:
        data.loc[:, "locus"] = data.loc[:, "v_call"].apply(lambda x: x[:3])

    # ===== BCR CHAIN MAPPING (ORIGINAL CODE) =====
    data.loc[:, "chain"] = data.loc[:, "locus"].apply(lambda x: "H" if x in ["IGH", "TRB", "TRD"] else "L")

    if cell_id_col not in data.columns:
        data_type = "bulk-only"
    elif data[cell_id_col].notna().all():
        data_type = "single-cell-only"
    else:
        data_type = "mixed"

    if data_type == "bulk-only":
        logger.info(
            "No %s column detected. Processing as bulk data. If the data is single-cell, please specify cell_id_col for the barcode column.",
            cell_id_col,
        )
        if chain == "HL":
            raise ValueError('chain = "HL" invalid for bulk mode.')
        else:
            colnames = ["sequence_id", sequence_col]
            data = data.loc[data.chain == chain, colnames]

    elif data_type == "single-cell-only":
        logger.info("Processing single-cell BCR data...")
        if chain == "HL":
            logging.info("Concatenating heavy and light chain per cell...")
            data = concatenate_heavylight(data, sequence_col, cell_id_col)
        else:
            colnames = [cell_id_col, sequence_col]
            data = data.loc[data.chain == chain, colnames]

    elif data_type == "mixed":
        logger.info("Missing values in %s column. Processing as mixed bulk and single-cell BCR data...", cell_id_col)
        if chain == "HL":
            logger.info("Concatenating heavy and light chain per cell...")
            data = data.loc[data[cell_id_col].notna(),]
            data = concatenate_heavylight(data, sequence_col, cell_id_col)
        else:
            colnames = ["sequence_id", cell_id_col, sequence_col]
            data = data.loc[data.chain == chain, colnames]

    return data


def concatenate_heavylight(data: pd.DataFrame, sequence_col: str, cell_id_col: str):
    """
    Concatenates heavy and light chain per cell and returns a pandas DataFrame.\n
    If a cell contains several light or heavy chains, it will take the one with highest duplicate count.\n


    Parameters:
        data (pandas.DataFrame): Input data containing information about heavy and light chains.
        sequence_col (str): The name of the column containing the amino acid sequences to embed.
        cell_id_col (str): The name of the column containing the single-cell barcode.

    Returns:
        pandas.DataFrame: Dataframe with concatenated heavy and light chains per cell.
    """
    colnames = [cell_id_col, "locus", "duplicate_count", sequence_col]
    missing_cols = [col for col in colnames if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Column(s) {missing_cols} is/are not present in the input data and are needed to concatenate heavy and light chains."
        )

    # if tie in maximum duplicate_count, return the first occurrence
    data = data.loc[data.groupby([cell_id_col, "chain"])["duplicate_count"].idxmax()]
    data = data.pivot(index=cell_id_col, columns="chain", values=sequence_col)
    data = data.reset_index(level=cell_id_col)
    n_cells = data.shape[0]
    data = data.dropna(axis=0)
    n_dropped = n_cells - data.shape[0]
    if n_dropped > 0:
        logging.info("Dropping %s cells with missing heavy or light chain...", n_dropped)
    data.loc[:, sequence_col] = data.H + "<cls><cls>" + data.L
    return data


# ========================================
# ===== TCR-SPECIFIC FUNCTIONS START =====
# ========================================

def process_tcr_airr(inpath: str, chain: str, sequence_col: str = "sequence_vdj_aa", cell_id_col: str = "cell_id"):
    """
    Processes AIRR-seq data specifically for TCR (T-Cell Receptor) sequences and returns a pandas DataFrame containing the sequence to embed.

    This function handles TCR data similarly to BCR processing but with TCR-specific chain recognition:
    - TRA/TRG loci are mapped to alpha chains (A)
    - TRB/TRD loci are mapped to beta chains (B)
    - Supports single-cell, bulk, and mixed data types
    - For AB mode: concatenates alpha and beta chains per cell and drops cells with missing chains
    - For bulk data: raises error if chain = "AB" since pairing requires cell information

    Compatible with standard AIRR format files containing TCR repertoire data from:
    - 10X Genomics single-cell TCR-seq
    - Bulk TCR-seq datasets
    - Mixed single-cell and bulk datasets
    - Custom AIRR-formatted TCR data

    Parameters:
        inpath (str): The file path to the input AIRR data file (TSV format).
        chain (str): The input chain type - "A" (alpha), "B" (beta), or "AB" (alpha-beta concatenated).
        sequence_col (str): The name of the column containing the amino acid sequences to embed (default: "sequence_vdj_aa").
        cell_id_col (str): The name of the column containing the single-cell barcode (default: "cell_id").

    Returns:
        pandas.DataFrame: Dataframe with formatted TCR sequences ready for embedding.

    Raises:
        ValueError: If chain is not one of ["A", "B", "AB"] or if AB mode is used with bulk-only data.

    Example:
        # Process alpha chains only
        alpha_data = process_tcr_airr("tcr_data.tsv", "A")

        # Process alpha-beta pairs from single-cell data
        paired_data = process_tcr_airr("tcr_data.tsv", "AB", cell_id_col="barcode")
    """
    allowed_sequence_input = ["A", "B", "AB"]
    if chain not in allowed_sequence_input:
        raise ValueError(f"TCR chain input must be one of {allowed_sequence_input}")

    data = pd.read_table(inpath)
    if "locus" not in data.columns:
        data.loc[:, "locus"] = data.loc[:, "v_call"].apply(lambda x: x[:3])

    # ===== TCR CHAIN MAPPING =====
    # Map TCR loci to simplified chain types: A for alpha chains, B for beta chains
    data.loc[:, "tcr_chain"] = data.loc[:, "locus"].apply(
        lambda x: "B" if x in ["TRB", "TRD"] else (
            "A" if x in ["TRA", "TRG"] else None
        )
    )

    # Filter for TCR chains only
    data = data[data.tcr_chain.notna()]
    logger.info("Filtered %s TCR sequences", data.shape[0])

    if cell_id_col not in data.columns:
        data_type = "bulk-only"
    elif data[cell_id_col].notna().all():
        data_type = "single-cell-only"
    else:
        data_type = "mixed"

    if data_type == "bulk-only":
        logger.info(
            "No %s column detected. Processing as bulk TCR data. If the data is single-cell, please specify cell_id_col for the barcode column.",
            cell_id_col,
        )
        if chain == "AB":
            raise ValueError('chain = "AB" invalid for bulk mode.')
        else:
            colnames = ["sequence_id", sequence_col]
            data = data.loc[data.tcr_chain == chain, colnames]

    elif data_type == "single-cell-only":
        logger.info("Processing single-cell TCR data...")
        if chain == "AB":
            logger.info("Concatenating alpha and beta chain per cell...")
            data = concatenate_alphabeta(data, sequence_col, cell_id_col)
        else:
            colnames = [cell_id_col, sequence_col]
            data = data.loc[data.tcr_chain == chain, colnames]

    elif data_type == "mixed":
        logger.info("Missing values in %s column. Processing as mixed bulk and single-cell TCR data...", cell_id_col)
        if chain == "AB":
            logger.info("Concatenating alpha and beta chain per cell...")
            data = data.loc[data[cell_id_col].notna(),]
            data = concatenate_alphabeta(data, sequence_col, cell_id_col)
        else:
            colnames = ["sequence_id", cell_id_col, sequence_col]
            data = data.loc[data.tcr_chain == chain, colnames]

    return data


def concatenate_alphabeta(data: pd.DataFrame, sequence_col: str, cell_id_col: str):
    """
    Concatenates TCR alpha and beta chains per cell and returns a pandas DataFrame.

    This function pairs TCR alpha and beta chains from the same cell, similar to how
    concatenate_heavylight pairs BCR heavy and light chains. The concatenation format
    follows the pattern: BETA_SEQUENCE<cls><cls>ALPHA_SEQUENCE

    Key features:
    - Handles multiple chains per cell by selecting the one with highest duplicate count
    - Drops cells that are missing either alpha or beta chains
    - Maintains cell-level information for downstream analysis
    - Compatible with 10X Genomics and other single-cell TCR-seq formats

    Chain selection priority:
    1. If multiple alpha or beta chains exist per cell, selects the one with highest duplicate_count
    2. In case of ties, selects the first occurrence
    3. Only cells with both alpha and beta chains are retained

    Parameters:
        data (pandas.DataFrame): Input data containing TCR alpha and beta chain information.
                                Must include columns: cell_id_col, "tcr_chain", "duplicate_count", sequence_col
        sequence_col (str): The name of the column containing the amino acid sequences to embed.
        cell_id_col (str): The name of the column containing the single-cell barcode.

    Returns:
        pandas.DataFrame: Dataframe with concatenated alpha and beta chains per cell.
                         Format: BETA<cls><cls>ALPHA for each cell.

    Raises:
        ValueError: If required columns are missing from the input data.

    Example:
        # Concatenate alpha and beta chains for single-cell analysis
        paired_tcr = concatenate_alphabeta(tcr_data, "sequence_vdj_aa", "cell_id")
    """
    colnames = [cell_id_col, "tcr_chain", "duplicate_count", sequence_col]
    missing_cols = [col for col in colnames if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Column(s) {missing_cols} is/are not present in the input data and are needed to concatenate alpha and beta chains."
        )

    # if tie in maximum duplicate_count, return the first occurrence
    data = data.loc[data.groupby([cell_id_col, "tcr_chain"])["duplicate_count"].idxmax()]
    data = data.pivot(index=cell_id_col, columns="tcr_chain", values=sequence_col)
    data = data.reset_index(level=cell_id_col)
    n_cells = data.shape[0]
    data = data.dropna(axis=0)
    n_dropped = n_cells - data.shape[0]
    if n_dropped > 0:
        logger.info("Dropping %s cells with missing alpha or beta chain...", n_dropped)
    data.loc[:, sequence_col] = data.B + "<cls><cls>" + data.A  # Beta first, then alpha (similar to Heavy + Light)
    return data

# ======================================
# ===== TCR-SPECIFIC FUNCTIONS END =====
# ======================================
