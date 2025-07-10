"""Main module."""
import logging
import warnings
from typing import Iterable

import pandas as pd

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


def process_airr(
    airr_df: pd.DataFrame,
    chain: str,
    sequence_col: str = "sequence_vdj_aa",
    cell_id_col: str = "cell_id",
    receptor_type: str = "all",
    selection_col: str = "duplicate_count",
):
    """
    Processes AIRR-seq data and returns a pandas DataFrame containing sequences to embed.

    Uses AMULETY's unified H/L/HL interface for both BCR and TCR data. See embed_airr()
    function documentation for detailed chain parameter explanations.

    Parameters:
        airr_df (pandas.DataFrame): Input AIRR rearrangement table as a pandas DataFrame.
        chain (str): The input chain, one of ["H", "L", "HL", "LH", "H+L"].

        sequence_col (str): The name of the column containing the amino acid sequences to embed.
        cell_id_col (str): The name of the column containing the single-cell barcode.
        receptor_type (str): The receptor type to validate, one of ["BCR", "TCR", "all"].
                           - "BCR": validates only BCR chains (IGH, IGL, IGK) are present
                           - "TCR": validates only TCR chains (TRA, TRB, TRG, TRD) are present
                           - "all": allows both BCR and TCR chains in the same file
        selection_col (str): The name of the numeric column used to select the best chain when
                           multiple chains of the same type exist per cell. Default: "duplicate_count".

    Returns:
        pandas.DataFrame: Dataframe with formatted sequences.

    Raises:
        ValueError: If chain is not one of ["H", "L", "HL", "LH", "H+L"] or receptor_type validation fails.
    """
    allowed_sequence_input = ["H", "L", "HL", "LH", "H+L"]
    if chain not in allowed_sequence_input:
        raise ValueError(f"Input x must be one of {allowed_sequence_input}")

    # Warning for LH order
    if chain == "LH":
        warnings.warn(
            "LH (Light-Heavy) chain order detected. Most paired models are trained on HL (Heavy-Light) order. "
            "Using LH order may result in reduced accuracy. Consider using --chain HL for better performance.",
            UserWarning,
        )

    data = airr_df.copy()
    if "locus" not in data.columns:
        data.loc[:, "locus"] = data.loc[:, "v_call"].apply(lambda x: x[:3])

    # ===== RECEPTOR TYPE VALIDATION =====
    bcr_loci = {"IGH", "IGL", "IGK"}
    tcr_loci = {"TRA", "TRB", "TRG", "TRD"}
    present_loci = set(data["locus"].unique())

    bcr_present = bool(present_loci & bcr_loci)
    tcr_present = bool(present_loci & tcr_loci)

    if receptor_type.upper() == "BCR":
        if tcr_present and bcr_present:
            tcr_chains = present_loci & tcr_loci
            logger.warning(
                "TCR chains (%s) detected in BCR-only mode. These will be removed and only BCR chains used.",
                list(tcr_chains),
            )
            data = data[data["locus"].isin(bcr_loci)]
        elif tcr_present and not bcr_present:
            raise ValueError(
                "No BCR chains (IGH, IGL, IGK) found in data. This embedding model is trained for BCR data and should not be used for TCR-only data."
            )
    elif receptor_type.upper() == "TCR":
        if bcr_present and tcr_present:
            bcr_chains = present_loci & bcr_loci
            logger.warning(
                "BCR chains (%s) detected in TCR-only mode. These will be removed and only TCR chains used.",
                list(bcr_chains),
            )
            data = data[data["locus"].isin(tcr_loci)]
        elif bcr_present and not tcr_present:
            raise ValueError(
                "No TCR chains (TRA, TRB, TRG, TRD) found in data. This embedding model is trained for TCR data and should not be used for BCR-only data."
            )
    elif receptor_type.upper() == "ALL":
        logger.info("Processing both BCR and TCR sequences from the file.")
    else:
        raise ValueError(f"receptor_type must be one of ['BCR', 'TCR', 'all'], got '{receptor_type}'")

    # ===== UNIFIED CHAIN MAPPING =====
    # Map loci to unified H/L interface
    data.loc[:, "chain"] = data.loc[:, "locus"].apply(lambda x: "H" if x in ["IGH", "TRB", "TRD"] else "L")

    # Check for gamma/delta TCR and warn about model compatibility
    gamma_delta_present = bool(present_loci & {"TRG", "TRD"})
    if gamma_delta_present and receptor_type.upper() in ["TCR", "ALL"]:
        gamma_delta_chains = present_loci & {"TRG", "TRD"}
        logger.warning(
            "Gamma/Delta TCR chains (%s) detected. Note: TCR-specific models (TCR-BERT, Trex, TCREMP, DeepTCR) "
            "are primarily trained on Alpha/Beta TCRs. For Gamma/Delta TCRs, consider using general protein "
            "models (ESM2, ProtT5) which support all TCR types.",
            list(gamma_delta_chains),
        )

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
        if chain in ["HL", "LH", "H+L"]:
            raise ValueError(f'chain = "{chain}" invalid for bulk mode. Use "H" or "L" for bulk data.')
        else:
            colnames = ["sequence_id", sequence_col]
            data = data.loc[data.chain == chain, colnames]

    elif data_type == "single-cell-only":
        logger.info("Processing single-cell data...")
        if chain == "HL":
            logging.info("Concatenating heavy and light chain per cell (HL order)...")
            data = concatenate_heavylight(data, sequence_col, cell_id_col, selection_col, order="HL")
        elif chain == "LH":
            logger.info("Concatenating light and heavy chain per cell (LH order)...")
            data = concatenate_heavylight(data, sequence_col, cell_id_col, selection_col, order="LH")
        elif chain == "H+L":
            logger.info("Processing both heavy and light chains separately...")
            data = process_h_plus_l(data, sequence_col, cell_id_col, selection_col)
        else:
            colnames = [cell_id_col, sequence_col]
            data = data.loc[data.chain == chain, colnames]

    elif data_type == "mixed":
        logger.info("Missing values in %s column. Processing as mixed bulk and single-cell data...", cell_id_col)
        if chain == "HL":
            logger.info("Concatenating heavy and light chain per cell (HL order)...")
            data = data.loc[data[cell_id_col].notna(),]
            data = concatenate_heavylight(data, sequence_col, cell_id_col, selection_col, order="HL")
        elif chain == "LH":
            logger.info("Concatenating light and heavy chain per cell (LH order)...")
            data = data.loc[data[cell_id_col].notna(),]
            data = concatenate_heavylight(data, sequence_col, cell_id_col, selection_col, order="LH")
        elif chain == "H+L":
            logger.info("Processing both heavy and light chains separately...")
            data = data.loc[data[cell_id_col].notna(),]
            data = process_h_plus_l(data, sequence_col, cell_id_col, selection_col)
        else:
            colnames = ["sequence_id", cell_id_col, sequence_col]
            data = data.loc[data.chain == chain, colnames]

    return data


def concatenate_heavylight(
    data: pd.DataFrame, sequence_col: str, cell_id_col: str, selection_col: str = "duplicate_count", order: str = "HL"
):
    """
    Concatenates heavy and light chain per cell using AMULETY's unified H/L interface.

    Concatenates sequences as: Heavy<cls><cls>Light (HL order) or Light<cls><cls>Heavy (LH order)
    for both BCR (IGH + IGL/IGK) and TCR (TRB/TRD + TRA/TRG) data.
    See embed_airr() documentation for chain mappings.

    If a cell contains multiple chains of the same type, selects the one with highest
    value in the selection column.

    Parameters:
        order (str): Chain concatenation order, either "HL" (Heavy-Light) or "LH" (Light-Heavy).
                    Default: "HL".

    Parameters:
        data (pandas.DataFrame): Input data containing heavy and light chain information.
                                 Must include columns: cell_id_col, "chain", selection_col, sequence_col
        sequence_col (str): The name of the column containing the amino acid sequences to embed.
        cell_id_col (str): The name of the column containing the single-cell barcode.
        selection_col (str): The name of the numeric column used to select the best chain when
                           multiple chains of the same type exist per cell. Default: "duplicate_count".

    Returns:
        pandas.DataFrame: Dataframe with concatenated heavy and light chains per cell.
                         Format: HEAVY<cls><cls>LIGHT for each cell.

    Raises:
        ValueError: If required columns are missing or selection_col is not numeric.
    """
    colnames = [cell_id_col, "locus", selection_col, sequence_col]
    missing_cols = [col for col in colnames if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Column(s) {missing_cols} is/are not present in the input data and are needed to concatenate heavy and light chains."
        )

    # Check that selection_col is numeric
    if not pd.api.types.is_numeric_dtype(data[selection_col]):
        raise ValueError(
            f"Selection column '{selection_col}' must be numeric. Found dtype: {data[selection_col].dtype}"
        )

    # if tie in maximum selection_col value, return the first occurrence
    data = data.loc[data.groupby([cell_id_col, "chain"])[selection_col].idxmax()]
    data = data.pivot(index=cell_id_col, columns="chain", values=sequence_col)
    data = data.reset_index(level=cell_id_col)
    n_cells = data.shape[0]
    data = data.dropna(axis=0)
    n_dropped = n_cells - data.shape[0]
    if n_dropped > 0:
        logging.info("Dropping %s cells with missing heavy or light chain...", n_dropped)

    # Concatenate based on order parameter
    if order == "HL":
        data.loc[:, sequence_col] = data.H + "<cls><cls>" + data.L
    elif order == "LH":
        data.loc[:, sequence_col] = data.L + "<cls><cls>" + data.H
    else:
        raise ValueError(f"Invalid order parameter: {order}. Must be 'HL' or 'LH'.")

    return data


def process_h_plus_l(data: pd.DataFrame, sequence_col: str, cell_id_col: str, selection_col: str = "duplicate_count"):
    """
    Processes both heavy and light chains separately for H+L format.

    Returns a DataFrame with both heavy and light chain sequences for each cell,
    but keeps them as separate entries rather than concatenating them.

    If a cell contains multiple chains of the same type, selects the one with highest
    value in the selection column.
    """
    # Validate selection column
    if selection_col not in data.columns:
        raise ValueError(f"Selection column '{selection_col}' not found in data.")

    if not pd.api.types.is_numeric_dtype(data[selection_col]):
        raise ValueError(
            f"Selection column '{selection_col}' must be numeric. Found dtype: {data[selection_col].dtype}"
        )

    # Select best chain for each cell and chain type
    data = data.loc[data.groupby([cell_id_col, "chain"])[selection_col].idxmax()]

    # Keep both H and L chains as separate entries
    colnames = [cell_id_col, "chain", sequence_col]
    data = data.loc[:, colnames]

    # Add chain type identifier to sequence_id for tracking
    data.loc[:, "sequence_id"] = data[cell_id_col] + "_" + data["chain"]

    return data
