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


def get_cdr3_sequence_column(airr_df: pd.DataFrame, sequence_col: str = "sequence_vdj_aa"):
    """
    Determines the appropriate CDR3 sequence column for TCR models.

    TCR embedding models (TCR-BERT, TCRT5, TCREMP) require CDR3 sequences, not full VDJ sequences.
    This function checks for standard AIRR CDR3 columns and falls back to the specified sequence column.

    Parameters:
        airr_df (pandas.DataFrame): Input AIRR rearrangement table.
        sequence_col (str): Default sequence column to use if no CDR3 column is found.

    Returns:
        str: The name of the column to use for CDR3 sequences.

    Note:
        Priority order: junction_aa > cdr3_aa > sequence_col
        For TCR models, using full VDJ sequences may reduce accuracy.
    """
    # Check for standard AIRR CDR3 columns in priority order
    cdr3_columns = ["junction_aa", "cdr3_aa"]

    for col in cdr3_columns:
        if col in airr_df.columns and not airr_df[col].isna().all():
            logger.info(f"Using CDR3 sequences from column '{col}' for TCR embedding")
            return col

    # Fall back to the specified sequence column with a warning
    logger.warning(
        f"No CDR3-specific columns (junction_aa, cdr3_aa) found. Using '{sequence_col}' column. "
        f"Note: TCR models (TCR-BERT, TCRT5, TCREMP) are trained on CDR3 sequences, not full VDJ sequences. "
        f"Using full sequences may reduce embedding accuracy."
    )
    return sequence_col


def process_airr(
    airr_df: pd.DataFrame,
    chain_mode: str,
    sequence_col: str = "sequence_vdj_aa",
    cell_id_col: str = "cell_id",
    receptor_type: str = "all",
    selection_col: str = "duplicate_count",
    use_cdr3_for_tcr: bool = True,
    mode: str = "concat",
):
    """
    Processes AIRR-seq data and returns a pandas DataFrame containing sequences to embed.

    Uses AMULETY's unified H/L/HL interface for both BCR and TCR data. See embed_airr()
    function documentation for detailed chain parameter explanations.

    Parameters:
        airr_df (pandas.DataFrame): Input AIRR rearrangement table as a pandas DataFrame.
        chain_mode (str): The input chain, one of ["H", "L", "HL", "LH", "H+L"].

        sequence_col (str): The name of the column containing the amino acid sequences to embed.
        cell_id_col (str): The name of the column containing the single-cell barcode.
        receptor_type (str): The receptor type to validate, one of ["BCR", "TCR", "all"].
                           - "BCR": validates only BCR chains (IGH, IGL, IGK) are present
                           - "TCR": validates only TCR chains (TRA, TRB, TRG, TRD) are present
                           - "all": allows both BCR and TCR chains in the same file
        selection_col (str): The name of the numeric column used to select the best chain when
                           multiple chains of the same type exist per cell. Default: "duplicate_count".
        use_cdr3_for_tcr (bool): Whether to automatically use CDR3 sequences for TCR data when available.
                               Default: True. Set to False to force use of sequence_col for all data.

    Returns:
        pandas.DataFrame: Dataframe with formatted sequences.

    Raises:
        ValueError: If chain is not one of ["H", "L", "HL", "LH", "H+L"] or receptor_type validation fails.
    """
    allowed_sequence_input = ["H", "L", "HL", "LH", "H+L"]
    if chain_mode not in allowed_sequence_input:
        raise ValueError(f"Input x must be one of {allowed_sequence_input}")

    # Warning for LH order
    if chain_mode == "LH":
        warnings.warn(
            "LH (Light-Heavy) chain order detected. Most paired models are trained on HL (Heavy-Light) order. "
            "Using LH order may result in reduced accuracy. Consider using --chain_mode HL for better performance.",
            UserWarning,
        )

    data = airr_df.copy()
    if "locus" not in data.columns:
        data.loc[:, "locus"] = data.loc[:, "v_call"].apply(lambda x: x[:3])

    # Detect if this is TCR data and adjust sequence column for CDR3 if requested
    tcr_loci = {"TRA", "TRB", "TRG", "TRD"}
    present_loci = set(data["locus"].unique())
    is_tcr_data = bool(present_loci & tcr_loci)

    # Use CDR3 sequences for TCR data if available and requested
    effective_sequence_col = sequence_col
    if use_cdr3_for_tcr and is_tcr_data:
        effective_sequence_col = get_cdr3_sequence_column(data, sequence_col)

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
            "Gamma/Delta TCR chains (%s) detected. Note: TCR-specific models (TCR-BERT, Trex, TCREMP) "
            "are primarily trained on Alpha/Beta TCRs. For Gamma/Delta TCRs, consider using general protein "
            "models (ESM2, ProtT5) which support all TCR types.",
            list(gamma_delta_chains),
        )

    # bulk only
    if cell_id_col not in data.columns:
        logger.info(
            "No %s column detected. Processing as bulk data. If the data is single-cell, please specify cell_id_col for the barcode column.",
            cell_id_col,
        )
        if chain_mode in ["HL", "LH", "H+L"]:
            raise ValueError(f'chain = "{chain_mode}" invalid for bulk mode')
        else:
            # For bulk data with single chain (H or L)
            colnames = ["sequence_id", effective_sequence_col]
            data = data.loc[data.chain == chain_mode, colnames]

    # single-cell only
    elif data[cell_id_col].notna().all():
        logger.info("Processing single-cell data...")
        if chain_mode == "HL":
            logging.info("Concatenating heavy and light chain per cell (HL order)...")
            data = concatenate_heavylight(
                data, effective_sequence_col, cell_id_col, selection_col, order="HL", mode=mode
            )
        elif chain_mode == "LH":
            logger.info("Concatenating light and heavy chain per cell (LH order)...")
            data = concatenate_heavylight(
                data, effective_sequence_col, cell_id_col, selection_col, order="LH", mode=mode
            )
        elif chain_mode == "H+L":
            logger.info("Processing both heavy and light chains separately...")
            if mode == "tab_locus_gene":
                # For models like TCREMP that need H+L in tab_locus_gene format
                data = process_h_plus_l(data, effective_sequence_col, cell_id_col, selection_col, mode=mode)
            else:
                # For other models that need H+L in separate entries
                data = process_h_plus_l(data, effective_sequence_col, cell_id_col, selection_col, mode="tab")
        else:
            # For single-cell data with single chain (H or L)
            if mode == "tab_locus_gene":
                # For models like TCREMP that need single chains in tab_locus_gene format
                data = process_h_plus_l(data, effective_sequence_col, cell_id_col, selection_col, mode=mode)
            else:
                # For other models that need simple single chain processing
                colnames = [cell_id_col, effective_sequence_col]
                data = data.loc[data.chain == chain_mode, colnames]
    # mixed
    else:
        logger.info("Missing values in %s column. Processing as mixed bulk and single-cell data...", cell_id_col)
        if chain_mode == "HL":
            logger.info("Concatenating heavy and light chain per cell (HL order)...")
            data = data.loc[data[cell_id_col].notna(),]
            data = concatenate_heavylight(
                data, effective_sequence_col, cell_id_col, selection_col, order="HL", mode=mode
            )
        elif chain_mode == "LH":
            logger.info("Concatenating light and heavy chain per cell (LH order)...")
            data = data.loc[data[cell_id_col].notna(),]
            data = concatenate_heavylight(
                data, effective_sequence_col, cell_id_col, selection_col, order="LH", mode=mode
            )
        elif chain_mode == "H+L":
            logger.info("Processing both heavy and light chains separately...")
            data = data.loc[data[cell_id_col].notna(),]
            if mode == "tab_locus_gene":
                # For models like TCREMP that need H+L in tab_locus_gene format
                data = process_h_plus_l(data, effective_sequence_col, cell_id_col, selection_col, mode=mode)
            else:
                # For other models that need H+L in separate entries
                data = process_h_plus_l(data, effective_sequence_col, cell_id_col, selection_col, mode="tab")
        else:
            # For mixed data with single chain (H or L)
            if mode == "tab_locus_gene":
                # For models like TCREMP that need single chains in tab_locus_gene format
                data = data.loc[data[cell_id_col].notna(),]
                data = process_h_plus_l(data, effective_sequence_col, cell_id_col, selection_col, mode=mode)
            else:
                # For other models that need simple single chain processing
                colnames = ["sequence_id", cell_id_col, effective_sequence_col]
                data = data.loc[data.chain == chain_mode, colnames]

    return data


def concatenate_heavylight(
    data: pd.DataFrame,
    sequence_col: str,
    cell_id_col: str,
    selection_col: str = "duplicate_count",
    order: str = "HL",
    mode: str = "concat",
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
        mode (str): Mode to use in concatenating sequences. By default it concatenates the sequences (concat),
                    it can also tabulate the sequences alone (tab) or together with the locus and segment (tab_locus_gene).

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

    # TODO: implement the case for all modes
    # First pivot dataframe according to chain column values (H and L)
    data_chain = data.pivot(index=cell_id_col, columns="chain", values=sequence_col)
    data_chain = data_chain.reset_index(level=cell_id_col)
    n_cells = data_chain.shape[0]
    data_chain = data_chain.dropna(axis=0)
    n_dropped = n_cells - data.shape[0]
    if n_dropped > 0:
        logging.info("Dropping %s cells with missing heavy or light chain...", n_dropped)

    if mode == "concat":
        # Concatenate based on order parameter
        if order == "HL":
            data_chain.loc[:, sequence_col] = data_chain.H + "<cls><cls>" + data_chain.L
        elif order == "LH":
            data_chain.loc[:, sequence_col] = data_chain.L + "<cls><cls>" + data_chain.H
        else:
            raise ValueError(f"Invalid order parameter: {order}. Must be 'HL' or 'LH'.")
        return data_chain
    elif mode == "tab":
        return data_chain
    elif mode == "tab_locus_gene":
        # Create locus_vgene and locus_jgene columns for TCREMP format
        data_full = data.copy()  # Work with full data before pivot

        # Extract V and J gene information
        # For V genes: extract locus + V (e.g., TRA -> TRAV, TRB -> TRBV)
        data_full.loc[:, "locus_vgene"] = data_full["locus"] + "V"
        # For J genes: extract locus + J (e.g., TRA -> TRAJ, TRB -> TRBJ)
        data_full.loc[:, "locus_jgene"] = data_full["locus"] + "J"

        # Second pivot for V genes
        data_vgene = data_full.pivot(index=cell_id_col, columns="locus_vgene", values="v_call")
        data_vgene = data_vgene.reset_index()

        # Third pivot for J genes
        data_jgene = data_full.pivot(index=cell_id_col, columns="locus_jgene", values="j_call")
        data_jgene = data_jgene.reset_index()

        # Merge all three pivoted dataframes
        result = data_chain.merge(data_vgene, on=cell_id_col, how="outer")
        result = result.merge(data_jgene, on=cell_id_col, how="outer")

        # Remove columns ending with 'D' (D gene related) as TCREMP doesn't need them
        d_columns = [col for col in result.columns if col.endswith("D")]
        if d_columns:
            result = result.drop(columns=d_columns)

        return result
    else:
        raise ValueError(f"Invalid mode parameter: {mode}. Must be 'concat', 'tab', or 'tab_locus_gene'.")


def process_h_plus_l(
    data: pd.DataFrame, sequence_col: str, cell_id_col: str, selection_col: str = "duplicate_count", mode: str = "tab"
):
    """
    Processes both heavy and light chains separately for H+L, H, or L formats.

    Returns a DataFrame with heavy and/or light chain sequences for each cell,
    keeping them as separate entries rather than concatenating them.
    Supports different output modes including tab_locus_gene format.

    If a cell contains multiple chains of the same type, selects the one with highest
    value in the selection column.

    Parameters:
        data (pandas.DataFrame): Input data containing chain information.
        sequence_col (str): The name of the column containing the amino acid sequences.
        cell_id_col (str): The name of the column containing the single-cell barcode.
        selection_col (str): The name of the numeric column used to select the best chain.
        mode (str): Output mode - "tab" for simple tabular format, "tab_locus_gene" for
                   extended format with V/J gene information.

    Returns:
        pandas.DataFrame: Dataframe with processed chain sequences in the specified format.
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

    # Ensure the sequence column is properly included in the output
    if sequence_col not in data.columns:
        raise ValueError(f"Sequence column '{sequence_col}' not found in data.")

    if mode == "tab":
        # Simple tabular format - keep chains as separate entries
        # Add chain type identifier to sequence_id for tracking
        data.loc[:, "sequence_id"] = data[cell_id_col] + "_" + data["chain"]
        return data

    elif mode == "tab_locus_gene":
        # Extended format with V/J gene information for models like TCREMP
        # This handles H+L, H, or L chains separately with gene information

        # Add chain type identifier to sequence_id for tracking
        data.loc[:, "sequence_id"] = data[cell_id_col] + "_" + data["chain"]

        # Create locus_vgene and locus_jgene columns
        data.loc[:, "locus_vgene"] = data["locus"] + "V"
        data.loc[:, "locus_jgene"] = data["locus"] + "J"

        # For each row, create columns for the specific locus V and J genes
        # This creates a wide format where each chain type gets its own V/J columns
        result_rows = []

        for _, row in data.iterrows():
            result_row = {
                cell_id_col: row[cell_id_col],
                "sequence_id": row["sequence_id"],
                sequence_col: row[sequence_col],
                "chain": row["chain"],
                "locus": row["locus"],
            }

            # Add V gene column for this locus
            v_col_name = row["locus_vgene"]
            result_row[v_col_name] = row["v_call"]

            # Add J gene column for this locus
            j_col_name = row["locus_jgene"]
            result_row[j_col_name] = row["j_call"]

            result_rows.append(result_row)

        result = pd.DataFrame(result_rows)

        # Remove any columns ending with 'D' (D gene related) as TCREMP doesn't need them
        d_columns = [col for col in result.columns if col.endswith("D")]
        if d_columns:
            result = result.drop(columns=d_columns)

        return result

    else:
        raise ValueError(f"Invalid mode parameter: {mode}. Must be 'tab' or 'tab_locus_gene'.")
