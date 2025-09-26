"""Main module."""
import logging
import subprocess
import warnings
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


class ConditionalFormatter(logging.Formatter):
    def format(self, record):
        if hasattr(record, "simple") and record.simple:
            return record.getMessage()
        else:
            return logging.Formatter.format(self, record)


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


def get_cdr3_sequence_column(airr: pd.DataFrame, default_sequence_col: str):
    """
    Get the best CDR3 sequence column for TCR data.

    Parameters:
        airr (pd.DataFrame): AIRR DataFrame
        default_sequence_col (str): Default sequence column name

    Returns:
        str: The best CDR3 sequence column name
    """
    # Preferred CDR3 columns in order of preference
    cdr3_columns = ["junction_aa", "cdr3_aa"]

    for col in cdr3_columns:
        if col in airr.columns:
            # Check if column has non-null values
            if not airr[col].isna().all():
                logger.info(f"Using CDR3 column: {col}")
                return col

    # If no CDR3 columns found, return the default
    logger.warning(f"No CDR3 columns found, using default: {default_sequence_col}")
    return default_sequence_col


def process_airr(
    airr_df: pd.DataFrame,
    chain_mode: str,
    sequence_col: str = "sequence_vdj_aa",
    cell_id_col: str = "cell_id",
    duplicate_col: str = "duplicate_count",
    receptor_type: str = "all",
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
        duplicate_col (str): The name of the numeric column used to select the best chain when
                           multiple chains of the same type exist per cell. Default: "duplicate_count".
        mode (str): Mode to use in concatenating sequences. By default it concatenates the sequences when the HL chain is provided (concat),
                    it can also tabulate the sequences alone (tab) or together with the locus and segment (tab_locus_gene).

    Returns:
        pandas.DataFrame: Dataframe with formatted sequences.

    Raises:
        ValueError: If chain is not one of ["H", "L", "HL", "LH", "H+L"] or receptor_type validation fails.
    """
    allowed_sequence_input = ["H", "L", "HL", "LH", "H+L"]
    if chain_mode not in allowed_sequence_input:
        raise ValueError(f"Input x must be one of {allowed_sequence_input}.")
    allowed_modes = ["concat", "tab", "tab_locus_gene"]
    if mode not in allowed_modes:
        raise ValueError(f"Mode must be one of {allowed_modes}.")

    # Check that required columns exist
    required_cols = [sequence_col, "v_call"]
    missing_cols = [col for col in required_cols if col not in airr_df.columns]
    if missing_cols:
        raise ValueError(f"Column(s) {missing_cols} are not present in the input data and are needed for processing.")

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
            "Gamma/Delta TCR chains (%s) detected. Note: TCR-specific models (TCR-BERT, Trex, TCREMP) "
            "are primarily trained on Alpha/Beta TCRs. For Gamma/Delta TCRs, consider using general protein "
            "models (ESM2, ProtT5) which support all TCR types.",
            list(gamma_delta_chains),
        )

    # Determine data type
    is_bulk = cell_id_col not in data.columns
    is_single_cell = not is_bulk and data[cell_id_col].notna().all()
    is_mixed = not is_bulk and not is_single_cell

    if is_bulk:
        logger.info("Bulk AIRR data detected (no cell_id column).")
    elif is_single_cell:
        logger.info("Single-cell AIRR data detected (all entries have cell_id).")
    elif is_mixed:
        logger.info("Mixed AIRR data detected (some entries have cell_id, others do not).")

    # Process based on chain_mode
    if chain_mode in ["HL", "LH"]:
        # HL/LH modes: error for bulk, process for single-cell/mixed
        if is_bulk:
            raise ValueError(f'Chain = "{chain_mode}" is invalid for bulk mode. Please use "H+L", "H" or "L" instead.')

        # Warning for LH order
        if chain_mode == "LH":
            warnings.warn(
                "LH (Light-Heavy) chain order detected. Most paired models are trained on HL (Heavy-Light) order. "
                "Using LH order may result in reduced accuracy. Consider using --chain_mode HL for better performance.",
                UserWarning,
            )

        # If mixed data, filter to only sequences with cell_id
        if is_mixed:
            before_filter = len(data)
            data = data.loc[data[cell_id_col].notna(),]
            after_filter = len(data)
            removed_count = before_filter - after_filter
            if removed_count > 0:
                logger.info("Removed %d sequences without cell_id for paired chain processing", removed_count)

        data = concatenate_heavylight(data, sequence_col, cell_id_col, duplicate_col, order=chain_mode, mode=mode)

    elif chain_mode == "H+L":
        # H+L mode: same processing for all data types

        # Add dummy cell_id col if bulk data
        if is_bulk:
            data[cell_id_col] = pd.NA

        # For models like TCREMP that need H+L in tab_locus_gene format
        if mode == "tab_locus_gene":
            data = process_h_plus_l(data, sequence_col, cell_id_col, duplicate_col, mode=mode)
        else:
            # For other models that need H+L in separate entries
            data = process_h_plus_l(data, sequence_col, cell_id_col, duplicate_col, mode="tab")

    else:
        # Single chain mode (H or L): same processing for all data types
        before_filter = len(data)
        data = data.loc[data.chain == chain_mode]
        after_filter = len(data)
        removed_count = before_filter - after_filter
        if removed_count > 0:
            logger.info("Removed %d sequences not matching %s chain", removed_count, chain_mode)

        if is_bulk:
            data[cell_id_col] = pd.NA

        elif is_single_cell:
            # For models like TCREMP that need H+L in tab_locus_gene format
            if mode == "concat":
                data = process_h_plus_l(data, sequence_col, cell_id_col, duplicate_col, mode="tab")
            else:
                data = process_h_plus_l(data, sequence_col, cell_id_col, duplicate_col, mode=mode)
    return data.loc[:, sequence_col], data


def concatenate_heavylight(
    data: pd.DataFrame,
    sequence_col: str,
    cell_id_col: str,
    duplicate_col: str = "duplicate_count",
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
        ValueError: If required columns are missing or duplicate_col is not numeric.
    """
    # TODO add check if multiple heavy chains per cell and warn users
    colnames = [cell_id_col, "locus", duplicate_col, sequence_col]
    missing_cols = [col for col in colnames if col not in data.columns]
    if missing_cols:
        raise ValueError(
            f"Column(s) {missing_cols} is/are not present in the input data and are needed to concatenate heavy and light chains."
        )

    # Check that duplicate_col is numeric
    if not pd.api.types.is_numeric_dtype(data[duplicate_col]):
        raise ValueError(
            f"Selection column '{duplicate_col}' must be numeric. Found dtype: {data[duplicate_col].dtype}"
        )
    # Check that duplicate_col does not contain NaN values
    if data[duplicate_col].isna().any():
        raise ValueError(f"Selection column '{duplicate_col}' contains NaN values. Please remove them or fix them.")

    # if tie in maximum duplicate_col value, return the first occurrence
    data = data.loc[data.groupby([cell_id_col, "chain"])[duplicate_col].idxmax()]

    # First pivot dataframe according to chain column values (H and L)
    data_chain = data.pivot(index=cell_id_col, columns="chain", values=sequence_col)
    data_chain = data_chain.reset_index(level=cell_id_col)
    n_cells = data_chain.shape[0]
    data_chain = data_chain.dropna(axis=0)
    n_dropped = n_cells - data_chain.shape[0]
    if n_dropped > 0:
        logging.info("Dropping %s cells with missing heavy or light chain...", n_dropped)

    # Throw error if no rows left after dropping
    if data_chain.shape[0] == 0:
        raise ValueError("No cells with both heavy and light chains found.")

    if mode == "concat":
        # Concatenate based on order parameter
        if order == "HL":
            data_chain.loc[:, sequence_col] = data_chain["H"] + "<cls><cls>" + data_chain["L"]
        elif order == "LH":
            data_chain.loc[:, sequence_col] = data_chain["L"] + "<cls><cls>" + data_chain["H"]
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

        # Third pivot for J genes (only if j_call column exists)
        if "j_call" in data_full.columns:
            data_jgene = data_full.pivot(index=cell_id_col, columns="locus_jgene", values="j_call")
            data_jgene = data_jgene.reset_index()

            # Merge all three pivoted dataframes
            result = data_chain.merge(data_vgene, on=cell_id_col, how="outer")
            result = result.merge(data_jgene, on=cell_id_col, how="outer")
        else:
            # Only merge chain and V gene data if J gene data is not available
            result = data_chain.merge(data_vgene, on=cell_id_col, how="outer")
            # Add placeholder J gene columns for consistency
            for locus in data_full["locus"].unique():
                j_col = f"{locus}J"
                result[j_col] = "Unknown"

        # Remove columns ending with 'D' (D gene related) as TCREMP doesn't need them
        # d_columns = [col for col in result.columns if col.endswith("D")]
        # if d_columns:
        #     result = result.drop(columns=d_columns)

        return result
    else:
        raise ValueError(f"Invalid mode parameter: {mode}. Must be 'concat', 'tab', or 'tab_locus_gene'.")


def process_h_plus_l(
    data: pd.DataFrame, sequence_col: str, cell_id_col: str, duplicate_col: str = "duplicate_count", mode: str = "tab"
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
        duplicate_col (str): The name of the numeric column used to select the best chain.
        mode (str): Output mode - "tab" for simple tabular format, "tab_locus_gene" for
                   extended format with V/J gene information.

    Returns:
        pandas.DataFrame: Dataframe with processed chain sequences in the specified format.
    """
    # Supports only single-cell or mixed data with cell_id_col
    # Validate selection column
    if duplicate_col not in data.columns:
        raise ValueError(f"Selection column '{duplicate_col}' not found in data.")

    if cell_id_col not in data.columns:
        raise ValueError(f"Cell ID column '{cell_id_col}' not found in data.")

    if not pd.api.types.is_numeric_dtype(data[duplicate_col]):
        raise ValueError(
            f"Selection column '{duplicate_col}' must be numeric. Found dtype: {data[duplicate_col].dtype}"
        )

    # Check whether data is mixed single-cell and bulk
    # For single-cell data select best chain per cell and chain type

    # TODO this is not needed I think remove
    # is_mixed = data[cell_id_col].isna().any()
    # if is_mixed:
    #     data_bulk = data.loc[data[cell_id_col].isna(),]
    #     data_sc = data.loc[data[cell_id_col].notna(),]
    #     data_sc = data_sc.loc[data_sc.groupby([cell_id_col, "chain"])[duplicate_col].idxmax()]
    #     data = pd.concat([data_sc, data_bulk], ignore_index=True)
    # else:
    #     data = data.loc[data.groupby([cell_id_col, "chain"])[duplicate_col].idxmax()]

    # Ensure the sequence column is properly included in the output
    if sequence_col not in data.columns:
        raise ValueError(f"Sequence column '{sequence_col}' not found in data.")

    if mode == "tab":
        # Simple tabular format - keep chains as separate entries
        # Keep original sequence_id for metadata merging - chain info is preserved in 'chain' column
        return data

    elif mode == "tab_locus_gene":
        # Extended format with V/J gene information for models like TCREMP
        # This handles H+L, H, or L chains separately with gene information

        # Create locus_vgene and locus_jgene columns
        data.loc[:, "locus_vgene"] = data["locus"] + "V"
        data.loc[:, "locus_jgene"] = data["locus"] + "J"

        # Start with base columns
        result = data[[cell_id_col, "sequence_id", sequence_col, "chain", "locus"]].copy()

        # Use vectorized operations to create dynamic V and J gene columns
        # Get unique loci to create the appropriate columns
        unique_loci = data["locus"].unique()

        # Initialize all possible V and J gene columns with NaN
        for locus in unique_loci:
            v_col = f"{locus}V"
            j_col = f"{locus}J"
            result[v_col] = pd.NA
            result[j_col] = pd.NA

        # Use vectorized assignment to populate the appropriate columns
        for locus in unique_loci:
            v_col = f"{locus}V"
            j_col = f"{locus}J"
            mask = data["locus"] == locus
            result.loc[mask, v_col] = data.loc[mask, "v_call"]
            # Only populate J gene if j_call column exists
            if "j_call" in data.columns:
                result.loc[mask, j_col] = data.loc[mask, "j_call"]
            else:
                # Set placeholder values for missing J gene information
                result.loc[mask, j_col] = "Unknown"

        return result

    else:
        raise ValueError(f"Invalid mode parameter: {mode}. Must be 'tab' or 'tab_locus_gene'.")


def check_dependencies():
    """
    Check if optional embedding dependencies are installed and provide installation instructions.

    This function checks all model types (BCR, TCR, and protein language models) for missing dependencies.

    Returns:
        list: List of tuples (model_name, installation_command) for missing dependencies
    """
    missing_deps = []
    available_models = []

    # Check BCR models (included in requirements.txt but good to verify installation)
    try:
        from antiberty import AntiBERTyRunner  # noqa: F401

        available_models.append("AntiBERTy")
    except ImportError:
        missing_deps.append(("AntiBERTy", "pip install antiberty"))

    try:
        import ablang  # noqa: F401

        available_models.append("AbLang")
    except ImportError:
        missing_deps.append(("AbLang", "pip install ablang"))

    # Check TCR models
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

    try:
        from transformers import BertModel, BertTokenizer  # noqa: F401

        available_models.append("TCR-BERT")
    except ImportError:
        missing_deps.append(("TCR-BERT", "pip install transformers"))

    try:
        from transformers import T5ForConditionalGeneration, T5Tokenizer  # noqa: F401

        available_models.append("TCRT5")
    except ImportError:
        missing_deps.append(("TCRT5", "pip install transformers"))

    # Check protein language models
    try:
        from transformers import EsmModel, EsmTokenizer  # noqa: F401

        available_models.append("ESM2")
    except ImportError:
        missing_deps.append(("ESM2", "pip install transformers"))

    try:
        import sentencepiece  # noqa: F401
        from transformers import T5EncoderModel, T5Tokenizer  # noqa: F401

        available_models.append("ProtT5")
    except ImportError as e:
        if "sentencepiece" in str(e):
            missing_deps.append(("ProtT5", "pip install transformers sentencepiece"))
        else:
            missing_deps.append(("ProtT5", "pip install transformers"))

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
        logger.info("Available models: %s", ", ".join(available_models))

    if missing_deps:
        logger.warning("Missing model dependencies: %s", ", ".join([dep[0] for dep in missing_deps]))
    else:
        logger.info("All embedding dependencies are available!")

    return missing_deps
