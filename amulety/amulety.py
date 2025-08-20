"""Console script for amulety"""
import logging
import os
import subprocess
import time
import warnings
from importlib.metadata import version

import pandas as pd
import torch
import typer
from rich.console import Console
from typing_extensions import Annotated

from amulety.bcr_embeddings import ablang, antiberta2, antiberty, balm_paired
from amulety.protein_embeddings import custommodel, esm2, immune2vec, prott5
from amulety.tcr_embeddings import tcr_bert, tcremp, tcrt5
from amulety.utils import (
    check_dependencies,
    get_cdr3_sequence_column,
    process_airr,
)

__version__ = version("amulety")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def process_chain_data(
    airr: pd.DataFrame,
    chain: str,
    sequence_col: str,
    cell_id_col: str,
    duplicate_col: str,
    receptor_type: str,
    model_type: str = "standard",
):
    """
    Universal chain data processing function for different chain modes and model types.

    Parameters:
        airr: AIRR DataFrame
        chain: Chain mode (H, L, HL, LH, H+L)
        sequence_col: Sequence column name
        cell_id_col: Cell ID column name
        duplicate_col: Selection column name
        receptor_type: Receptor type
        model_type: Model type ("standard", "tabular")

    Returns:
        tuple: (processed_data_for_embedding, full_processed_data_for_output)
    """
    # Check that the mode is supported
    allowed_model_types = ["standard", "tabular"]
    if model_type not in allowed_model_types:
        raise ValueError(f"Unsupported model_type '{model_type}'. Allowed values: {allowed_model_types}")

    if model_type == "tabular":
        # Tabular processing mode (e.g. TCREMP)
        dat = process_airr(
            airr,
            chain,
            sequence_col=sequence_col,
            cell_id_col=cell_id_col,
            duplicate_col=duplicate_col,
            receptor_type=receptor_type,
            mode="tab_locus_gene",
        )
        return dat, dat

    else:
        # Standard model processing mode
        if chain in ["HL", "LH"]:
            # Paired chains: concatenate sequences
            dat = process_airr(
                airr,
                chain,
                sequence_col=sequence_col,
                cell_id_col=cell_id_col,
                duplicate_col=duplicate_col,
                receptor_type=receptor_type,
                mode="concat",
            )
            # The sequence column should now be consistently named as sequence_col
            if sequence_col in dat.columns:
                return dat.loc[:, sequence_col], dat
            else:
                raise ValueError(
                    f"Sequence column '{sequence_col}' not found in processed data. Available columns: {list(dat.columns)}"
                )
        elif chain == "H+L":
            # Separate chains: return DataFrame with H and L columns
            dat = process_airr(
                airr,
                chain,
                sequence_col=sequence_col,
                cell_id_col=cell_id_col,
                duplicate_col=duplicate_col,
                receptor_type=receptor_type,
                mode="tab",
            )
            return dat, dat  # Return complete DataFrame for both
        else:  # H or L single chain
            # Single chain: return single sequence column
            dat = process_airr(
                airr,
                chain,
                sequence_col=sequence_col,
                cell_id_col=cell_id_col,
                duplicate_col=duplicate_col,
                receptor_type=receptor_type,
                mode="tab",
            )
            # The sequence column should now be consistently named as sequence_col
            if sequence_col in dat.columns:
                return dat.loc[:, sequence_col], dat
            else:
                raise ValueError(
                    f"Sequence column '{sequence_col}' not found in processed data. Available columns: {list(dat.columns)}"
                )


app = typer.Typer()
stderr = Console(stderr=True)
stdout = Console()


def check_igblast_available():
    """Check if IgBlast is available in the system."""
    try:
        subprocess.run(["igblastn", "-help"], capture_output=True, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def translate_airr(
    airr: pd.DataFrame, tmpdir: str, reference_dir: str, keep_regions: bool = False, sequence_col: str = "sequence"
):
    """
    Translates nucleotide sequences to amino acid sequences using IgBlast.

    Requires IgBlast to be installed and available in PATH.
    Install with: conda install -c bioconda igblast
    """
    data = airr.copy()

    # Warn if translations already exist
    columns_reserved = ["sequence_aa", "sequence_alignment_aa", "sequence_vdj_aa"]
    overlap = [col for col in data.columns if col in columns_reserved]
    if len(overlap) > 0:
        logger.warning("Existing amino acid columns (%s) will be overwritten.", ", ".join(overlap))
        data = data.drop(overlap, axis=1)

    if tmpdir is None:
        tmpdir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmpdir, exist_ok=True)
    else:
        os.makedirs(tmpdir, exist_ok=True)

    out_fasta = os.path.join(tmpdir, "airr.fasta")
    out_igblast = os.path.join(tmpdir, "airr_igblast.tsv")

    start_time = time.time()
    logger.info("Converting AIRR table to FastA for IgBlast translation...")

    # Write out FASTA file
    with open(out_fasta, "w") as f:
        for _, row in data.iterrows():
            f.write(">" + row["sequence_id"] + "\n")
            f.write(row[sequence_col] + "\n")

    command_igblastn = [
        "igblastn",
        "-germline_db_V",
        f"{reference_dir}/database/imgt_human_ig_v",
        "-germline_db_D",
        f"{reference_dir}/database/imgt_human_ig_d",
        "-germline_db_J",
        f"{reference_dir}/database/imgt_human_ig_j",
        "-query",
        out_fasta,
        "-organism",
        "human",
        "-auxiliary_data",
        f"{reference_dir}/optional_file/human_gl.aux",
        "-show_translation",
        "-outfmt",
        "19",
        "-out",
        out_igblast,
    ]

    logger.info("Calling IgBlast for running translation...")
    try:
        pipes = subprocess.Popen(command_igblastn, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = pipes.communicate()

        if pipes.returncode != 0:
            raise Exception(f"IgBlast failed with error code {pipes.returncode}. {stderr.decode('utf-8')}")
    except FileNotFoundError:
        raise Exception(
            "IgBlast (igblastn) not found. Please install IgBlast:\n"
            "  1. Add conda channels:\n"
            "     conda config --add channels conda-forge\n"
            "     conda config --add channels bioconda\n"
            "  2. Install IgBlast:\n"
            "     conda install -c bioconda igblast\n"
            "  3. Or use mamba (faster):\n"
            "     conda install mamba -n base -c conda-forge\n"
            "     mamba install -c bioconda igblast\n"
            "  4. Or download manually from:\n"
            "     https://ftp.ncbi.nlm.nih.gov/blast/executables/igblast/release/\n"
            "Make sure 'igblastn' is in your system PATH."
        )

    keep_cols = ["sequence_id", "sequence_aa", "sequence_alignment_aa"]
    if keep_regions:
        keep_cols += [
            "fwr1_aa",
            "cdr1_aa",
            "fwr2_aa",
            "cdr2_aa",
            "fwr3_aa",
            "cdr3_aa",
            "fwr4_aa",
        ]
    igblast_transl = pd.read_csv(out_igblast, sep="\t", usecols=keep_cols)

    sequence_vdj_aa = [sa.replace("-", "") for sa in igblast_transl["sequence_alignment_aa"]]
    igblast_transl["sequence_vdj_aa"] = sequence_vdj_aa

    logger.info(
        "Saved the translations in the dataframe (sequence_aa contains the full translation and sequence_vdj_aa contains the VDJ translation)."
    )

    data_transl = pd.merge(data, igblast_transl, on="sequence_id", how="left")

    os.remove(out_fasta)
    os.remove(out_igblast)
    os.rmdir(tmpdir)

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))
    return data_transl


def embed_airr(
    airr: pd.DataFrame,
    chain: str,
    model: str,
    sequence_col: str = "sequence_vdj_aa",
    cell_id_col: str = "cell_id",
    cache_dir: str = "/tmp/amulety",
    batch_size: int = 50,
    embedding_dimension: int = None,
    max_length: int = None,
    model_path: str = None,
    output_type: str = "pickle",
    duplicate_col: str = "duplicate_count",
    skip_clustering: bool = True,
):
    """
    Embeds sequences from an AIRR DataFrame using the specified model.
    Parameters:
        airr (pd.DataFrame): Input AIRR rearrangement table as a pandas DataFrame.
        chain (str): The input chain, which can be one of ["H", "L", "HL", "LH", "H+L"].
                    For BCR: H=Heavy, L=Light, HL=Heavy-Light pairs, LH=Light-Heavy pairs, H+L=Both chains separately
                    For TCR: H=Beta/Delta, L=Alpha/Gamma, HL=Beta-Alpha/Delta-Gamma pairs, LH=Alpha-Beta/Gamma-Delta pairs, H+L=Both chains separately
        model (str): The embedding model to use.
                    BCR models: ["ablang", "antiberta2", "antiberty", "balm-paired"]
                    TCR models: ["tcr-bert", "tcremp", "tcrt5"]
                    Immune models (BCR & TCR): ["immune2vec"]
                    Protein models: ["esm2", "prott5", "custom"]
                    Use "custom" for fine-tuned models (requires model_path, embedding_dimension, max_length)
        sequence_col (str): The name of the column containing the amino acid sequences to embed.
        cell_id_col (str): The name of the column containing the single-cell barcode.
        cache_dir (Optional[str]): Cache dir for storing the pre-trained model weights.
        batch_size (int): The batch size of sequences to embed.
        embedding_dimension (int): The embedding dimension for custom models.
        max_length (int): The maximum sequence length for custom models.
        model_path (str): The path to the custom model.
        output_type (str): The type of output to return. Can be "df" for a pandas DataFrame or "pickle" for a serialized torch object.
        duplicate_col (str): The name of the numeric column used to select the best chain when
                           multiple chains of the same type exist per cell. Default: "duplicate_count".

    """
    # Check valid chain - unified interface for both BCR and TCR
    valid_chains = ["H", "L", "HL", "LH", "H+L"]
    if chain not in valid_chains:
        raise ValueError(f"Input chain must be one of {valid_chains}")

    # Use the chain parameter directly - no mapping needed
    if output_type not in ["df", "pickle"]:
        raise ValueError("Output type must be one of ['df', 'pickle']")
    if sequence_col not in airr.columns:
        raise ValueError(f"Column {sequence_col} not found in the input AIRR data.")

    # ===== BASIC CHAIN VALIDATION =====
    # Check if requested chains are available in the data based on locus information
    data_copy = airr.copy()
    if "locus" not in data_copy.columns:
        data_copy.loc[:, "locus"] = data_copy.loc[:, "v_call"].apply(lambda x: x[:3])

    present_loci = set(data_copy["locus"].unique())

    # Determine available chains based on locus
    available_chains = set()
    heavy_loci = {"IGH", "TRB", "TRD"}  # Heavy chains: IGH for BCR, TRB/TRD for TCR
    light_loci = {"IGL", "IGK", "TRA", "TRG"}  # Light chains: IGL/IGK for BCR, TRA/TRG for TCR

    if present_loci & heavy_loci:
        available_chains.add("H")
    if present_loci & light_loci:
        available_chains.add("L")

    # Validate chain availability
    if chain == "H" and "H" not in available_chains:
        raise ValueError(
            f"Chain parameter 'H' requires heavy chain data, but no heavy chain loci found. "
            f"Available loci: {', '.join(sorted(present_loci))}. Use --chain L for light chain analysis."
        )
    elif chain == "L" and "L" not in available_chains:
        raise ValueError(
            f"Chain parameter 'L' requires light chain data, but no light chain loci found. "
            f"Available loci: {', '.join(sorted(present_loci))}. Use --chain H for heavy chain analysis."
        )
    elif chain in ["HL", "LH", "H+L"] and not available_chains.issuperset({"H", "L"}):
        missing = {"H", "L"} - available_chains
        if "H" in missing:
            raise ValueError(
                f"Chain parameter '{chain}' requires heavy chain data, but no heavy chain loci found. "
                f"Available loci: {', '.join(sorted(present_loci))}. Use --chain L for light chain analysis."
            )
        elif "L" in missing:
            raise ValueError(
                f"Chain parameter '{chain}' requires light chain data, but no light chain loci found. "
                f"Available loci: {', '.join(sorted(present_loci))}. Use --chain H for heavy chain analysis."
            )
    # ===== DETERMINE RECEPTOR TYPE FOR VALIDATION =====
    # Automatically determine receptor type based on model
    bcr_models = {"ablang", "antiberty", "antiberta2", "balm-paired"}
    tcr_models = {"tcr-bert", "tcremp", "tcrt5"}
    protein_models = {"esm2", "prott5", "immune2vec", "custom"}

    if model in bcr_models:
        receptor_type = "BCR"
    elif model in tcr_models:
        receptor_type = "TCR"
    elif model in protein_models:
        receptor_type = "all"  # Protein models can handle both BCR and TCR
    else:
        # Raise error for unknown models
        all_models = bcr_models | tcr_models | protein_models
        raise ValueError(f"Unknown model '{model}'. Supported models are: {', '.join(sorted(all_models))}")

    # ===== VALIDATE DATA TYPE MATCHES MODEL EXPECTATIONS =====
    # Check if the data type matches the model's expected receptor type
    # First, ensure we have locus information (same logic as later in the function)
    data_copy_for_validation = airr.copy()
    if "locus" not in data_copy_for_validation.columns:
        data_copy_for_validation.loc[:, "locus"] = data_copy_for_validation.loc[:, "v_call"].apply(lambda x: x[:3])

    bcr_loci = {"IGH", "IGK", "IGL"}
    tcr_loci = {"TRA", "TRB", "TRG", "TRD"}
    present_loci = set(data_copy_for_validation["locus"].unique())

    has_bcr_data = bool(present_loci & bcr_loci)
    has_tcr_data = bool(present_loci & tcr_loci)

    if receptor_type == "BCR" and not has_bcr_data:
        raise ValueError(
            f"Model '{model}' is a BCR-specific model but no BCR data (IGH, IGK, IGL loci) found. "
            f"Found loci: {', '.join(sorted(present_loci))}. "
            f"Please use BCR data or choose a different model."
        )
    elif receptor_type == "TCR" and not has_tcr_data:
        raise ValueError(
            f"Model '{model}' is a TCR-specific model but no TCR data (TRA, TRB, TRG, TRD loci) found. "
            f"Found loci: {', '.join(sorted(present_loci))}. "
            f"Please use TCR data or choose a different model."
        )
    # For receptor_type == "all", both BCR and TCR data are acceptable

    # ===== DETECT DATA TYPE AND VALIDATE CHAIN COMPATIBILITY =====
    # Check if this is bulk data (no cell_id column) or single-cell data
    is_bulk_data = cell_id_col not in airr.columns

    if is_bulk_data:
        # Bulk data validation: only individual chains (H, L, H+L) are supported
        # Paired chains (HL, LH) require cell_id for pairing and are not supported
        if chain in ["HL", "LH"]:
            raise ValueError(f'chain = "{chain}" invalid for bulk mode')
        logger.info("Detected bulk data format (no cell_id column)")
    else:
        logger.info("Detected single-cell data format")

    # ===== PROCESS DATA =====
    # Dropping rows were sequence column is NA
    n_dat = airr.shape[0]

    airr = airr.dropna(subset=[sequence_col])
    n_dropped = n_dat - airr.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    # Data processing is now moved inside each model for customization
    # Each model has different requirements for chain processing and input format

    # ===== MODEL EXECUTION WITH CHAIN VALIDATION =====
    # BCR models
    if model == "ablang":
        # Check chain compatibility - AbLang supports individual chains only
        if chain in ["HL", "LH"]:
            raise ValueError(
                f"Model 'ablang' was trained on individual chains only. Using --chain H, --chain L, or --chain H+L instead of --chain {chain} is recommended."
            )

        # Process data with unified pattern
        X, dat = process_chain_data(
            airr, chain, sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="standard"
        )

        embedding = ablang(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "antiberta2":
        # Check chain compatibility - AntiBERTa2 supports individual chains only
        if chain in ["HL", "LH"]:
            raise ValueError(
                f"Model 'antiberta2' was trained on individual chains only. Using --chain H, --chain L, or --chain H+L instead of --chain {chain} is recommended."
            )

        # Process data with unified pattern
        X, dat = process_chain_data(
            airr, chain, sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="standard"
        )

        embedding = antiberta2(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "antiberty":
        # Check chain compatibility - AntiBERTy supports individual chains only
        if chain in ["HL", "LH"]:
            raise ValueError(
                f"Model 'antiberty' was trained on individual chains only. Using --chain H, --chain L, or --chain H+L instead of --chain {chain} is recommended."
            )

        # Process data for antiberty
        X, dat = process_chain_data(
            airr, chain, sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="standard"
        )

        embedding = antiberty(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "balm-paired":
        # Check compatible chains - BALM-paired ONLY supports paired chains
        supported_chains = ["HL", "LH"]
        if chain not in supported_chains:
            raise ValueError(
                f"Model 'balm-paired' was trained on paired chains (HL/LH). Using --chain HL or --chain LH instead of --chain {chain} is recommended."
            )
        if chain == "LH":
            warnings.warn(
                "Model 'balm-paired' was trained with H-L order. Using L-H order may reduce accuracy.",
                UserWarning,
            )

        # Process data for BALM-paired - use standard mode for paired chains
        X, dat = process_chain_data(
            airr,
            chain,
            sequence_col,
            cell_id_col,
            duplicate_col,
            receptor_type,
            model_type="standard",
        )

        embedding = balm_paired(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    # TCR models
    elif model == "tcr-bert":
        # TCR-BERT supports all chain types

        # TCR-BERT: Only CDR3, supports H+L, H, L, HL/LH
        X, dat = process_chain_data(
            airr, chain, sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="standard"
        )

        embedding = tcr_bert(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "tcremp":
        # For TCR data, try to get the best CDR3 column
        if has_tcr_data:
            # For TCR data, try to get the best CDR3 column
            effective_sequence_col = get_cdr3_sequence_column(airr, sequence_col)
            if effective_sequence_col == sequence_col and sequence_col not in ["junction_aa", "cdr3_aa"]:
                logger.warning(
                    f"No CDR3-specific columns (junction_aa, cdr3_aa) found. Using '{sequence_col}' column. "
                    f"Note: TCR models (TCR-BERT, TCRT5, TCREMP) were trained on CDR3 sequences, not full VDJ sequences. "
                    f"Using full sequences may reduce embedding accuracy."
                )
        else:
            # For non-TCR data, require explicit CDR3 column
            if sequence_col not in ["CDR3_aa", "junction_aa", "cdr3_aa"]:
                raise ValueError(
                    "Model tcremp was trained on CDR3 aa sequences. For non-TCR data, please provide CDR3_aa, junction_aa, or cdr3_aa as sequence column."
                )
            effective_sequence_col = sequence_col
        # TCREMP supports all chain types

        # TCREMP: All chain modes use tab_locus_gene to get CDR3 + V/J gene information
        raw_dat, dat = process_chain_data(
            airr, chain, effective_sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="tabular"
        )

        # For TCREMP, raw_dat format depends on chain type:
        # - For HL/LH: pivoted format with H, L columns
        # - For H, L, H+L: non-pivoted format with 'chain' column

        # Extract sequences based on chain type
        if chain == "H":
            # H chain: extract from chain column
            if "chain" in raw_dat.columns:
                # Non-pivoted format
                h_data = raw_dat[raw_dat["chain"] == "H"]
                sequences_series = h_data[effective_sequence_col].dropna()
            elif "H" in raw_dat.columns:
                # Pivoted format
                sequences_series = raw_dat["H"].dropna()
            else:
                raise ValueError("No H chain sequences found in data")
        elif chain == "L":
            # L chain: extract from chain column
            if "chain" in raw_dat.columns:
                # Non-pivoted format
                l_data = raw_dat[raw_dat["chain"] == "L"]
                sequences_series = l_data[effective_sequence_col].dropna()
            elif "L" in raw_dat.columns:
                # Pivoted format
                sequences_series = raw_dat["L"].dropna()
            else:
                raise ValueError("No L chain sequences found in data")
        elif chain == "H+L":
            # Both chains separately: combine H and L sequences
            sequences_list = []
            if "chain" in raw_dat.columns:
                # Non-pivoted format
                h_data = raw_dat[raw_dat["chain"] == "H"]
                l_data = raw_dat[raw_dat["chain"] == "L"]
                sequences_list.extend(h_data[effective_sequence_col].dropna().tolist())
                sequences_list.extend(l_data[effective_sequence_col].dropna().tolist())
            else:
                # Pivoted format
                if "H" in raw_dat.columns:
                    h_sequences = raw_dat["H"].dropna()
                    sequences_list.extend(h_sequences.tolist())
                if "L" in raw_dat.columns:
                    l_sequences = raw_dat["L"].dropna()
                    sequences_list.extend(l_sequences.tolist())
            sequences_series = pd.Series(sequences_list)
        elif chain in ["HL", "LH"]:
            # Paired chains: combine H and L sequences per cell
            sequences_list = []
            if "H" in raw_dat.columns and "L" in raw_dat.columns:
                # Pivoted format
                for _, row in raw_dat.iterrows():
                    h_seq = row.get("H", None)
                    l_seq = row.get("L", None)

                    if pd.notna(h_seq) and pd.notna(l_seq):
                        if chain == "LH":
                            sequences_list.append(f"{l_seq}_{h_seq}")
                        else:  # HL
                            sequences_list.append(f"{h_seq}_{l_seq}")
            else:
                raise ValueError(f"Paired chain data not available for chain type '{chain}'")
            sequences_series = pd.Series(sequences_list)
        else:
            raise ValueError(f"Unsupported chain type '{chain}' for TCREMP")

        if len(sequences_series) == 0:
            raise ValueError(f"No valid sequences found for chain type '{chain}' after filtering")

        embedding = tcremp(
            sequences=sequences_series,
            cache_dir=cache_dir,
            batch_size=batch_size,
            chain=chain,
            skip_clustering=skip_clustering,
        )

    elif model == "tcrt5":
        # Check compatible chains - TCRT5 only supports H (beta) chains
        supported_chains = ["H"]
        if chain not in supported_chains:
            warnings.warn(
                f"TCRT5 model was trained on {supported_chains} chains (beta chains for TCR) only. Use --chain H instead of --chain {chain} is recommended.",
                UserWarning,
            )

        # TCRT5: Only CDR3, only supports H (beta) chains
        X, dat = process_chain_data(
            airr, chain, sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="standard"
        )

        embedding = tcrt5(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    # Immune-specific models (BCR & TCR)
    elif model == "immune2vec":
        # Warn about paired chain compatibility
        if chain in ["HL", "LH"]:
            warnings.warn(
                f"Protein language model 'immune2vec' does not have mechanisms to understand paired chain relationships. "
                f"Chain '{chain}' will be processed as concatenated sequences, but results may be inaccurate.",
                UserWarning,
            )

        # Process data for immune2vec
        X, dat = process_chain_data(
            airr, chain, sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="standard"
        )

        embedding = immune2vec(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    # Protein models
    elif model == "esm2":
        # Warn about paired chain compatibility
        if chain in ["HL", "LH"]:
            warnings.warn(
                f"Protein language model 'esm2' does not have mechanisms to understand paired chain relationships. "
                f"Chain '{chain}' will be processed as concatenated sequences, but results may be inaccurate.",
                UserWarning,
            )

        # Process data for esm2
        X, dat = process_chain_data(
            airr, chain, sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="standard"
        )

        embedding = esm2(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "prott5":
        # Warn about paired chain compatibility
        if chain in ["HL", "LH"]:
            warnings.warn(
                f"Protein language model 'prott5' does not have mechanisms to understand paired chain relationships. "
                f"Chain '{chain}' will be processed as concatenated sequences, but results may be inaccurate.",
                UserWarning,
            )

        # Process data for prott5
        X, dat = process_chain_data(
            airr, chain, sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="standard"
        )

        embedding = prott5(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "custom":
        # Warn about paired chain compatibility
        if chain in ["HL", "LH"]:
            warnings.warn(
                f"Custom protein language model does not understand paired chain relationships. "
                f"Chain '{chain}' will be processed as concatenated sequences, but results may be inaccurate.",
                UserWarning,
            )
        if model_path is None or embedding_dimension is None or max_length is None:
            raise ValueError("For custom model, modelpath, embedding_dimension, and max_length must be provided.")

        # Process data for custom model
        X, dat = process_chain_data(
            airr, chain, sequence_col, cell_id_col, duplicate_col, receptor_type, model_type="standard"
        )

        embedding = custommodel(
            sequences=X,
            model_path=model_path,
            embedding_dimension=embedding_dimension,
            max_seq_length=max_length,
            cache_dir=cache_dir,
            batch_size=batch_size,
        )
    else:
        raise ValueError(f"Model {model} not supported.")

    if output_type == "pickle":
        return embedding
    elif output_type == "df":
        allowed_index_cols = ["sequence_id", cell_id_col, "chain"]
        index_cols = [col for col in dat.columns if col in allowed_index_cols]
        embedding_df = pd.DataFrame(embedding.numpy())
        result_df = pd.concat([dat.loc[:, index_cols].reset_index(drop=True), embedding_df], axis=1)
        return result_df


@app.command()
def translate_igblast(
    input_file_path: Annotated[
        str, typer.Argument(..., help="The path to the input data file. The data file should be in AIRR format.")
    ],
    output_dir: Annotated[str, typer.Argument(..., help="The directory where the generated embeddings will be saved.")],
    reference_dir: Annotated[str, typer.Argument(..., help="The directory to the igblast references.")],
    keep_regions: Annotated[
        bool,
        typer.Option(
            help="If True, keeps the region translations in the output airr file. If False, it removes them.",
        ),
    ] = False,
    sequence_col: Annotated[
        str,
        typer.Option(
            "--sequence-col",
            help="The name of the column containing the nucleotide sequences to translate.",
        ),
    ] = "sequence",
):
    """
    Translates nucleotide sequences to amino acid sequences using IgBlast.

    This function takes a AIRR file containing nucleotide sequences
    and translates them into amino acid sequences using IgBlast, a tool for analyzing
    immunoglobulin and T cell receptor sequences. It performs the following steps:\n

    1. Reads the input TSV file containing nucleotide sequences.\n
    2. Writes the nucleotide sequences into a FASTA file, required as input for IgBlast.\n
    3. Runs IgBlast on the FASTA file to perform sequence alignment and translation.\n
    4. Reads the IgBlast output, which includes the translated amino acid sequences.\n
    5. Removes gaps introduced by IgBlast from the sequence alignment.\n
    6. Saves the translated data into a new TSV file in the specified output directory.\n\n
    """
    # Check if IgBlast is available
    if not check_igblast_available():
        stderr.print(
            "[red]Error: IgBlast (igblastn) not found![/red]\n"
            "Please install IgBlast:\n"
            "  1. Add conda channels:\n"
            "     [cyan]conda config --add channels conda-forge[/cyan]\n"
            "     [cyan]conda config --add channels bioconda[/cyan]\n"
            "  2. Install IgBlast:\n"
            "     [cyan]conda install -c bioconda igblast[/cyan]\n"
            "  3. Or use mamba (faster):\n"
            "     [cyan]mamba install -c bioconda igblast[/cyan]\n"
            "  4. Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/igblast/release/\n"
            "Make sure 'igblastn' is in your system PATH."
        )
        raise typer.Exit(1)

    data = pd.read_csv(input_file_path, sep="\t")

    # Output filename
    bn = os.path.splitext(os.path.basename(input_file_path))[0]
    out_translated = os.path.join(output_dir, f"{bn}_translated.tsv")

    data_transl = translate_airr(
        data, tmpdir=None, reference_dir=reference_dir, keep_regions=keep_regions, sequence_col=sequence_col
    )

    logger.info(f"Saved the translations in {out_translated} file.")
    data_transl.to_csv(out_translated, sep="\t", index=False)


@app.command()
def embed(
    input_airr: Annotated[
        str, typer.Option(default=..., help="The path to the input data file. The data file should be in AIRR format.")
    ],
    chain: Annotated[
        str,
        typer.Option(
            default=...,
            help="Input sequences. For BCR: H=Heavy, L=Light, HL=Heavy-Light pairs, LH=Light-Heavy pairs, H+L=Both chains separately. For TCR: H=Beta/Delta, L=Alpha/Gamma, HL=Beta-Alpha/Delta-Gamma pairs, LH=Alpha-Beta/Gamma-Delta pairs, H+L=Both chains separately.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(
            default=...,
            help="The embedding model to use. BCR: ['ablang', 'antiberta2', 'antiberty', 'balm-paired']. TCR: ['tcr-bert', 'tcremp', 'tcrt5']. Immune (BCR & TCR): ['immune2vec']. Protein: ['esm2', 'prott5', 'custom']. Use 'custom' for fine-tuned models with --model-path, --embedding-dimension, and --max-length parameters. Note: TCREMP may require --skip-clustering=True for stability.",
        ),
    ],
    output_file_path: Annotated[
        str,
        typer.Option(
            default=...,
            help="The path where the generated embeddings will be saved. The file extension should be .pt, .csv, or .tsv.",
        ),
    ],
    cache_dir: Annotated[
        str,
        typer.Option(help="Cache dir for storing the pre-trained model weights."),
    ] = "/tmp/amulety",
    sequence_col: Annotated[
        str, typer.Option("--sequence-col", help="The name of the column containing the amino acid sequences to embed.")
    ] = "sequence_vdj_aa",
    cell_id_col: Annotated[
        str, typer.Option(help="The name of the column containing the single-cell barcode.")
    ] = "cell_id",
    batch_size: Annotated[int, typer.Option(help="The batch size of sequences to embed.")] = 50,
    model_path: Annotated[
        str,
        typer.Option(help="Path to custom model (HuggingFace model name or local path). Required for 'custom' model."),
    ] = None,
    embedding_dimension: Annotated[
        int,
        typer.Option(help="Embedding dimension for custom model. Required for 'custom' model."),
    ] = None,
    max_length: Annotated[
        int,
        typer.Option(help="Maximum sequence length for custom model. Required for 'custom' model."),
    ] = None,
    duplicate_col: Annotated[
        str,
        typer.Option(
            "--duplicate-col",
            help="The name of the numeric column used to select the best chain when multiple chains of the same type exist per cell. Default: 'duplicate_count'. Custom columns must be numeric and user-defined.",
        ),
    ] = "duplicate_count",
    skip_clustering: Annotated[
        bool,
        typer.Option(
            "--skip-clustering",
            help="Skip clustering step for TCREMP model (default: True to avoid errors). Only applies to TCREMP model.",
        ),
    ] = True,
):
    """
    Embeds sequences from an AIRR rearrangement file using the specified model.
    Example usage:\n
        amulety embed --chain HL --model antiberta2 --output-file-path out.pt airr_rearrangement.tsv
    """
    out_extension = os.path.splitext(output_file_path)[-1][1:]

    if out_extension not in ["tsv", "csv", "pt"]:
        raise ValueError("Output suffix must be one of ['tsv', 'csv', 'pt']")

    output_type = "pickle" if out_extension == "pt" else "df"

    airr = pd.read_csv(input_airr, sep="\t")

    embedding = embed_airr(
        airr,
        chain,
        model,
        sequence_col=sequence_col,
        cell_id_col=cell_id_col,
        cache_dir=cache_dir,
        batch_size=batch_size,
        embedding_dimension=embedding_dimension,
        max_length=max_length,
        model_path=model_path,
        output_type=output_type,
        duplicate_col=duplicate_col,
        skip_clustering=skip_clustering,
    )

    if output_type == "pickle":
        torch.save(embedding, output_file_path)
    else:
        embedding.to_csv(output_file_path, sep="\t" if out_extension == "tsv" else ",", index=False)
    logger.info("Saved embedding at %s", output_file_path)


@app.command()
def check_deps():
    """Check if optional embedding dependencies and tools are installed."""

    print("Checking AMULETY dependencies...\n")

    # Check IgBlast availability
    print("🧬 IgBlast (for translate-igblast command):")
    if check_igblast_available():
        print("  ✅ IgBlast (igblastn) is available")
    else:
        print("  ❌ IgBlast (igblastn) not found")
        print("     1. Add conda channels:")
        print("        conda config --add channels conda-forge")
        print("        conda config --add channels bioconda")
        print("     2. Install IgBlast:")
        print("        conda install -c bioconda igblast")
        print("     3. Or use mamba: mamba install -c bioconda igblast")
        print("     4. Or download from: https://ftp.ncbi.nlm.nih.gov/blast/executables/igblast/release/")

    print("\n🤖 Embedding model dependencies:")
    missing = check_dependencies()

    if not missing:
        print("  ✅ All embedding dependencies are installed!")
    else:
        print(f"  ❌ {len(missing)} dependencies are missing.")
        print("  AMULETY will raise ImportError with installation instructions when these models are used.")
        print("\n  To install missing dependencies:")
        for name, install_cmd in missing:
            print(f"    • {name}: {install_cmd}")
        print("\n  Note: Models will provide detailed installation instructions when used.")


def main():
    asci_art = r"""
 █████  ███    ███ ██    ██ ██      ███████ ████████     ██    ██
██   ██ ████  ████ ██    ██ ██      ██         ██         ██  ██
███████ ██ ████ ██ ██    ██ ██      █████      ██          ████
██   ██ ██  ██  ██ ██    ██ ██      ██         ██           ██
██   ██ ██      ██  ██████  ███████ ███████    ██           ██
"""
    stderr.print(asci_art)
    stderr.print(f"AMULETY: Adaptive imMUne receptor Language model Embedding Tool\n version {__version__}\n")

    app()


if __name__ == "amulety":
    main()
