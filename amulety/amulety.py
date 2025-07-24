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
    process_airr,
)

__version__ = version("amulety")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()
stderr = Console(stderr=True)
stdout = Console()


def translate_airr(
    airr: pd.DataFrame, tmpdir: str, reference_dir: str, keep_regions: bool = False, sequence_col: str = "sequence"
):
    """
    Translates nucleotide sequences to amino acid sequences using IgBlast.
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
    pipes = subprocess.Popen(command_igblastn, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = pipes.communicate()

    if pipes.returncode != 0:
        raise Exception(f"IgBlast failed with error code {pipes.returncode}. {stderr.decode('utf-8')}")

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
    selection_col: str = "duplicate_count",
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
        selection_col (str): The name of the numeric column used to select the best chain when
                           multiple chains of the same type exist per cell. Default: "duplicate_count".

    """
    # Check valid chain - unified interface for both BCR and TCR
    valid_chains = ["H", "L", "HL", "LH", "H+L"]
    if chain not in valid_chains:
        raise ValueError(f"Input chain must be one of {valid_chains}")

    # Warning for LH order - models are trained on HL order
    if chain == "LH":
        warnings.warn(
            "LH (Light-Heavy) chain order detected. Most paired models are trained on HL (Heavy-Light) order. "
            "Using LH order may result in reduced accuracy. Consider using --chain HL for better performance.",
            UserWarning,
        )

    # Use the chain parameter directly - no mapping needed
    internal_chain = chain
    if output_type not in ["df", "pickle"]:
        raise ValueError("Output type must be one of ['df', 'pickle']")
    if sequence_col not in airr.columns:
        raise ValueError(f"Column {sequence_col} not found in the input AIRR data.")

    # ===== BASIC CHAIN VALIDATION =====
    # Check if requested chains are available in the data
    available_chains = set()
    if "sequence_vdj_aa" in airr.columns and airr["sequence_vdj_aa"].notna().any():
        available_chains.add("H")
    if "sequence_vj_aa" in airr.columns and airr["sequence_vj_aa"].notna().any():
        available_chains.add("L")

    # Validate chain availability
    if chain == "H" and "H" not in available_chains:
        raise ValueError("Chain 'H' requested but no heavy chain sequences found in 'sequence_vdj_aa' column")
    elif chain == "L" and "L" not in available_chains:
        raise ValueError("Chain 'L' requested but no light chain sequences found in 'sequence_vj_aa' column")
    elif chain in ["HL", "LH", "H+L"] and not available_chains.issuperset({"H", "L"}):
        missing = {"H", "L"} - available_chains
        raise ValueError(f"Chain '{chain}' requested but missing chains: {', '.join(missing)}")
    # ===== PROCESS DATA =====

    dat = process_airr(
        airr, internal_chain, sequence_col=sequence_col, cell_id_col=cell_id_col, selection_col=selection_col
    )
    n_dat = dat.shape[0]

    dat = dat.dropna(subset=[sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:, sequence_col]

    # ===== MODEL EXECUTION WITH CHAIN VALIDATION =====
    # BCR models
    if model == "ablang":
        # Check compatible chains
        supported_chains = ["H", "L", "H+L"]
        if chain not in supported_chains:
            raise ValueError(f"Model ablang only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        embedding = ablang(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "antiberta2":
        # Check compatible chains
        supported_chains = ["H", "L", "H+L"]
        if chain not in supported_chains:
            raise ValueError(f"Model antiberta2 only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        embedding = antiberta2(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "antiberty":
        # Check compatible chains
        supported_chains = ["H", "L", "H+L"]
        if chain not in supported_chains:
            raise ValueError(f"Model antiberty only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        embedding = antiberty(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "balm-paired":
        # Check compatible chains
        supported_chains = ["HL", "LH"]
        if chain not in supported_chains:
            raise ValueError(f"Model balm-paired only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        embedding = balm_paired(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    # TCR models
    elif model == "tcr-bert":
        # Check compatible chains
        supported_chains = ["H", "L", "HL", "LH", "H+L"]
        if chain not in supported_chains:
            raise ValueError(f"Model tcr-bert only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        embedding = tcr_bert(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "tcremp":
        # Check compatible chains
        supported_chains = ["H", "L", "HL", "LH", "H+L"]
        if chain not in supported_chains:
            raise ValueError(f"Model tcremp only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        embedding = tcremp(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "tcrt5":
        # Check compatible chains
        supported_chains = ["H"]
        if chain not in supported_chains:
            raise ValueError(f"Model tcrt5 only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        embedding = tcrt5(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    # Immune-specific models (BCR & TCR)
    elif model == "immune2vec":
        # Check compatible chains (with warning for paired chains)
        supported_chains = ["H", "L", "HL", "LH", "H+L"]
        if chain not in supported_chains:
            raise ValueError(f"Model immune2vec only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        if chain in ["HL", "LH"]:
            warnings.warn(
                f"Protein language model 'immune2vec' does not understand paired chain relationships. "
                f"Chain '{chain}' will be processed as concatenated sequences, but results may be inaccurate.",
                UserWarning,
            )
        embedding = immune2vec(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    # Protein models
    elif model == "esm2":
        # Check compatible chains (with warning for paired chains)
        supported_chains = ["H", "L", "HL", "LH", "H+L"]
        if chain not in supported_chains:
            raise ValueError(f"Model esm2 only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        if chain in ["HL", "LH"]:
            warnings.warn(
                f"Protein language model 'esm2' does not understand paired chain relationships. "
                f"Chain '{chain}' will be processed as concatenated sequences, but results may be inaccurate.",
                UserWarning,
            )
        embedding = esm2(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "prott5":
        # Check compatible chains (with warning for paired chains)
        supported_chains = ["H", "L", "HL", "LH", "H+L"]
        if chain not in supported_chains:
            raise ValueError(f"Model prott5 only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        if chain in ["HL", "LH"]:
            warnings.warn(
                f"Protein language model 'prott5' does not understand paired chain relationships. "
                f"Chain '{chain}' will be processed as concatenated sequences, but results may be inaccurate.",
                UserWarning,
            )
        embedding = prott5(sequences=X, cache_dir=cache_dir, batch_size=batch_size)

    elif model == "custom":
        # Check compatible chains (with warning for paired chains)
        supported_chains = ["H", "L", "HL", "LH", "H+L"]
        if chain not in supported_chains:
            raise ValueError(f"Model custom only accepts {', '.join(supported_chains)} inputs! Got: {chain}")
        if chain in ["HL", "LH"]:
            warnings.warn(
                f"Custom protein language model does not understand paired chain relationships. "
                f"Chain '{chain}' will be processed as concatenated sequences, but results may be inaccurate.",
                UserWarning,
            )
        if model_path is None or embedding_dimension is None or max_length is None:
            raise ValueError("For custom model, modelpath, embedding_dimension, and max_length must be provided.")
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
            help="The embedding model to use. BCR: ['ablang', 'antiberta2', 'antiberty', 'balm-paired']. TCR: ['tcr-bert', 'tcremp', 'tcrt5']. Immune (BCR & TCR): ['immune2vec']. Protein: ['esm2', 'prott5', 'custom']. Use 'custom' for fine-tuned models with --model-path, --embedding-dimension, and --max-length parameters.",
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
    selection_col: Annotated[
        str,
        typer.Option(
            help="The name of the numeric column used to select the best chain when multiple chains of the same type exist per cell. Default: 'duplicate_count'. Custom columns must be numeric and user-defined."
        ),
    ] = "duplicate_count",
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
        selection_col=selection_col,
    )

    if output_type == "pickle":
        torch.save(embedding, output_file_path)
    else:
        embedding.to_csv(output_file_path, sep="\t" if out_extension == "tsv" else ",", index=False)
    logger.info("Saved embedding at %s", output_file_path)


@app.command()
def check_deps():
    """Check if optional TCR embedding dependencies are installed."""
    from amulety.tcr_embeddings import check_tcr_dependencies

    print("Checking TCR embedding dependencies...")
    missing = check_tcr_dependencies()

    if not missing:
        print("✓ All TCR embedding dependencies are installed!")
    else:
        print(f"\n❌ {len(missing)} dependencies are missing.")
        print("AMULETY will raise ImportError with installation instructions when these models are used.")
        print("\nTo install missing dependencies:")
        for name, install_cmd in missing:
            print(f"  • {name}: {install_cmd}")
        print("\nNote: Models will provide detailed installation instructions when used.")


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
