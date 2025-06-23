"""Console script for amulety"""
import logging
import os
import subprocess
import time
from importlib.metadata import version

import pandas as pd
import torch
import typer
from rich.console import Console
from typing_extensions import Annotated

from amulety.bcr_embeddings import antiberta2, antiberty, balm_paired
from amulety.protein_embeddings import custommodel, esm2, prott5
from amulety.tcr_embeddings import deep_tcr, tcr_bert, tcremp, trex
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
):
    """
    Embeds sequences from an AIRR DataFrame using the specified model.
    Parameters:
        airr (pd.DataFrame): Input AIRR rearrangement table as a pandas DataFrame.
        chain (str): The input chain, which can be one of ["H", "L", "HL"].
                    For BCR: H=Heavy, L=Light, HL=Heavy-Light pairs
                    For TCR: H=Beta/Delta, L=Alpha/Gamma, HL=Beta-Alpha/Delta-Gamma pairs
        model (str): The embedding model to use.
        sequence_col (str): The name of the column containing the amino acid sequences to embed.
        cell_id_col (str): The name of the column containing the single-cell barcode.
        cache_dir (Optional[str]): Cache dir for storing the pre-trained model weights.
        batch_size (int): The batch size of sequences to embed.
        output_type (str): The type of output to return. Can be "df" for a pandas DataFrame or "pickle" for a serialized torch object.
    """
    # Check valid chain - unified interface for both BCR and TCR
    valid_chains = ["H", "L", "HL"]
    if chain not in valid_chains:
        raise ValueError(f"Input chain must be one of {valid_chains}")

    # Use the chain parameter directly - no mapping needed
    internal_chain = chain
    if output_type not in ["df", "pickle"]:
        raise ValueError("Output type must be one of ['df', 'pickle']")
    if sequence_col not in airr.columns:
        raise ValueError(f"Column {sequence_col} not found in the input AIRR data.")

    dat = process_airr(airr, internal_chain, sequence_col=sequence_col, cell_id_col=cell_id_col)
    n_dat = dat.shape[0]

    dat = dat.dropna(subset=[sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:, sequence_col]

    # BCR models
    if model == "antiberta2":
        embedding = antiberta2(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
    elif model == "antiberty":
        embedding = antiberty(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
    elif model == "balm-paired":
        embedding = balm_paired(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
    # TCR models
    elif model == "deep-tcr":
        embedding = deep_tcr(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
    elif model == "tcr-bert":
        embedding = tcr_bert(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
    elif model == "tcremp":
        embedding = tcremp(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
    elif model == "trex":
        embedding = trex(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
    # Protein models
    elif model == "esm2":
        embedding = esm2(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
    elif model == "prott5":
        embedding = prott5(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
    elif model == "custom":
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
        allowed_index_cols = ["sequence_id", cell_id_col]
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
            help="Input sequences. For BCR: H=Heavy, L=Light, HL=Heavy-Light pairs. For TCR: H=Beta/Delta, L=Alpha/Gamma, HL=Beta-Alpha/Delta-Gamma pairs.",
        ),
    ],
    model: Annotated[
        str,
        typer.Option(default=..., help="The embedding model to use."),
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
        output_type=output_type,
    )

    if output_type == "pickle":
        torch.save(embedding, output_file_path)
    else:
        embedding.to_csv(output_file_path, sep="\t" if out_extension == "tsv" else ",", index=False)
    logger.info("Saved embedding at %s", output_file_path)


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
