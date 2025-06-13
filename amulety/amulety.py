"""Console script for amulety"""
import logging
import math
import os
import subprocess
import time
from importlib.metadata import version
from typing import Optional

import pandas as pd
import torch
import typer
from airr import validate_airr
from antiberty import AntiBERTyRunner
from rich.console import Console
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    RoFormerForMaskedLM,
    RoFormerTokenizer,
)
from typing_extensions import Annotated

from amulety.utils import (
    batch_loader,
    insert_space_every_other_except_cls,
    process_airr,
)

__version__ = version("amulety")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()
stderr = Console(stderr=True)
stdout = Console()


def antiberty(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
):
    """
    Embeds sequences using the AntiBERTy model.\n
    The maximum length of the sequences to be embedded is 510.
    """
    max_seq_length = 510

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    X = X.str.replace("<cls><cls>", "[CLS][CLS]")
    X = X.apply(insert_space_every_other_except_cls)
    sequences = X.str.replace("  ", " ")

    antiberty_runner = AntiBERTyRunner()
    model_size = sum(p.numel() for p in antiberty_runner.model.parameters())
    logger.info("AntiBERTy loaded. Size: %s M", round(model_size / 1e6, 2))
    start_time = time.time()
    n_seqs = len(sequences)
    dim = max_seq_length + 2

    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("Batch %s/%s", i, n_batches)
        x = antiberty_runner.embed(batch)
        x = [a.mean(axis=0) for a in x]
        embeddings[start:end] = torch.stack(x)
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))
    return embeddings


def antiberta2(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
):
    """
    Embeds sequences using the antiBERTa2 RoFormer model.\n
    The maximum length of the sequences to be embedded is 256.
    """
    max_seq_length = 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    X = X.str.replace("<cls><cls>", "[CLS][CLS]")
    X = X.apply(insert_space_every_other_except_cls)
    X = X.str.replace("  ", " ")
    sequences = X.values

    tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2", cache_dir=cache_dir)
    model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2", cache_dir=cache_dir)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("AntiBERTa2 loaded. Size: %s M", model_size / 1e6)

    start_time = time.time()
    n_seqs = len(sequences)
    dim = 1024
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
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


def esm2(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
):
    """
    Embeds sequences using the ESM2 model. The maximum length of the sequences to be embedded is 512. The embedding dimension is 1280.
    """
    max_seq_length = 512
    dim = 1280
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir=cache_dir)
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir=cache_dir)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("ESM2 650M model size: %s M", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
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


def custommodel(
    sequences: pd.Series,
    model_path: str,
    embedding_dimension: int,
    max_seq_length: int,
    cache_dir: Optional[str] = "/tmp/amulety",
    batch_size: Optional[int] = 50,
):
    """
    Embeds sequences using a custom model specified by the user. The maximum length of the sequences to be embedded is specified by the user.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForMaskedLM.from_pretrained(model_path)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("Model size: %sM", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, embedding_dimension))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        print(f"Batch {i}/{n_batches}\n")
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
    sequences: pd.Series,
    cache_dir: str = "/tmp/amulety",
    batch_size: int = 50,
):
    """
    Embeds sequences using the BALM-paired model. The maximum length of the sequences to be embedded is 1024. The embedding dimension is 1024.
    """
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)

    # Download BALM-paired model if not already cached
    model_name = "BALM-paired_LC-coherence_90-5-5-split_122222"
    model_path = os.path.join(cache_dir, model_name)
    embedding_dimension = 1024
    max_seq_length = 510

    if not os.path.exists(model_path):
        try:
            # Download and extract model
            command = f"""
                wget -O {os.path.join(cache_dir, "BALM-paired.tar.gz")} https://zenodo.org/records/8237396/files/BALM-paired.tar.gz
                tar -xzf {os.path.join(cache_dir, "BALM-paired.tar.gz")} -C {cache_dir}
                rm {os.path.join(cache_dir, "BALM-paired.tar.gz")}
            """
            subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading or extracting model: {e}")
            return

    embeddings = custommodel(
        sequences=sequences,
        model_path=model_path,
        embedding_dimension=embedding_dimension,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        cache_dir=cache_dir,
    )
    return embeddings


def translate_airr(airr: pd.DataFrame, tmpdir: str, reference_dir: str):
    """
    Translates nucleotide sequences to amino acid sequences using IgBlast.
    """
    data = airr.copy()

    if not validate_airr(data):
        raise ValueError("The input data is not in a valid AIRR rearrangement schema.")

    # Warn if translations already exist
    columns_reserved = ["sequence_aa", "sequence_alignment_aa", "sequence_vdj_aa"]
    overlap = [col for col in data.columns if col in columns_reserved]
    if len(overlap) > 0:
        logger.warning("Existing amino acid columns (%s) will be overwritten.", ", ".join(overlap))
        data = data.drop(overlap, axis=1)

    if tmpdir is None:
        tmpdir = os.path.join(os.getcwd(), "tmp")
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
            f.write(row["sequence"] + "\n")

    # Run IgBlast on FASTA
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

    # Read IgBlast output
    igblast_transl = pd.read_csv(out_igblast, sep="\t", usecols=["sequence_id", "sequence_aa", "sequence_alignment_aa"])

    # Remove IMGT gaps
    sequence_vdj_aa = [sa.replace("-", "") for sa in igblast_transl["sequence_alignment_aa"]]
    igblast_transl["sequence_vdj_aa"] = sequence_vdj_aa

    logger.info(
        "Saved the translations in the dataframe (sequence_aa contains the full translation and sequence_vdj_aa contains the VDJ translation)."
    )
    # Merge and save the translated data with original data
    data_transl = pd.merge(data, igblast_transl, on="sequence_id", how="left")

    # Clean up
    os.rmdir(tmpdir, recursive=True)

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
        model (str): The embedding model to use. Currently, only "antiberta2" is supported.
        sequence_col (str): The name of the column containing the amino acid sequences to embed.
        cell_id_col (str): The name of the column containing the single-cell barcode.
        cache_dir (Optional[str]): Cache dir for storing the pre-trained model weights.
        batch_size (int): The batch size of sequences to embed.
        output_type (str): The type of output to return. Can be "df" for a pandas DataFrame or "pickle" for a serialized torch object.
    """
    # Check valid chain
    if chain not in ["H", "L", "HL"]:
        raise ValueError("Input x must be one of ['H', 'L', 'HL']")
    if output_type not in ["df", "pickle"]:
        raise ValueError("Output type must be one of ['df', 'pickle']")

    dat = process_airr(airr, chain, sequence_col=sequence_col, cell_id_col=cell_id_col)
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
    # Protein models
    elif model == "esm2":
        embedding = esm2(sequences=X, cache_dir=cache_dir, batch_size=batch_size)
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

    data_transl = translate_airr(data, tmpdir=None, reference_dir=reference_dir)

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
            help="Input sequences (H for heavy chain, L for light chain, HL for heavy and light concatenated)",
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
        str, typer.Option(help="The name of the column containing the amino acid sequences to embed.")
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
