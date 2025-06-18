"""Console script for amulety"""
import logging
import math
import os
import subprocess
import time

import pandas as pd
import torch
import typer
from antiberty import AntiBERTyRunner
from rich.console import Console
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    RoFormerForMaskedLM,
    RoFormerTokenizer,
)
from typing_extensions import Annotated

from amulety import __version__
from amulety.utils import (
    batch_loader,
    check_output_file_type,
    insert_space_every_other_except_cls,
    process_airr,
    save_embedding,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()
stderr = Console(stderr=True)
stdout = Console()


@app.command()
def antiberty(
    input_file_path: Annotated[
        str, typer.Argument(..., help="The path to the input data file. The data file should be in AIRR format.")
    ],
    chain: Annotated[
        str,
        typer.Argument(
            ..., help="Input sequences (H for heavy chain, L for light chain, HL for heavy and light concatenated)"
        ),
    ],
    output_file_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="The path where the generated embeddings will be saved. The file extension should be .pt, .csv, or .tsv.",
        ),
    ],
    sequence_col: Annotated[
        str, typer.Option(help="The name of the column containing the amino acid sequences to embed.")
    ] = "sequence_vdj_aa",
    cell_id_col: Annotated[
        str, typer.Option(help="The name of the column containing the single-cell barcode.")
    ] = "cell_id",
    batch_size: Annotated[int, typer.Option(help="The batch size of sequences to embed.")] = 500,
):
    """
    Embeds sequences using the AntiBERTy model.\n

    Note:\n
    This function prints the number of sequences being embedded, the batch number during the
    embedding process, the time taken for the embedding, and the location where the embeddings are saved.\n\n

    Example usage:\n
        amulety antiberty tests/AIRR_rearrangement_translated_single-cell.tsv HL out.pt

    """
    out_format = check_output_file_type(output_file_path)
    dat = process_airr(input_file_path, chain, sequence_col=sequence_col, cell_id_col=cell_id_col, receptor_type="BCR")
    logger.info("Embedding %s sequences using antiberty...", dat.shape[0])
    max_length = 512 - 2
    n_dat = dat.shape[0]

    dat = dat.dropna(subset=[sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:, sequence_col]
    X = X.apply(lambda a: a[:max_length])
    X = X.str.replace("<cls><cls>", "[CLS][CLS]")
    X = X.apply(insert_space_every_other_except_cls)
    sequences = X.str.replace("  ", " ")

    antiberty_runner = AntiBERTyRunner()
    model_size = sum(p.numel() for p in antiberty_runner.model.parameters())
    logger.info("AntiBERTy loaded. Size: %s M", round(model_size / 1e6, 2))
    start_time = time.time()
    n_seqs = len(sequences)
    dim = 512

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

    save_embedding(dat, embeddings, output_file_path, out_format, cell_id_col)
    logger.info("Saved embedding at %s", output_file_path)


@app.command()
def antiberta2(
    input_file_path: Annotated[
        str, typer.Argument(..., help="The path to the input data file. The data file should be in AIRR format.")
    ],
    chain: Annotated[
        str,
        typer.Argument(
            ..., help="Input sequences (H for heavy chain, L for light chain, HL for heavy and light concatenated)"
        ),
    ],
    output_file_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="The path where the generated embeddings will be saved. The file extension should be .pt, .csv, or .tsv.",
        ),
    ],
    cache_dir: Annotated[
        str,
        typer.Option(help="Cache dir for storing the pre-trained model weights."),
    ] = None,
    sequence_col: Annotated[
        str, typer.Option(help="The name of the column containing the amino acid sequences to embed.")
    ] = "sequence_vdj_aa",
    cell_id_col: Annotated[
        str, typer.Option(help="The name of the column containing the single-cell barcode.")
    ] = "cell_id",
    batch_size: Annotated[int, typer.Option(help="The batch size of sequences to embed.")] = 128,
):
    """
    Embeds sequences using the antiBERTa2 RoFormer model.\n

    Note:\n
    This function uses the ESM2 model for embedding. The maximum length of the sequences to be embedded is 512.
    It prints the size of the model used for embedding, the batch number during the embedding process,
    and the time taken for the embedding. The embeddings are saved at the location specified by `output_file_path`.\n\n

    Example Usage:\n
        amulety antiberta2 tests/AIRR_rearrangement_translated_single-cell.tsv HL out.pt\n
    """
    out_format = check_output_file_type(output_file_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dat = process_airr(input_file_path, chain, sequence_col=sequence_col, cell_id_col=cell_id_col, receptor_type="BCR")
    max_length = 256
    n_dat = dat.shape[0]

    dat = dat.dropna(subset=[sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:, sequence_col]
    X = X.apply(lambda a: a[:max_length])
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
                    seq, padding="max_length", truncation=True, max_length=max_length, return_special_tokens_mask=True
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

    save_embedding(dat, embeddings, output_file_path, out_format, cell_id_col)
    logger.info("Saved embedding at %s", output_file_path)


@app.command()
def esm2(
    input_file_path: Annotated[
        str, typer.Argument(..., help="The path to the input data file. The data file should be in AIRR format.")
    ],
    chain: Annotated[
        str,
        typer.Argument(
            ...,
            help="Chain type: 'H' (heavy), 'L' (light), 'HL' (heavy-light pairs) for BCR; 'A' (alpha), 'B' (beta), 'AB' (alpha-beta pairs) for TCR",
        ),
    ],
    output_file_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="The path where the generated embeddings will be saved. The file extension should be .pt, .csv, or .tsv.",
        ),
    ],
    cache_dir: Annotated[
        str,
        typer.Option(help="Cache dir for storing the pre-trained model weights."),
    ] = None,
    sequence_col: Annotated[
        str, typer.Option(help="The name of the column containing the amino acid sequences to embed.")
    ] = "sequence_vdj_aa",
    cell_id_col: Annotated[
        str, typer.Option(help="The name of the column containing the single-cell barcode.")
    ] = "cell_id",
    batch_size: Annotated[int, typer.Option(help="The batch size of sequences to embed.")] = 50,
):
    """
    Embeds sequences using the ESM2 model.

    Example usage:\n
        amulety esm2 tests/AIRR_rearrangement_translated_single-cell.tsv HL out.pt\n\n

    Note:\n
    This function uses the ESM2 model for embedding. The maximum length of the sequences to be embedded is 512.
    It prints the size of the model used for embedding, the batch number during the embedding process,
    and the time taken for the embedding. The embeddings are saved at the location specified by `output_file_path`.
    """
    out_format = check_output_file_type(output_file_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dat = process_airr(input_file_path, chain, sequence_col=sequence_col, cell_id_col=cell_id_col, receptor_type="all")
    max_length = 512
    n_dat = dat.shape[0]

    dat = dat.dropna(subset=[sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:, sequence_col]
    X = X.apply(lambda a: a[:max_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir=cache_dir)
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D", cache_dir=cache_dir)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("ESM2 650M model size: %s M", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    dim = 1280
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("Batch %s/%s.", i, n_batches)
        x = torch.tensor(
            [
                tokenizer.encode(
                    seq, padding="max_length", truncation=True, max_length=max_length, return_special_tokens_mask=True
                )
                for seq in batch
            ]
        ).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
            outputs = outputs.hidden_states[-1]
            outputs = list(outputs.detach())

        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1, :].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))

    save_embedding(dat, embeddings, output_file_path, out_format, cell_id_col)
    logger.info("Saved embedding at %s", output_file_path)


@app.command()
def custommodel(
    modelpath: Annotated[str, typer.Argument(..., help="The path to the pretrained model.")],
    input_file_path: Annotated[
        str, typer.Argument(..., help="The path to the input data file. The data file should be in AIRR format.")
    ],
    chain: Annotated[
        str,
        typer.Argument(
            ..., help="Input sequences (H for heavy chain, L for light chain, HL for heavy and light concatenated)"
        ),
    ],
    output_file_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="The path where the generated embeddings will be saved. The file extension should be .pt, .csv, or .tsv.",
        ),
    ],
    embedding_dimension: Annotated[int, typer.Option(help="The dimension of the embedding layer.")] = 100,
    max_length: Annotated[int, typer.Option(help="The maximum length that the model can take.")] = 512,
    batch_size: Annotated[int, typer.Option(help="The batch size of sequences to embed.")] = 50,
    sequence_col: Annotated[
        str, typer.Option(help="The name of the column containing the amino acid sequences to embed.")
    ] = "sequence_vdj_aa",
    cell_id_col: Annotated[
        str, typer.Option(help="The name of the column containing the single-cell barcode.")
    ] = "cell_id",
):
    """
    This function generates embeddings for a given dataset using a pretrained model. The function first checks if a CUDA device is available for PyTorch to use. It then loads the data from the input file and preprocesses it.
    The sequences are tokenized and fed into the pretrained model to generate embeddings. The embeddings are then saved to the specified output path.\n\n

    Note:\n
    This function uses the transformers library's AutoTokenizer and AutoModelForMaskedLM classes to handle the tokenization and model loading.\n\n

    Example Usage:\n
        amulety custom_model <custom_model_path> tests/AIRR_rearrangement_translated_single-cell.tsv HL out.pt\n

    """
    out_format = check_output_file_type(output_file_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dat = process_airr(input_file_path, chain, sequence_col=sequence_col, cell_id_col=cell_id_col, receptor_type="BCR")
    X = dat.loc[:, sequence_col]
    X = X.apply(lambda a: a[:max_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForMaskedLM.from_pretrained(modelpath)
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
                    seq, padding="max_length", truncation=True, max_length=max_length, return_special_tokens_mask=True
                )
                for seq in batch
            ]
        ).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
            outputs = outputs.hidden_states[-1]
            outputs = list(outputs.detach())

        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1, :].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))

    save_embedding(dat, embeddings, output_file_path, out_format, cell_id_col)
    logger.info("Saved embedding at %s", output_file_path)


@app.command()
def balm_paired(
    input_file_path: Annotated[
        str, typer.Argument(..., help="The path to the input data file. The data file should be in AIRR format.")
    ],
    chain: Annotated[
        str,
        typer.Argument(
            ..., help="Input sequences (H for heavy chain, L for light chain, HL for heavy and light concatenated)"
        ),
    ],
    output_file_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="The path where the generated embeddings will be saved. The file extension should be .pt, .csv, or .tsv.",
        ),
    ],
    cache_dir: Annotated[
        str,
        typer.Option(help="Cache dir for storing the pre-trained model weights."),
    ] = "/tmp/amulety/",
    sequence_col: Annotated[
        str, typer.Option(help="The name of the column containing the amino acid sequences to embed.")
    ] = "sequence_vdj_aa",
    cell_id_col: Annotated[
        str, typer.Option(help="The name of the column containing the single-cell barcode.")
    ] = "cell_id",
    batch_size: Annotated[int, typer.Option(help="The batch size of sequences to embed.")] = 50,
):
    """
    Embeds sequences using the BALM-paired model.

    Example usage:\n
        amulety balm-paired tests/AIRR_rearrangement_translated_single-cell.tsv HL out.pt\n\n

    Note:\n
    This function uses the BALM-paired model for embedding. The maximum length of the sequences to be embedded is 1024.
    It prints the size of the model used for embedding, the batch number during the embedding process,
    and the time taken for the embedding. The embeddings are saved at the location specified by `output_file_path`.
    """

    os.makedirs(cache_dir, exist_ok=True)

    model_name = "BALM-paired_LC-coherence_90-5-5-split_122222"
    model_path = os.path.join(cache_dir, model_name)

    if not os.path.exists(model_path):
        try:
            command = f"""
                wget -O {os.path.join(cache_dir, "BALM-paired.tar.gz")} https://zenodo.org/records/8237396/files/BALM-paired.tar.gz
                tar -xzf {os.path.join(cache_dir, "BALM-paired.tar.gz")} -C {cache_dir}
                rm {os.path.join(cache_dir, "BALM-paired.tar.gz")}
            """
            subprocess.run(command, shell=True, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"Error downloading or extracting model: {e}")
            return

    custommodel(
        model_path,
        input_file_path,
        chain,
        output_file_path,
        embedding_dimension=1024,
        batch_size=batch_size,
        max_length=510,
        sequence_col=sequence_col,
        cell_id_col=cell_id_col,
    )


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

    columns_reserved = ["sequence_aa", "sequence_alignment_aa", "sequence_vdj_aa"]
    overlap = [col for col in data.columns if col in columns_reserved]
    if len(overlap) > 0:
        logger.warn("Existing amino acid columns (%s) will be overwritten.", ", ".join(overlap))
        data = data.drop(overlap, axis=1)

    out_fasta = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file_path))[0] + ".fasta")
    out_igblast = os.path.join(output_dir, os.path.splitext(os.path.basename(input_file_path))[0] + "_igblast.tsv")
    out_translated = os.path.join(
        output_dir, os.path.splitext(os.path.basename(input_file_path))[0] + "_translated.tsv"
    )

    start_time = time.time()
    logger.info("Converting AIRR table to FastA for IgBlast translation...")

    with open(out_fasta, "w") as f:
        for _, row in data.iterrows():
            f.write(">" + row["sequence_id"] + "\n")
            f.write(row["sequence"] + "\n")

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

    igblast_transl = pd.read_csv(out_igblast, sep="\t", usecols=["sequence_id", "sequence_aa", "sequence_alignment_aa"])

    sequence_vdj_aa = [sa.replace("-", "") for sa in igblast_transl["sequence_alignment_aa"]]
    igblast_transl["sequence_vdj_aa"] = sequence_vdj_aa

    logger.info(
        "Saved the translations in the dataframe (sequence_aa contains the full translation and sequence_vdj_aa contains the VDJ translation)."
    )

    data_transl = pd.merge(data, igblast_transl, on="sequence_id", how="left")

    logger.info(f"Saved the translations in {out_translated} file.")
    data_transl.to_csv(out_translated, sep="\t", index=False)

    os.remove(out_fasta)
    os.remove(out_igblast)

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))


# TCR-specific commands
@app.command()
def prott5(
    input_file_path: Annotated[
        str, typer.Argument(..., help="The path to the input AIRR format file containing BCR or TCR sequences.")
    ],
    chain: Annotated[
        str,
        typer.Argument(
            ...,
            help="Chain type: 'H' (heavy), 'L' (light), 'HL' (heavy-light pairs) for BCR; 'A' (alpha), 'B' (beta), 'AB' (alpha-beta pairs) for TCR",
        ),
    ],
    output_file_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="Output path for embeddings. Supports .pt (PyTorch), .csv, or .tsv formats.",
        ),
    ],
    cache_dir: Annotated[
        str,
        typer.Option(help="Directory to cache ProtT5 model weights (default: system cache)."),
    ] = None,
    sequence_col: Annotated[str, typer.Option(help="Column name containing amino acid sequences.")] = "sequence_vdj_aa",
    cell_id_col: Annotated[str, typer.Option(help="Column name containing single-cell barcodes.")] = "cell_id",
    batch_size: Annotated[int, typer.Option(help="Batch size for processing (adjust based on GPU memory).")] = 32,
):
    """
    Embeds BCR or TCR sequences using the ProtT5 protein language model.

    Example usage:\n
        amulety prott5 data.tsv HL embeddings.pt\n\n

    Note:\n
    General protein language model pre-trained on UniRef50.
    Suitable for diverse protein tasks and large-scale transfer learning. The maximum length of the sequences to be embedded is 1024.
    It prints the size of the model used for embedding, the batch number during the embedding process,
    and the time taken for the embedding. The embeddings are saved at the location specified by `output_file_path`.
    """
    out_format = check_output_file_type(output_file_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # map TCR chain notation to BCR notation for compatibility
    # A (alpha) -> L (light), B (beta) -> H (heavy), AB -> HL
    bcr_chain = {"A": "L", "B": "H", "AB": "HL"}.get(chain, chain)
    dat = process_airr(
        input_file_path, bcr_chain, sequence_col=sequence_col, cell_id_col=cell_id_col, receptor_type="all"
    )
    max_length = 1024  # ProtT5 can't handle longer sequences
    n_dat = dat.shape[0]

    dat = dat.dropna(subset=[sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:, sequence_col]
    X = X.apply(lambda a: a[:max_length])

    # ProtT5 expects space-separated amino acids
    X = X.apply(lambda seq: " ".join(list(seq.replace("<cls><cls>", " <cls> <cls> "))))
    sequences = X.values

    logger.info("Loading ProtT5 model for protein sequence embedding...")

    try:
        from transformers import T5EncoderModel, T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=cache_dir, do_lower_case=False)
        model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50", cache_dir=cache_dir)
        logger.info("Successfully loaded ProtT5 XL encoder model")
        model_type = "encoder"
    except Exception as e:
        logger.warning("ProtT5 XL encoder failed, trying base model: %s", str(e))
        try:
            tokenizer = T5Tokenizer.from_pretrained(
                "Rostlab/prot_t5_base_uniref50", cache_dir=cache_dir, do_lower_case=False
            )
            model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_base_uniref50", cache_dir=cache_dir)
            logger.info("Successfully loaded ProtT5 base encoder model")
            model_type = "encoder"
        except Exception as e2:
            logger.error("Failed to load ProtT5 models: %s", str(e2))

            # Check if it's a SentencePiece issue
            if "SentencePiece" in str(e2):
                raise RuntimeError(
                    "ProtT5 requires the SentencePiece library. Please install it with:\n"
                    "pip install sentencepiece\n"
                    "Then restart your environment and try again.\n"
                    "If you want to use ESM2 instead, please use the 'esm2' command."
                ) from e2
            else:
                raise RuntimeError(
                    f"Failed to load ProtT5 model: {str(e2)}\n"
                    "Please check your internet connection and try again.\n"
                    "If you want to use ESM2 instead, please use the 'esm2' command."
                ) from e2
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("ProtT5 model loaded. Size: %s M", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)

    # ProtT5 always has 1024 dimensions
    dim = 1024
    embeddings = torch.empty((n_seqs, dim))
    logger.info("Initialized embeddings tensor with ProtT5 dimension: %s", dim)

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("ProtT5 Batch %s/%s.", i, n_batches)

        x = torch.tensor(
            [
                tokenizer.encode(
                    seq, padding="max_length", truncation=True, max_length=max_length, return_special_tokens_mask=True
                )
                for seq in batch
            ]
        ).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)

        with torch.no_grad():
            if model_type == "encoder":
                outputs = model(input_ids=x, attention_mask=attention_mask)
                outputs = outputs.last_hidden_state
            else:
                outputs = model(x, attention_mask=attention_mask, output_hidden_states=True)
                outputs = outputs.hidden_states[-1]
            outputs = list(outputs.detach())

        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1, :].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("ProtT5 embedding took %s seconds", round(end_time - start_time, 2))

    save_embedding(dat, embeddings, output_file_path, out_format, cell_id_col)
    logger.info("Saved ProtT5 embedding at %s", output_file_path)


@app.command()
def tcr_bert(
    input_file_path: Annotated[
        str, typer.Argument(..., help="The path to the input AIRR format file containing TCR sequences.")
    ],
    chain: Annotated[
        str,
        typer.Argument(..., help="TCR chain type: 'A' (alpha), 'B' (beta), or 'AB' (alpha-beta concatenated pairs)"),
    ],
    output_file_path: Annotated[
        str,
        typer.Argument(
            ...,
            help="Output path for embeddings. Supports .pt (PyTorch), .csv, or .tsv formats.",
        ),
    ],
    cache_dir: Annotated[
        str,
        typer.Option(help="Directory to cache TCR-BERT model weights (default: system cache)."),
    ] = None,
    sequence_col: Annotated[str, typer.Option(help="Column name containing amino acid sequences.")] = "sequence_vdj_aa",
    cell_id_col: Annotated[str, typer.Option(help="Column name containing single-cell barcodes.")] = "cell_id",
    batch_size: Annotated[int, typer.Option(help="Batch size for processing (adjust based on GPU memory).")] = 32,
):
    """
    Embeds T-Cell Receptor (TCR) sequences using the TCR-BERT model.

    Example usage:\n
        amulety tcr-bert tcr_data.tsv AB tcr_embeddings.pt\n\n

    Note:\n
    Pretrained on 88,403 human TRA/TRB sequences from VDJdb and PIRD.
    Non-fine-tuned version focused on human TCR data only. The maximum length of the sequences to be embedded is 64.
    It prints the size of the model used for embedding, the batch number during the embedding process,
    and the time taken for the embedding. The embeddings are saved at the location specified by `output_file_path`.
    """
    out_format = check_output_file_type(output_file_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # map TCR chain notation to BCR notation for compatibility
    bcr_chain = {"A": "L", "B": "H", "AB": "HL"}.get(chain, chain)
    dat = process_airr(
        input_file_path, bcr_chain, sequence_col=sequence_col, cell_id_col=cell_id_col, receptor_type="TCR"
    )
    max_length = 64
    n_dat = dat.shape[0]

    dat = dat.dropna(subset=[sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:, sequence_col]
    X = X.apply(lambda a: a[:max_length])

    # TCR-BERT expects standard amino acid sequences without special tokens
    X = X.apply(lambda seq: seq.replace("<cls><cls>", " "))
    sequences = X.values

    logger.info("Loading TCR-BERT model for TCR embedding...")

    try:
        from transformers import BertModel, BertTokenizer

        # non-fine-tuned TCR-BERT model pre-trained on human data only
        model_name = "wukevin/tcr-bert"

        tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=cache_dir, do_lower_case=False)
        model = BertModel.from_pretrained(model_name, cache_dir=cache_dir)
        logger.info("Successfully loaded TCR-BERT model")

    except Exception as e:
        logger.warning("TCR-BERT model not available, using BERT-base as fallback: %s", str(e))
        try:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
            model = BertModel.from_pretrained("bert-base-uncased", cache_dir=cache_dir)
            logger.info("Using BERT-base as fallback for TCR-BERT")
        except Exception as e2:
            logger.error("Failed to load any BERT model: %s", str(e2))
            raise RuntimeError("Could not load TCR-BERT or fallback model") from e2

    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("TCR-BERT model loaded. Size: %s M", round(model_size / 1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    dim = 768
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("TCR-BERT Batch %s/%s.", i, n_batches)

        x = torch.tensor(
            [
                tokenizer.encode(
                    seq, padding="max_length", truncation=True, max_length=max_length, return_special_tokens_mask=True
                )
                for seq in batch
            ]
        ).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)

        with torch.no_grad():
            outputs = model(input_ids=x, attention_mask=attention_mask)
            outputs = outputs.last_hidden_state
            outputs = list(outputs.detach())

        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1, :].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("TCR-BERT embedding took %s seconds", round(end_time - start_time, 2))

    save_embedding(dat, embeddings, output_file_path, out_format, cell_id_col)
    logger.info("Saved TCR-BERT embedding at %s", output_file_path)


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
