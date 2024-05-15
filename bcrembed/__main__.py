"""Console script for bcrembed"""
import os
import logging
import time
import math
import typer
from typing_extensions import Annotated
from rich.console import Console
from antiberty import AntiBERTyRunner
import torch
from transformers import (
    RoFormerForMaskedLM,
    RoFormerTokenizer,
    AutoTokenizer,
    AutoModelForMaskedLM,
)


from bcrembed import __version__
from bcrembed.utils import (
    process_airr,
    insert_space_every_other_except_cls,
    batch_loader,
    save_embedding
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = typer.Typer()
stderr = Console(stderr=True)
stdout = Console()

@app.command()
def antiberty(inpath: Annotated[str, typer.Argument(..., help= 'The path to the input data file. The data file should be in AIRR format.')],
              chain: Annotated[str, typer.Argument(..., help= 'Input sequences (H for heavy chain, L for light chain, HL for heavy and light concatenated)')],
              outpath: Annotated[str, typer.Argument(..., help= 'The path where the generated embeddings will be saved.')],
              sequence_col: Annotated[str, typer.Option(help= 'The name of the column containing the amino acid sequences to embed.')] = "sequence_vdj_aa",
              batch_size: Annotated[int, typer.Option(help= 'The batch size of sequences to embed.')] = 500):
    """
    Embeds sequences using the AntiBERTy model.\n

    Note:\n
        This function prints the number of sequences being embedded, the batch number during the
        embedding process, the time taken for the embedding, and the location where the embeddings
        are saved.\n

    Example usage:\n
        bcrembed antiberty tests/AIRR_rearrangement_translated_single-cell.tsv HL out.pt

    """

    dat = process_airr(inpath, chain, sequence_col=sequence_col)
    logger.info("Embedding %s sequences using antiberty...", dat.shape[0])
    max_length = 512-2
    n_dat = dat.shape[0]

    dat = dat.dropna(subset = [sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:,sequence_col]
    X = X.apply(lambda a: a[:max_length])
    X = X.str.replace('<cls><cls>', '[CLS][CLS]')
    X = X.apply(insert_space_every_other_except_cls)
    sequences = X.str.replace('  ', ' ')

    antiberty_runner = AntiBERTyRunner()
    model_size = sum(p.numel() for p in antiberty_runner.model.parameters())
    logger.info("AntiBERTy loaded. Size: %s M", round(model_size/1e6, 2))
    start_time = time.time()
    n_seqs = len(sequences)
    dim = 512

    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info('Batch %s/%s', i, n_batches)
        x = antiberty_runner.embed(batch)
        x = [a.mean(axis = 0) for a in x]
        embeddings[start:end] = torch.stack(x)
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))

    save_embedding(dat, embeddings, outpath)
    logger.info("Saved embedding at %s", outpath)

@app.command()
def antiberta2(inpath: Annotated[str, typer.Argument(..., help= 'The path to the input data file. The data file should be in AIRR format.')],
               chain: Annotated[str, typer.Argument(..., help= 'Input sequences (H for heavy chain, L for light chain, HL for heavy and light concatenated)')],
               outpath: Annotated[str, typer.Argument(..., help= 'The path where the generated embeddings will be saved.')],
               sequence_col: Annotated[str, typer.Option(help= 'The name of the column containing the amino acid sequences to embed.')] = "sequence_vdj_aa",
               batch_size: Annotated[int, typer.Option(help= 'The batch size of sequences to embed.')] = 128):
    """
    Embeds sequences using the antiBERTa2 RoFormer model.\n

    Note:\n
        This function uses the ESM2 model for embedding. The maximum length of the sequences to be embedded is 512.
        It prints the size of the model used for embedding, the batch number during the embedding process,
        and the time taken for the embedding. The embeddings are saved at the location specified by `outpath`.

    Example Usage:\n
        bcrembed antiberta2 tests/AIRR_rearrangement_translated_single-cell.tsv HL out.pt\n
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dat = process_airr(inpath, chain, sequence_col=sequence_col)
    max_length = 256
    n_dat = dat.shape[0]

    dat = dat.dropna(subset = [sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:, sequence_col]
    X = X.apply(lambda a: a[:max_length])
    X = X.str.replace('<cls><cls>', '[CLS][CLS]')
    X = X.apply(insert_space_every_other_except_cls)
    X = X.str.replace('  ', ' ')
    sequences = X.values

    tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
    model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2")
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("AntiBERTa2 loaded. Size: %s M", model_size/1e6)

    start_time = time.time()
    n_seqs = len(sequences)
    dim = 1024
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info('Batch %s/%s.', i, n_batches)
        x = torch.tensor([
        tokenizer.encode(seq,
                         padding="max_length",
                         truncation=True,
                         max_length=max_length,
                         return_special_tokens_mask=True) for seq in batch]).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            outputs = model(x, attention_mask = attention_mask,
                           output_hidden_states = True)
            outputs = outputs.hidden_states[-1]
            outputs = list(outputs.detach())

        # aggregate across the residuals, ignore the padded bases
        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1,:].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))

    save_embedding(dat, embeddings, outpath)
    logger.info("Saved embedding at %s", outpath)

@app.command()
def esm2(inpath: Annotated[str, typer.Argument(..., help= 'The path to the input data file. The data file should be in AIRR format.')],
         chain: Annotated[str, typer.Argument(..., help= 'Input sequences (H for heavy chain, L for light chain, HL for heavy and light concatenated)')],
         outpath: Annotated[str, typer.Argument(..., help= 'The path where the generated embeddings will be saved.')],
         sequence_col: Annotated[str, typer.Option(help= 'The name of the column containing the amino acid sequences to embed.')] = "sequence_vdj_aa",
         batch_size: Annotated[int, typer.Option(help= 'The batch size of sequences to embed.')] = 50):
    """
    Embeds sequences using the ESM2 model.

    Example usage:
        bcrembed esm2 tests/AIRR_rearrangement_translated_single-cell.tsv HL out.pt

    Note:
        This function uses the ESM2 model for embedding. The maximum length of the sequences to be embedded is 512.
        It prints the size of the model used for embedding, the batch number during the embedding process,
        and the time taken for the embedding. The embeddings are saved at the location specified by `outpath`.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dat = process_airr(inpath, chain, sequence_col=sequence_col)
    max_length = 512
    n_dat = dat.shape[0]

    dat = dat.dropna(subset = [sequence_col])
    n_dropped = n_dat - dat.shape[0]
    if n_dropped > 0:
        logger.info("Removed %s rows with missing values in %s", n_dropped, sequence_col)

    X = dat.loc[:, sequence_col]
    X = X.apply(lambda a: a[:max_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("ESM2 650M model size: %s M", round(model_size/1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    dim = 1280
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info('Batch %s/%s.', i, n_batches)
        x = torch.tensor([
        tokenizer.encode(seq,
                         padding="max_length",
                         truncation=True,
                         max_length=max_length,
                         return_special_tokens_mask=True) for seq in batch]).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            outputs = model(x, attention_mask = attention_mask,
                           output_hidden_states = True)
            outputs = outputs.hidden_states[-1]
            outputs = list(outputs.detach())

        # aggregate across the residuals, ignore the padded bases
        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1,:].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))

    save_embedding(dat, embeddings, outpath)
    logger.info("Saved embedding at %s", outpath)

@app.command()
def custommodel(modelpath: Annotated[str, typer.Argument(..., help= 'The path to the pretrained model.')],
                inpath: Annotated[str, typer.Argument(..., help= 'The path to the input data file. The data file should be in AIRR format.')],
                chain: Annotated[str, typer.Argument(..., help= 'Input sequences (H for heavy chain, L for light chain, HL for heavy and light concatenated)')],
                outpath: Annotated[str, typer.Argument(..., help= 'The path where the generated embeddings will be saved.')],
                embedding_dimension: Annotated[int, typer.Option(help= 'The dimension of the embedding layer.')] = 100,
                max_length: Annotated[int, typer.Option(help= 'The maximum length that the model can take.')] = 512,
                batch_size: Annotated[int, typer.Option(help= 'The batch size of sequences to embed.')] = 50,
                sequence_col: Annotated[str, typer.Option(help= 'The name of the column containing the amino acid sequences to embed.')] = "sequence_vdj_aa"):
    """
    This function generates embeddings for a given dataset using a pretrained model. The function first checks if a CUDA device is available for PyTorch to use. It then loads the data from the input file and preprocesses it.
    The sequences are tokenized and fed into the pretrained model to generate embeddings. The embeddings are then saved to the specified output path.\n

    Note:\n
        This function uses the transformers library's AutoTokenizer and AutoModelForMaskedLM classes to handle the tokenization and model loading.\n

    Example Usage:\n
        bcrembed custom_model <custom_model_path> tests/AIRR_rearrangement_translated_single-cell.tsv HL out.pt\n

    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dat = process_airr(inpath, chain, sequence_col=sequence_col)
    X = dat.loc[:, sequence_col]
    X = X.apply(lambda a: a[:max_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForMaskedLM.from_pretrained(modelpath)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    logger.info("Model size: %sM", round(model_size/1e6, 2))

    start_time = time.time()
    n_seqs = len(sequences)
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, embedding_dimension))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        print(f'Batch {i}/{n_batches}\n')
        x = torch.tensor([
        tokenizer.encode(seq,
                         padding="max_length",
                         truncation=True,
                         max_length=max_length,
                         return_special_tokens_mask=True) for seq in batch]).to(device)
        attention_mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            outputs = model(x, attention_mask = attention_mask,
                           output_hidden_states = True)
            outputs = outputs.hidden_states[-1]
            outputs = list(outputs.detach())

        # aggregate across the residuals, ignore the padded bases
        for j, a in enumerate(attention_mask):
            outputs[j] = outputs[j][a == 1,:].mean(0)

        embeddings[start:end] = torch.stack(outputs)
        del x
        del attention_mask
        del outputs
        i += 1

    end_time = time.time()
    logger.info("Took %s seconds", round(end_time - start_time, 2))

    save_embedding(dat, embeddings, outpath)
    logger.info("Saved embedding at %s", outpath)

@app.command()
def translate_igblast(inpath: Annotated[str, typer.Argument(..., help= 'The path to the input data file. The data file should be in AIRR format.')],
                      outdir: Annotated[str, typer.Argument(..., help= 'The directory where the generated embeddings will be saved.')],
                      reference_dir: Annotated[str, typer.Argument(..., help= 'The directory to the igblast references.')]):
    """
    Translates nucleotide sequences to amino acid sequences using IgBlast.

    This function takes a AIRR file containing nucleotide sequences
    and translates them into amino acid sequences using IgBlast, a tool for analyzing
    immunoglobulin and T cell receptor sequences. It performs the following steps:

    1. Reads the input TSV file containing nucleotide sequences.
    2. Writes the nucleotide sequences into a FASTA file, required as input for IgBlast.
    3. Runs IgBlast on the FASTA file to perform sequence alignment and translation.
    4. Reads the IgBlast output, which includes the translated amino acid sequences.
    5. Removes gaps introduced by IgBlast from the sequence alignment.
    6. Saves the translated data into a new TSV file in the specified output directory.

    Args:
        inpath (str): Path to the input TSV file containing nucleotide sequences.
        outdir (str): Directory to save the translated output files.
        reference_dir (str): Directory with reference for igblast
    """
    data = pd.read_csv(inpath, sep="\t")
    out_fasta = os.path.join(outdir, os.path.splitext(os.path.basename(inpath))[0]+".fasta")
    out_igblast = os.path.join(outdir, os.path.splitext(os.path.basename(inpath))[0]+"_igblast.tsv")
    out_translated = os.path.join(outdir, os.path.splitext(os.path.basename(inpath))[0]+"_translated.tsv")

    # Write out FASTA file
    with open(out_fasta, "w") as f:
        for _, row in data.iterrows():
            f.write(">" + row["sequence_id"] + "\n")
            f.write(row["sequence"] + "\n")

    # Run IgBlast on FASTA
    command_igblastn = ["igblastn", "-germline_db_V", f"{reference_dir}/database/imgt_human_ig_v",
           "-germline_db_D", f"{reference_dir}/database/imgt_human_ig_d",
           "-germline_db_J", f"{reference_dir}/database/imgt_human_ig_j",
           "-query", out_fasta,
           "-organism", "human",
           "-auxiliary_data", f"{reference_dir}/optional_file/human_gl.aux",
           "-show_translation",
           "-outfmt", "19",
           "-out", out_igblast]
    pipes = subprocess.Popen(command_igblastn, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = pipes.communicate()

    if pipes.returncode != 0:
        raise Exception(f"IgBlast failed with error code {pipes.returncode}. {stderr.decode('utf-8')}")

    # Read IgBlast output
    igblast_transl = pd.read_csv(out_igblast, sep="\t", usecols=["sequence_id","sequence_aa", "sequence_alignment_aa"])

    # Remove IMGT gaps
    sequence_vdj_aa = [ sa.replace("-","") for sa in igblast_transl["sequence_alignment_aa"]]
    igblast_transl["sequence_vdj_aa"] = sequence_vdj_aa

    # Merge and save the translated data with original data
    data_transl = pd.merge(data, igblast_transl, on="sequence_id", how="left")
    data_transl.to_csv(out_translated, sep="\t", index=False)

    # Clean up
    os.remove(out_fasta)
    os.remove(out_igblast)

def main():
    asci_art = "BCR EMBED\n"
    asci_art = r"""
 ____   ____ ____                _              _       __   __
| __ ) / ___|  _ \ ___ _ __ ___ | |__   ___  __| |      \ \ / /
|  _ \| |   | |_) / _ \ '_ ` _ \| '_ \ / _ \/ _` |       \ V /
| |_) | |___|  _ <  __/ | | | | | |_) |  __/ (_| |        | |
|____/ \____|_| \_\___|_| |_| |_|_.__/ \___|\__,_|        |_|
"""
    stderr.print(asci_art)
    stderr.print(f"BCR EMBED version {__version__}\n")

    app()

if __name__ == "__main__":
    main()
