"""Console script for bcrembed"""
import typer
from rich.console import Console
from antiberty import AntiBERTyRunner
import torch
from transformers import (
    RoFormerModel,
    RoFormerForMaskedLM,
    RoFormerTokenizer,
    pipeline,
    RoFormerForSequenceClassification,
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import numpy as np
import pandas as pd
import time
import math

from bcrembed import __version__
from bcrembed.utils import (
    pivot_airr,
    insert_space_every_other_except_cls,
    batch_loader
)

app = typer.Typer()
stderr = Console(stderr=True)
stdout = Console()

@app.command()
def antiberty(inpath: str, colname: str, outpath: str):
    """
    AntiBERTy
    Usage:
    bcrembed antiberty tests/AIRR_rearrangement_translated.tsv HL ~/palmer_scratch/test.pt
    """
    
    dat = pivot_airr(inpath) # H, L, HL
    stdout.print(f"Embedding {dat.shape[0]} sequences using antiberty...")
    max_length = 512-2
    X = dat.loc[:,colname]
    X = X.dropna()
    X = X.apply(lambda a: a[:max_length])
    X = X.str.replace('<cls><cls>', '[CLS][CLS]')
    X = X.apply(insert_space_every_other_except_cls)
    sequences = X.str.replace('  ', ' ')
    # detect 
    antiberty = AntiBERTyRunner()
    start_time = time.time()
    batch_size = 500
    n_seqs = len(sequences)
    dim = 512

    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        print(f'Batch {i}/{n_batches}\n')
        x = antiberty.embed(batch)
        x = [a.mean(axis = 0) for a in x]
        embeddings[start:end] = torch.stack(x)
        i += 1

    end_time = time.time()
    stdout.print(f"Took {end_time - start_time} seconds")

    torch.save(embeddings, outpath)
    stdout.print(f"Saved embedding at {outpath}")

@app.command()
def antiberta2(inpath: str, colname: str, outpath: str):
    """Console script for bcrembedder."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dat = pivot_airr(inpath)
    X = dat.loc[:,colname]
    max_length = 256
    X = X.apply(lambda a: a[:max_length])
    X = X.str.replace('<cls><cls>', '[CLS][CLS]')
    X = X.apply(insert_space_every_other_except_cls)
    X = X.str.replace('  ', ' ')
    sequences = X.values

    tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
    model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2")
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    stdout.print(f"Model loaded. Size: {model_size/1e6:.2f}M")

    start_time = time.time()
    batch_size = 128
    n_seqs = len(sequences)
    dim = 1024
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

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
    stdout.print(f"Took {end_time - start_time} seconds")

    torch.save(embeddings, outpath)
    stdout.print(f"Saved embedding at {outpath}")

@app.command()
def esm2(inpath: str, colname: str, outpath: str):
    """Console script for bcrembedder."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dat = pivot_airr(inpath)
    X = dat.loc[:,colname]
    max_length = 512
    X = X.apply(lambda a: a[:max_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    stdout.print(f"Model size: {model_size/1e6:.2f}M")

    start_time = time.time()
    batch_size = 50
    n_seqs = len(sequences)
    dim = 1280
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

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
    stdout.print(f"Took {end_time - start_time} seconds")

    torch.save(embeddings, outpath)
    stdout.print(f"Saved embedding at {outpath}")

@app.command()
def custom_model(modelpath: str, inpath: str, colname: str, outpath: str):
    """Console script for bcrembedder."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dat = pd.read_table(inpath)
    X = dat.loc[:,colname]
    max_length = 512
    X = X.apply(lambda a: a[:max_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForMaskedLM.from_pretrained(modelpath)
    model = model.to(device)
    model_size = sum(p.numel() for p in model.parameters())
    stdout.print(f"Model size: {model_size/1e6:.2f}M")

    start_time = time.time()
    batch_size = 50
    n_seqs = len(sequences)
    dim = 1280
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

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
    stdout.print(f"Took {end_time - start_time} seconds")

    torch.save(embeddings, outpath)
    stdout.print(f"Saved embedding at {outpath}")

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
