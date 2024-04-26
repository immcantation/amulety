"""Console script for bcrembedder."""
import bcrembed

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

app = typer.Typer()
console = Console()

# TODO: detect CPU or GPU

@app.command()
def antiberty(inpath: str, colname: str, outpath: str):
    """
    AntiBERTy
    Usage:
    bcrembed antiberty /gpfs/gibbs/pi/kleinstein/embeddings/example_data/single_cell/MG-1__clone-pass_translated.tsv HL ~/palmer_scratch/test.pt
    """

    dat = bcrembedder.pivot_airr(inpath) # H, L, HL
    console.print(f"Embedding {dat.shape[0]} sequences using antiberty...")
    max_length = 512-2
    X = dat.loc[:,colname]
    X = X.dropna()
    X = X.apply(lambda a: a[:max_length])
    X = X.str.replace('<cls><cls>', '[CLS][CLS]')
    X = X.apply(bcrembedder.insert_space_every_other_except_cls)
    sequences = X.str.replace('  ', ' ')
    antiberty = AntiBERTyRunner()
    start_time = time.time()
    batch_size = 500
    n_seqs = len(sequences)
    dim = 512

    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in bcrembedder.batch_loader(sequences, batch_size):
        print(f'Batch {i}/{n_batches}\n')
        x = antiberty.embed(batch)
        x = [a.mean(axis = 0) for a in x]
        embeddings[start:end] = torch.stack(x)
        i += 1

    end_time = time.time()
    console.print(f"Took {end_time - start_time} seconds")

    torch.save(embeddings, outpath)
    console.print(f"Saved embedding at {outpath}")

@app.command()
def antiberta2(inpath: str, colname: str, outpath: str):
    """Console script for bcrembedder."""
    dat = pd.read_table(inpath)
    X = dat.loc[:,colname]
    max_length = 256
    X = X.apply(lambda a: a[:max_length])
    X = X.str.replace('<cls><cls>', '[CLS][CLS]')
    X = X.apply(bcrembedder.insert_space_every_other_except_cls)
    X = X.str.replace('  ', ' ')
    sequences = X.values

    tokenizer = RoFormerTokenizer.from_pretrained("alchemab/antiberta2")
    model = RoFormerForMaskedLM.from_pretrained("alchemab/antiberta2")
    model = model.to('cuda')
    model_size = sum(p.numel() for p in model.parameters())
    console.print(f"Model loaded. Size: {model_size/1e6:.2f}M")

    start_time = time.time()
    batch_size = 128
    n_seqs = len(sequences)
    dim = 1024
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in bcrembedder.batch_loader(sequences, batch_size):
        print(f'Batch {i}/{n_batches}\n')
        x = torch.tensor([
        tokenizer.encode(seq,
                         padding="max_length",
                         truncation=True,
                         max_length=max_length,
                         return_special_tokens_mask=True) for seq in batch]).to('cuda')
        attention_mask = (x != tokenizer.pad_token_id).float().to('cuda')
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
    console.print(f"Took {end_time - start_time} seconds")

    torch.save(embeddings, outpath)
    console.print(f"Saved embedding at {outpath}")

@app.command()
def esm2(inpath: str, colname: str, outpath: str):
    """Console script for bcrembedder."""
    dat = pd.read_table(inpath)
    X = dat.loc[:,colname]
    max_length = 512
    X = X.apply(lambda a: a[:max_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D")
    model = model.to('cuda')
    model_size = sum(p.numel() for p in model.parameters())
    console.print(f"Model size: {model_size/1e6:.2f}M")

    start_time = time.time()
    batch_size = 50
    n_seqs = len(sequences)
    dim = 1280
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in bcrembedder.batch_loader(sequences, batch_size):
        print(f'Batch {i}/{n_batches}\n')
        x = torch.tensor([
        tokenizer.encode(seq,
                         padding="max_length",
                         truncation=True,
                         max_length=max_length,
                         return_special_tokens_mask=True) for seq in batch]).to('cuda')
        attention_mask = (x != tokenizer.pad_token_id).float().to('cuda')
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
    console.print(f"Took {end_time - start_time} seconds")

    torch.save(embeddings, outpath)
    console.print(f"Saved embedding at {outpath}")

@app.command()
def custom_model(modelpath: str, inpath: str, colname: str, outpath: str):
    """Console script for bcrembedder."""
    dat = pd.read_table(inpath)
    X = dat.loc[:,colname]
    max_length = 512
    X = X.apply(lambda a: a[:max_length])
    sequences = X.values

    tokenizer = AutoTokenizer.from_pretrained(modelpath)
    model = AutoModelForMaskedLM.from_pretrained(modelpath)
    model = model.to('cuda')
    model_size = sum(p.numel() for p in model.parameters())
    console.print(f"Model size: {model_size/1e6:.2f}M")

    start_time = time.time()
    batch_size = 50
    n_seqs = len(sequences)
    dim = 1280
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in bcrembedder.batch_loader(sequences, batch_size):
        print(f'Batch {i}/{n_batches}\n')
        x = torch.tensor([
        tokenizer.encode(seq,
                         padding="max_length",
                         truncation=True,
                         max_length=max_length,
                         return_special_tokens_mask=True) for seq in batch]).to('cuda')
        attention_mask = (x != tokenizer.pad_token_id).float().to('cuda')
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
    console.print(f"Took {end_time - start_time} seconds")

    torch.save(embeddings, outpath)
    console.print(f"Saved embedding at {outpath}")

def main():
    app()

if __name__ == "__main__":
    main()
