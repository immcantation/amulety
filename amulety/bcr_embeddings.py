"""
BCR embedding functions using various models.
Please order alphabetically by function name.
"""
# ruff: noqa: N806

import logging
import math
import os
import subprocess
import time
from typing import Optional

import pandas as pd
import torch

from amulety.protein_embeddings import custommodel
from amulety.utils import batch_loader, insert_space_every_other_except_cls

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def antiberty(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 50,
):
    """
    Embeds sequences using the AntiBERTy model.\n
    The maximum length of the sequences to be embedded is 510.
    """
    from antiberty import AntiBERTyRunner

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

    from transformers import RoFormerForMaskedLM, RoFormerTokenizer

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


def balm_paired(
    sequences: pd.Series,
    cache_dir: str = "/tmp/amulety",
    batch_size: int = 50,
):
    """
    Embeds sequences using the BALM-paired model. The maximum length of the sequences to be embedded is 1024. The embedding dimension is 1024.
    """

    os.makedirs(cache_dir, exist_ok=True)

    model_name = "BALM-paired_LC-coherence_90-5-5-split_122222"
    model_path = os.path.join(cache_dir, model_name)
    embedding_dimension = 1024
    max_seq_length = 510

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

    embeddings = custommodel(
        sequences=sequences,
        model_path=model_path,
        embedding_dimension=embedding_dimension,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        cache_dir=cache_dir,
    )
    return embeddings
