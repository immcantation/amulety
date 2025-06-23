"""
Protein sequence embedding functions using various models.
Please order alphabetically by function name.
"""
# ruff: noqa: N806

import logging
import math
import time
from typing import Optional

import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from amulety.utils import batch_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    from transformers import AutoModelForMaskedLM, AutoTokenizer

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


def prott5(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Embeds BCR or TCR sequences using the ProtT5-XL protein language model (Rostlab/prot_t5_xl_uniref50). The maximum sequence length to embed is 1024 amino acids, and the generated embeddings have a dimension of 1024.
    """
    max_seq_length = 1024  # ProtT5 can't handle longer sequences
    dim = 1024

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # bcr_chain = {"A": "L", "B": "H", "AB": "HL"}.get(chain, chain)

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])

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

    embeddings = torch.empty((n_seqs, dim))
    logger.info("Initialized embeddings tensor with ProtT5 dimension: %s", dim)

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("ProtT5 Batch %s/%s.", i, n_batches)

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

    return embeddings
