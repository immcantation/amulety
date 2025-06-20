"""
TCR embedding functions using various models.
Please order alphabetically by function name.
"""
# ruff: noqa: N806

import logging
import math
import time
from typing import Optional

import pandas as pd
import torch

from amulety.utils import batch_loader

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def tcr_bert(
    sequences: pd.Series,
    cache_dir: Optional[str] = None,
    batch_size: int = 32,
):
    """
    Embeds T-Cell Receptor (TCR) sequences using the TCR-BERT model.

    Note:\n
    Pretrained on 88,403 human TRA/TRB sequences from VDJdb and PIRD.
    Non-fine-tuned version focused on human TCR data only. The maximum length of the sequences to be embedded is 64.
    It prints the size of the model used for embedding, the batch number during the embedding process,
    and the time taken for the embedding. The embeddings are saved at the location specified by `output_file_path`.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_seq_length = 64
    dim = 768

    X = sequences
    X = X.apply(lambda a: a[:max_seq_length])

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
    n_batches = math.ceil(n_seqs / batch_size)
    embeddings = torch.empty((n_seqs, dim))

    i = 1
    for start, end, batch in batch_loader(sequences, batch_size):
        logger.info("TCR-BERT Batch %s/%s.", i, n_batches)

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

    return embeddings
