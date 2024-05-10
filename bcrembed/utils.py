"""Main module."""
import os
import subprocess
import logging
from typing import Iterable
import torch
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def batch_loader(data: Iterable, batch_size: int):
    """
    This function generates batches from the provided data.

    Parameters:
    data (Iterable): The data to be batched.
    batch_size (int): The size of each batch.

    Yields:
    tuple: A tuple containing the start index, end index, and the batch of data.
    """
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]

def insert_space_every_other_except_cls(input_string: str):
    """
    This function inserts a space after every character in the input string, except for the '[CLS]' token.

    Parameters:
    input_string (str): The input string where spaces are to be inserted.

    Returns:
    str: The modified string with spaces inserted.
    """
    parts = input_string.split('[CLS]')
    modified_parts = [''.join([char + ' ' for char in part]).strip() for part in parts]
    result = ' [CLS] '.join(modified_parts)
    return result

def save_embedding(dat, embedding, outpath):
    """
    Saves the embedding data to a specified file path in the desired format.

    Args:
        dat (DataFrame): The original DataFrame containing index columns and possibly other data.
        embedding (Tensor): The embedding data to be saved.
        outpath (str): The file path where the embedding data will be saved.
        The output suffix should be one of the following:
        - 'pt': PyTorch binary format
        - 'tsv': Tab-separated values format
        - 'csv': Comma-separated values format

    Raises:
        ValueError: If the output format is not supported.

    Note:
        Index columns from the original DataFrame 'dat' will be included in the saved output.

    Example:
        save_embedding(dat, embeddings, "embedding.tsv")
    """
    out_format = os.path.splitext(outpath)[-1][1:]
    allowed_outputs = ["tsv", "csv", "pt"]
    if out_format not in allowed_outputs:
        raise ValueError(f"Output suffix must be one of {allowed_outputs}")

    allowed_index_cols = ["sequence_id", "cell_id"]
    index_cols = [col for col in dat.columns if col in allowed_index_cols]
    if out_format == 'pt':
        torch.save(embedding, outpath)
    elif out_format in ['tsv', 'csv']:
        embedding_df = pd.DataFrame(embedding.numpy())
        result_df = pd.concat([dat.loc[:,index_cols].reset_index(drop=True), embedding_df], axis=1)
        sep = '\t' if out_format == 'tsv' else ','
        result_df.to_csv(outpath, sep=sep, index=False)

def process_airr(inpath: str, chain: str, sequence_col: str = 'sequence_vdj_aa'):
    """
    Processes AIRR-seq data from the input file path and returns a pandas DataFrame containing the sequence to embed.

    Parameters:
        inpath (str): The file path to the input data.
        chain (str): The input chain, which can be one of ["H", "L", "HL"].
        sequence_col (str): The name of the column containing the amino acid sequences to embed.

    Returns:
        pandas.DataFrame: Dataframe with formatted sequences.

    Raises:
        ValueError: If chain is not one of ["H", "L", "HL"].
    """
    allowed_sequence_input = ["H", "L", "HL"]
    if chain not in allowed_sequence_input:
        raise ValueError(f"Input x must be one of {allowed_sequence_input}")
    
    data = pd.read_table(inpath)
    data.loc[:,'chain'] = data.loc[:,'locus'].apply(lambda x: 'H' if x == 'IGH' else 'L')

    if not 'cell_id' in data.columns:
        data_type = 'bulk-only'
    elif data['cell_id'].notna().all():
        data_type = 'single-cell-only'
    else:
        data_type = 'mixed'

    if data_type == 'bulk-only':
        logger.info("No cell_id column detected. Processsing as bulk data.")
        if chain == 'HL':
            raise ValueError('chain = "HL" invalid for bulk mode.')
        else:
            colnames = ['sequence_id', sequence_col]
            data = data.loc[data.chain == chain, colnames]

    elif data_type == 'single-cell-only':
        logger.info("Processing single-cell BCR data...")
        if chain == "HL":
            logging.info("Concatenating heavy and light chain per cell...")
            data = concatenate_HL(data, sequence_col)
        else:
            colnames = ['cell_id', sequence_col]
            data = data.loc[data.chain == chain, colnames]

    elif data_type == 'mixed':
        logger.info("Missing values in cell_id column. Processing as mixed bulk and single-cell BCR data...")
        if chain == "HL":
            logger.info("Concatenating heavy and light chain per cell...")
            data = data.loc[data.cell_id.notna(),]
            data = concatenate_HL(data, sequence_col)
        else:
            colnames = ['sequence_id', 'cell_id', sequence_col]
            data = data.loc[data.chain == chain, colnames]

    return data

def concatenate_HL(data: pd.DataFrame, sequence_col: str):
    """
    Concatenates heavy and light chain per cell and returns a pandas DataFrame.

    Parameters:
        data (pandas.DataFrame): Input data containing information about heavy and light chains.
        sequence_col (str): The name of the column containing the amino acid sequences to embed.

    Returns:
        pandas.DataFrame: Dataframe with concatenated heavy and light chains per cell.
    """
    colnames = ['cell_id', 'locus', 'consensus_count', sequence_col]
    missing_cols = [col for col in colnames if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Column(s) {missing_cols} is/are not present in the input data.")
    # if tie in maximum consensus_count, return the first occurrence
    data = data.loc[data.groupby(['cell_id', 'chain'])['consensus_count'].idxmax()]
    data = data.pivot(index='cell_id', columns='chain', values=sequence_col)
    data = data.reset_index(level = 'cell_id')
    n_cells = data.shape[0]
    data = data.dropna(axis = 0)
    n_dropped = n_cells - data.shape[0]
    if n_dropped > 0:
        logging.info("Dropping %s cells with missing heavy or light chain...", n_dropped)
    data.loc[:,sequence_col] = data.H + '<cls><cls>' + data.L
    return data

def translate_igblast(inpath: str, outdir: str, reference_dir: str):
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

# test case:
# dat = pivot_airr("/gpfs/gibbs/pi/kleinstein/embeddings/example_data/single_cell/MG-1__clone-pass_translated.tsv")
