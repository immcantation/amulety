"""Main module."""
import pandas as pd
import os
import subprocess

def batch_loader(data, batch_size):
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]

def insert_space_every_other_except_cls(input_string):
    parts = input_string.split('[CLS]')
    modified_parts = [''.join([char + ' ' for char in part]).strip() for part in parts]
    result = ' [CLS] '.join(modified_parts)
    return result

def pivot_airr(inpath):
    data = pd.read_table(inpath)
    colnames = ['cell_id', 'locus', 'consensus_count', 'sequence_vdj_aa']
    data = data.loc[:, colnames]
    data.loc[:,'chain'] = data.loc[:,'locus'].apply(lambda x: 'H' if x == 'IGH' else 'L')
    data = data.loc[data.groupby(['cell_id', 'chain'])['consensus_count'].idxmax()]
    data = data.pivot(index='cell_id', columns='chain', values='sequence_vdj_aa')
    data.loc[:,'HL'] = data.H + '<cls><cls>' + data.L
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
    command_igblastn = ["igblastn", "-germline_db_V", f"{reference_dir}/imgt_human_ig_v",
           "-germline_db_D", f"{reference_dir}/imgt_human_ig_d",
           "-germline_db_J", f"{reference_dir}/imgt_human_ig_j",
           "-query", out_fasta,
           "-organism", "human",
           "-auxiliary_data", f"{reference_dir}/human_gl.aux",
           "-show_translation",
           "-outfmt", "19",
           "-out", out_igblast]
    subprocess.run(command_igblastn)

    # Read IgBlast output
    igblast_transl = pd.read_csv(out_igblast, sep="\t", usecols=["sequence_id","sequence_aa", "sequence_alignment_aa"])

    # Remove IMGT gaps
    sequence_vdj_aa = [ sa.replace("-","") for sa in igblast_transl["sequence_alignment_aa"]]
    igblast_transl["sequence_vdj_aa"] = sequence_vdj_aa

    # Merge and save the translated data with original data
    data_transl = pd.merge(data, igblast_transl, on="sequence_id", how="left")
    data_transl.to_csv(out_translated, sep="\t", index=False)

# test case:
# dat = pivot_airr("/gpfs/gibbs/pi/kleinstein/embeddings/example_data/single_cell/MG-1__clone-pass_translated.tsv")
