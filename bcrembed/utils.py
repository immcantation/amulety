"""Main module."""
import pandas as pd

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

def translate_igblast(inpath):
    command_igblastn = ["igblastn", "-germline_db_V", "../databases/igblast_base/database/imgt_human_ig_v",
           "-germline_db_D", "../databases/igblast_base/database/imgt_human_ig_d",
           "-germline_db_J", "../databases/igblast_base/database/imgt_human_ig_j",
           "-query", "fasta",
           "-organism", "human",
           "-auxiliary_data", "../databases/igblast_base/database/human_gl.aux",
           "-show_translation",
           "-outfmt", "19",
           "-out", "tsv"]

    for file in rep_files:
        rep = pd.read_csv(file, sep="\t")
        filename = os.path.basename(file)
        filename_without_extension = os.path.splitext(filename)[0]
        print(filename_without_extension)

        with open(outdir + filename_without_extension +".fasta", "w") as f:
            for i in range(len(rep)):
                f.write(">" + rep["sequence_id"].iloc[i] + "\n")
                f.write(rep["sequence"].iloc[i] + "\n")

    fastas = [os.path.splitext(os.path.basename(file))[0]+".fasta" for file in rep_files]
    tsvs = [os.path.splitext(os.path.basename(file))[0]+"_igblast.tsv" for file in rep_files]

    for (fasta,tsv) in zip(fastas,tsvs):
        print(fasta,tsv)
        command_igblastn[-1] = outdir + tsv
        command_igblastn[-10] = outdir + fasta
        print(command_igblastn)
        subprocess.run(command_igblastn)

    for (file,transl) in zip(rep_files,tsvs):
        rep = pd.read_csv(file, sep="\t")
        #igblast_transl = pd.read_csv(outdir+transl, sep="\t", usecols=["sequence_id","sequence_aa"])
        igblast_transl = pd.read_csv(outdir+transl, sep="\t", usecols=["sequence_id","sequence_aa", "sequence_alignment_aa"])
        sequence_vdj_aa = [ sa.replace("-","") for sa in igblast_transl["sequence_alignment_aa"]]
        igblast_transl["sequence_vdj_aa"] = sequence_vdj_aa
        rep_transl = pd.merge(rep, igblast_transl, on="sequence_id", how="left")
        rep_transl.to_csv(outdir+os.path.splitext(os.path.basename(file))[0]+"_translated.tsv", sep="\t", index=False)
        rep_transl.head()





# test case:
# dat = pivot_airr("/gpfs/gibbs/pi/kleinstein/embeddings/example_data/single_cell/MG-1__clone-pass_translated.tsv")
