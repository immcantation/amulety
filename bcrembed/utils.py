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

# test case:
# dat = pivot_airr("/gpfs/gibbs/pi/kleinstein/embeddings/example_data/single_cell/MG-1__clone-pass_translated.tsv")
