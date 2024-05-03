"""Main module."""
import pandas as pd

def batch_loader(data, batch_size):
    """
    This function generates batches from the provided data.

    Parameters:
    data (iterable): The data to be batched.
    batch_size (int): The size of each batch.

    Yields:
    tuple: A tuple containing the start index, end index, and the batch of data.
    """
    num_samples = len(data)
    for i in range(0, num_samples, batch_size):
        end_idx = min(i + batch_size, num_samples)
        yield i, end_idx, data[i:end_idx]

def insert_space_every_other_except_cls(input_string):
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

def pivot_airr(inpath):
    """
    This function reads a table from a file, selects specific columns, and pivots the table based on 'cell_id' and 'chain'.
    It also creates a new column 'HL' which is a combination of 'H' and 'L' columns separated by '<cls><cls>'.

    Parameters:
    inpath (str): The path to the input data file. The data file should be in table format.

    Returns:
    DataFrame: The pivoted DataFrame.
    """
    data = pd.read_table(inpath)
    colnames = ['cell_id', 'locus', 'consensus_count', 'sequence_vdj_aa']
    data = data.loc[:, colnames]
    data.loc[:,'chain'] = data.loc[:,'locus'].apply(lambda x: 'H' if x == 'IGH' else 'L')
    data = data.loc[data.groupby(['cell_id', 'chain'])['consensus_count'].idxmax()]
    data = data.pivot(index='cell_id', columns='chain', values='sequence_vdj_aa')
    data.loc[:,'HL'] = data.H + '<cls><cls>' + data.L
    return data

