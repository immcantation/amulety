"""Main module."""
import pandas as pd
import logging

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

def process_airr(inpath, sequence_input):
    """
    Processes AIRR-seq data from the input file path and returns a pandas DataFrame containing the sequence to embed.
    
    Parameters:
        inpath (str): The file path to the input data.
        sequence_input (str): The type of sequence input, which can be one of ["H", "L", "HL"].
        
    Returns:
        pandas.DataFrame: Processed AIRR-seq data based on the sequence_input.
        
    Raises:
        ValueError: If sequence_input is not one of ["H", "L", "HL"].
    """
    allowed_sequence_input = ["H", "L", "HL"]
    if sequence_input not in allowed_sequence_input:
        raise ValueError("Input x must be one of {}".format(allowed_sequence_input))
        
    data = pd.read_table(inpath)
    data.loc[:,'chain'] = data.loc[:,'locus'].apply(lambda x: 'H' if x == 'IGH' else 'L')
    
    if not 'cell_id' in data.columns:
        data_type = 'bulk-only'
    elif data['cell_id'].notna().all():
        data_type = 'single-cell-only'
    else:
        data_type = 'mixed'

    if data_type == 'bulk-only':
        logging.info("No cell_id column detected. Processsing as bulk data.")
        if sequence_input == 'HL':
            raise ValueError('sequence_input invalid for bulk mode.')
        else: 
            colnames = ['sequence_id', 'sequence_vdj_aa']
            data = data.loc[data.chain == sequence_input, colnames]      
            
    elif data_type == 'single-cell-only':
        logging.info("Processing single-cell BCR data...")
        if sequence_input == "HL":
            logging.info("Concatenating heavy and light chain per cell...")
            data = concatenate_HL(data)
        else:
            colnames = ['cell_id', 'sequence_vdj_aa']
            data = data.loc[data.chain == sequence_input, colnames]
        
    else:
        logging.info("Missing values in cell_id column. Processing as mixed bulk and single-cell BCR data...")
        if sequence_input == "HL":
            logging.info("Concatenating heavy and light chain per cell...")
            data = data.loc[data.cell_id.notna(),]
            data = concatenate_HL(data)
        else:
            colnames = ['sequence_id', 'cell_id', 'sequence_vdj_aa']
            data = data.loc[data.chain == sequence_input, colnames]
        
    return data

def concatenate_HL(data):
    """
    Concatenates heavy and light chain per cell and returns a pandas DataFrame.
    
    Parameters:
        data (pandas.DataFrame): Input data containing information about heavy and light chains.
        
    Returns:
        pandas.DataFrame: Dataframe with concatenated heavy and light chains per cell.
    """
    colnames = ['cell_id', 'locus', 'consensus_count', 'sequence_vdj_aa']
    # if tie in maximum consensus_count, return the first occurrence
    data = data.loc[data.groupby(['cell_id', 'chain'])['consensus_count'].idxmax()] 
    data = data.pivot(index='cell_id', columns='chain', values='sequence_vdj_aa')
    logging.info("Dropping cells with missing heavy or light chain...")
    data = data.dropna(axis = 0)
    data.loc[:,'sequence_vdj_aa'] = data.H + '<cls><cls>' + data.L
    return data
    