# BCRembed

BCRembed is a Python command line tool for generating embeddings for amino acid sequences using pretrained protein or antibody language models.

## Features

- Generate embeddings using the protein / antibody language model (antiBERTy, antiBERTa2, ESM2)
- Utilities for data preprocessing and batch loading.

## Installation

You can install BCRembed using pip:

```bash
pip install bcrembed
```

## Usage
Here are some examples of how to use BCRembed:

```bash
bcrembed antiberty <input_file> <column_name> <output_file>
bcrembed antiberta2 <input_file> <column_name> <output_file>
bcrembed esm2 <input_file> <column_name> <output_file>
```

This command will generate embeddings for the sequences in the specified column of the input file using the specified model. The embeddings will be saved to the specified output file.


## License
This project is licensed under the terms of the GPL v3 license. See the LICENSE file for details.