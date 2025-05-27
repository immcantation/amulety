# TCR Support in Amulety - Usage Guide

## Overview

This guide explains how to use the newly added TCR (T-Cell Receptor) functionality in Amulety. The TCR support has been implemented as a separate module alongside the existing BCR functionality to avoid conflicts and make debugging easier.

## New TCR Commands

### 1. `tcr-prott5` - TCR Embedding with ProtT5

Uses the ProtT5 model from Hugging Face to embed TCR sequences.

```bash
amulety tcr-prott5 input_file.tsv CHAIN output_file.pt [OPTIONS]
```

**Parameters:**
- `input_file.tsv`: AIRR format file containing TCR sequences
- `CHAIN`: Chain type - "A" (alpha), "B" (beta), or "AB" (alpha-beta concatenated)
- `output_file.pt`: Output file (.pt, .csv, or .tsv)

**Options:**
- `--cache-dir`: Directory to cache model weights
- `--sequence-col`: Column name for sequences (default: "sequence_vdj_aa")
- `--cell-id-col`: Column name for cell IDs (default: "cell_id")
- `--batch-size`: Batch size for processing (default: 32)

**Example:**
```bash
amulety tcr-prott5 tcr_data.tsv AB tcr_embeddings.pt --batch-size 16
```

### 2. `tcr-esm2` - TCR Embedding with ESM2

Uses the ESM2 model to embed TCR sequences.

```bash
amulety tcr-esm2 input_file.tsv CHAIN output_file.pt [OPTIONS]
```

**Parameters and options are the same as tcr-prott5**

**Example:**
```bash
amulety tcr-esm2 tcr_data.tsv A tcr_alpha_embeddings.csv
```

## TCR Data Format

Your input AIRR file should contain TCR sequences with the following columns:

### Required Columns:
- `sequence_id`: Unique identifier for each sequence
- `locus`: TCR locus (TRA, TRB, TRG, TRD)
- `sequence_vdj_aa`: Amino acid sequence of the TCR
- `v_call`: V gene call (used to determine locus if missing)

### Optional Columns:
- `cell_id`: Single-cell barcode (required for AB mode)
- `duplicate_count`: Count for selecting best chain per cell

### Example TCR Data:
```
sequence_id	cell_id	locus	v_call	sequence_vdj_aa	duplicate_count
tcr_001	cell_1	TRA	TRAV1*01	QVQLVQSGAEVKKPGASVKVSCKASG...	10
tcr_002	cell_1	TRB	TRBV1*01	EVQLVESGGGLVQPGGSLRLSCAASG...	15
tcr_003	cell_2	TRA	TRAV2*01	QSALTQPASVSGSPGQSITISCTGTS...	8
tcr_004	cell_2	TRB	TRBV2*01	DIQMTQSPSSLSASVGDRVTITCRAS...	12
```

## Chain Types

### Alpha Chain (A)
- Processes only TRA and TRG loci
- Suitable for alpha chain-specific analysis

### Beta Chain (B)
- Processes only TRB and TRD loci
- Suitable for beta chain-specific analysis

### Alpha-Beta Concatenated (AB)
- Concatenates alpha and beta chains per cell
- Format: `BETA_SEQUENCE<cls><cls>ALPHA_SEQUENCE`
- Requires single-cell data with cell_id column
- Automatically selects best chain per cell based on duplicate_count

## Key Differences from BCR Processing

### 1. Chain Mapping
- **BCR**: H (Heavy), L (Light), HL (Heavy-Light)
- **TCR**: A (Alpha), B (Beta), AB (Alpha-Beta)

### 2. Locus Recognition
- **BCR**: IGH, IGL, IGK
- **TCR**: TRA, TRB, TRG, TRD

### 3. Concatenation Order
- **BCR**: Heavy + Light (`H<cls><cls>L`)
- **TCR**: Beta + Alpha (`B<cls><cls>A`)

## Model Specifications

### ProtT5 (Recommended for TCR)
- **Model**: `Rostlab/prot_t5_xl_half_uniref50-enc`
- **Embedding Dimension**: 1024
- **Max Sequence Length**: 1024
- **Preprocessing**: Space-separated amino acids
- **Best for**: General protein sequences, including TCRs

### ESM2
- **Model**: `facebook/esm2_t33_650M_UR50D`
- **Embedding Dimension**: 1280
- **Max Sequence Length**: 512
- **Best for**: Evolutionary analysis

## Code Organization

### Files Modified:
1. **`amulety/utils.py`**: Added TCR-specific functions
   - `process_tcr_airr()`: TCR data processing
   - `concatenate_alphabeta()`: Alpha-beta chain concatenation

2. **`amulety/amulety.py`**: Added TCR commands
   - `tcr_prott5()`: ProtT5 embedding command
   - `tcr_esm2()`: ESM2 embedding command

### Clear Separation:
- All TCR code is clearly marked with comments
- TCR functions are separate from BCR functions
- No modification of existing BCR functionality

## Troubleshooting

### Common Issues:

1. **Missing locus column**: The code automatically generates it from v_call
2. **Missing cell_id for AB mode**: Use A or B mode for bulk data
3. **Memory issues**: Reduce batch_size parameter
4. **Model download**: Ensure internet connection for first-time model download

### Debug Mode:
The code includes extensive logging. Check console output for:
- Number of sequences processed
- Chain type detection
- Cell pairing information
- Batch processing progress

## Testing

Run the included test script to verify functionality:
```bash
python simple_tcr_test.py
```

This tests:
- TCR chain mapping logic
- Alpha-beta concatenation
- File I/O operations

## Future Enhancements

Potential additions:
- TCR-specific pre-trained models
- CDR3-specific embedding options
- TCR translation functionality
- Additional TCR-optimized models
