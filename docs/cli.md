# Command line reference

## Translate sequences to amino acids using IgBLAST

```{click}
:prog: amulety
:module: amulety.amulety
:func: app
:command: translate-igblast
:width: 90
:show-nested:
:make-sections:
:preferred: text
```

## Embed sequences with different models

```{click}
:prog: amulety
:module: amulety.amulety
:func: app
:command: embed
:width: 90
:show-nested:
:make-sections:
:preferred: text
```

## Options

### Chain type requirements

Different models have specific input chain requirements based on how they were trained in the original publications. AMULETY supports the following chain types:

- **H**: Heavy chains (BCR) or Beta/Delta chains (TCR) - individual chain embedding
- **L**: Light chains (BCR) or Alpha/Gamma chains (TCR) - individual chain embedding
- **HL**: Paired chains - concatenated Heavy-Light (BCR) or Beta-Alpha/Delta-Gamma (TCR) sequences
- **LH**: Reverse paired chains - concatenated Light-Heavy (BCR) or Alpha-Beta/Gamma-Delta (TCR) sequences
- **H+L**: Both chains separately - processes H and L chains individually without pairing

### Custom light chain selection

When using paired chains (`--chain HL`), AMULETY automatically selects the best light chain when multiple light chains exist for the same cell. By default, it uses the `duplicate_count` column, but you can specify a custom numeric column using the `--duplicate-col` option. The column must contain numeric values (integers or floats), and AMULETY selects the chain with the highest value.

```bash
# Default behavior: use duplicate_count
amulety embed --chain HL --model antiberta2 --output-file-path embeddings.pt input.tsv

# Custom selection: use a quality score column
amulety embed --chain HL --model antiberta2 --duplicate-col quality_score --output-file-path embeddings.pt input.tsv

# Custom selection: use UMI count
amulety embed --chain HL --model antiberta2 --duplicate-col umi_count --output-file-path embeddings.pt input.tsv
```
