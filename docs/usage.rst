=====
Usage
=====


Translate sequence to amino acids
=================================

.. typer:: amulety.amulety.app:translate-igblast
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text


Embed sequences with different models
=====================================

..typer:: amulety.amulety.app: embed
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text


Chain Type Requirements
=======================

Different models have specific chain input requirements based on their training data:

**Paired-only models (HL required):**
- ``balm-paired``: Trained on concatenated Heavy-Light sequences

**Flexible paired models (all chain types):**
- ``tcr-bert``: Trained on concatenated Alpha-Beta sequences, supports all chain formats
- ``tcremp``: Supports all chain types including paired chains (HL/LH) and individual chains

**Individual chain models (H or L only):**
- ``antiberty``, ``antiberta2``: Trained on individual Heavy/Light chains

**H chain only models:**
- ``tcrt5``: Trained exclusively on TCR beta chains, only supports H chains

**Flexible models (H, L, or HL):**
- ``esm2``, ``prott5``, ``immune2vec``, ``custom``: Support all chain formats

AMULETY will validate chain compatibility and provide clear error messages for mismatched combinations.

Custom Light Chain Selection
=============================

When using paired chains (``--chain HL``), AMULETY automatically selects the best light chain when multiple light chains exist for the same cell. By default, it uses the ``duplicate_count`` column, but you can specify a custom numeric column using the ``--selection-col`` parameter.

**Default behavior:**

.. code-block:: bash

  amulety embed --chain HL --model antiberta2 --output-file-path embeddings.pt input.tsv

**Custom selection column:**

.. code-block:: bash

  # Use a quality score column
  amulety embed --chain HL --model antiberta2 --selection-col quality_score --output-file-path embeddings.pt input.tsv

  # Use UMI count
  amulety embed --chain HL --model antiberta2 --selection-col umi_count --output-file-path embeddings.pt input.tsv

**Requirements:**

- The column must contain numeric values (integers or floats)
- AMULETY selects the chain with the highest value
- Custom columns must be added to your AIRR data file by the user

**Common custom columns:**

- ``quality_score``: Sequence quality metrics
- ``umi_count``: Unique molecular identifier counts
- ``expression_level``: Gene expression levels
- ``confidence_score``: Assembly confidence scores


Protein embeddings
==================

These models support both B-Cell Receptor (BCR) and T-Cell Receptor (TCR) sequences with automatic receptor type detection and validation.

ProtT5
------

Protein language model pre-trained on UniRef50 sequences. Supports both BCR and TCR data with unified chain mapping (H=Heavy/Beta, L=Light/Alpha, HL=Heavy-Light/Beta-Alpha pairs).

Reference:
Elnaggar A, Heinzinger M, Dallago C, et al. ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing. bioRxiv. 2020. `doi:10.1101/2020.07.12.199554 <https://doi.org/10.1101/2020.07.12.199554>`_

ESM2
----

General protein language model pre-trained on 216 million UniRef50 protein sequences. Supports both BCR and TCR sequences.

Reference:
Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379: 1123–1130. `doi:10.1126/science.ade2574 <https://doi.org/10.1126/science.ade2574>`_

BCR-specific embeddings
=======================

AntiBERTa2
----------

RoFormer model pre-trained on 1.54 billion unpaired and 2.9 million paired human antibody sequences.

Reference:
Leem J, Mitchell LS, Farmery JHR, Barton J, Galson JD. Deciphering the language of antibodies using self-supervised learning. Patterns. 2022;3: 100513. `doi:10.1016/j.patter.2022.100513 <https://doi.org/10.1016/j.patter.2022.100513>`_

AntiBERTy
----------

Lightweight BERT model pre-trained on 588 million Observed Antibody Space (OAS) heavy and light antibody sequences from multiple species.

Reference:
Ruffolo JA, Gray JJ, Sulam J. Deciphering antibody affinity maturation with language models and weakly supervised learning. arXiv. 2021; 2112.07782. `doi:10.48550/arXiv.2112.07782 <https://doi.org/10.48550/arXiv.2112.07782>`_

BALM-paired
-----------

RoBERTa-based large model pre-trained on 1.34 million concatenated heavy and light chain human antibody sequences. Specialized model for paired chain embeddings.

Reference:
Burbach SM, Briney B. Improving antibody language models with native pairing. Patterns. 2024;5. `doi:10.1016/j.patter.2024.100967 <https://doi.org/10.1016/j.patter.2024.100967>`_

TCR-specific embeddings
=======================

**Important Note**: Most TCR embedding models listed below are primarily trained on alpha/beta TCRs (TRA/TRB sequences). While AMULETY's unified interface accepts gamma/delta TCRs (TRG/TRD), results may be less reliable for these sequences due to limited training data.

TCR-BERT
--------

BERT model pre-trained on 88,403 human TCR alpha and beta sequences (TRA/TRB) from VDJdb and PIRD databases. Specialized for alpha/beta T-Cell Receptor analysis.

Reference:
Lu T, Zhang Z, Zhu J, et al. Deep learning-based prediction of the T cell receptor–antigen binding specificity. bioRxiv. 2021. `doi:10.1101/2021.11.18.469186 <https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1>`_



TCREMP
------

TCR-specific embedding method trained for T-cell receptor repertoire-based representation learning. Focuses on repertoire-level prediction tasks with specialized TCR sequence understanding.

Reference:
Zhang H, Zeng T, Zhao Y, et al. TCREMP: T-cell receptor repertoire-based embedding for immunotherapy response prediction. Journal of Molecular Biology. 2025;437: 168712. `doi:10.1016/j.jmb.2025.168712 <https://www.sciencedirect.com/science/article/pii/S0022283625002712>`_

TCRT5
-----

T5-based model pre-trained on masked span reconstruction using ~14M CDR3 β sequences from TCRdb and ~780k peptide-pseudosequence pairs from IEDB. This model is specialized for TCR beta chains only and supports H chain embedding exclusively.

Reference:
Deng K, Guan R, Liu Z, et al. TCRT5: T-cell receptor sequence modeling with T5. bioRxiv. 2024. `doi:10.1101/2024.11.11.623124 <https://doi.org/10.1101/2024.11.11.623124>`_


