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

..typer:: amulety.amulety.app:embed
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text


Chain Type Requirements
=======================

Different models have specific input chain requirements based on how they were trained in the original publications. AMULETY supports the following chain types:

- **H**: Heavy chains (BCR) or Beta/Delta chains (TCR) - individual chain embedding
- **L**: Light chains (BCR) or Alpha/Gamma chains (TCR) - individual chain embedding
- **HL**: Paired chains - concatenated Heavy-Light (BCR) or Beta-Alpha/Delta-Gamma (TCR) sequences
- **LH**: Reverse paired chains - concatenated Light-Heavy (BCR) or Alpha-Beta/Gamma-Delta (TCR) sequences
- **H+L**: Both chains separately - processes H and L chains individually without pairing


Custom Light Chain Selection
=============================

When using paired chains (``--chain HL``), AMULETY automatically selects the best light chain when multiple light chains exist for the same cell. By default, it uses the ``duplicate_count`` column, but you can specify a custom numeric column using the ``--duplicate-col`` parameter.  The column must contain numeric values (integers or floats), and AMULETY selects the chain with the highest value.



Protein embeddings
==================

These models support both B-Cell Receptor (BCR) and T-Cell Receptor (TCR) sequences with automatic receptor type detection and validation.

ProtT5
------

Protein language model pre-trained on UniRef50 sequences (H, L, H+L chains). Supports both BCR and TCR data with unified chain mapping (H=Heavy/Beta, L=Light/Alpha). This is a general protein language model without paired chain mechanisms, so HL/LH paired chains will be processed but accuracy may be reduced as the model cannot understand chain pairing structure.

Reference:
Elnaggar A, Heinzinger M, Dallago C, et al. ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing. bioRxiv. 2020. `doi:10.1101/2020.07.12.199554 <https://doi.org/10.1101/2020.07.12.199554>`_

ESM2
----

General protein language model pre-trained on 216 million UniRef50 protein sequences (H, L, H+L chains). Supports both BCR and TCR sequences. This is a general protein language model without paired chain mechanisms, so HL/LH paired chains will be processed but accuracy may be reduced as the model cannot understand chain pairing structure.

Reference:
Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379: 1123–1130. `doi:10.1126/science.ade2574 <https://doi.org/10.1126/science.ade2574>`_

BCR-specific embeddings
=======================

AntiBERTa2
----------

RoFormer model pre-trained on 1.54 billion unpaired and 2.9 million paired human antibody sequences (H, L, H+L chains). This model was trained on individual Heavy/Light chains separately and cannot understand paired sequences.

Reference:
Leem J, Mitchell LS, Farmery JHR, Barton J, Galson JD. Deciphering the language of antibodies using self-supervised learning. Patterns. 2022;3: 100513. `doi:10.1016/j.patter.2022.100513 <https://doi.org/10.1016/j.patter.2022.100513>`_

AntiBERTy
----------

Lightweight BERT model pre-trained on 588 million Observed Antibody Space (OAS) heavy and light antibody sequences from multiple species (H, L, H+L chains). This model was trained on individual Heavy/Light chains separately and cannot understand paired sequences.

Reference:
Ruffolo JA, Gray JJ, Sulam J. Deciphering antibody affinity maturation with language models and weakly supervised learning. arXiv. 2021; 2112.07782. `doi:10.48550/arXiv.2112.07782 <https://doi.org/10.48550/arXiv.2112.07782>`_

AbLang
------

Antibody language model with separate models for heavy and light chains (H, L, H+L chains). Trained on antibody sequences in the OAS database, demonstrating power in restoring missing residues. This model was trained on individual Heavy/Light chains separately and cannot understand paired sequences.

Reference:
Olsen TH, Boyles F, Deane CM. AbLang: an antibody language model for completing antibody sequences. Bioinformatics Advances. 2022;2: vbac046. `doi:10.1093/bioadv/vbac046 <https://doi.org/10.1093/bioadv/vbac046>`_

BALM-paired
-----------

RoBERTa-based large model pre-trained on 1.34 million concatenated heavy and light chain human antibody sequences (HL, LH chains). This model requires paired input and cannot process individual chains. It was trained on concatenated Heavy-Light sequences and understands chain pairing structure.

Reference:
Burbach SM, Briney B. Improving antibody language models with native pairing. Patterns. 2024;5. `doi:10.1016/j.patter.2024.100967 <https://doi.org/10.1016/j.patter.2024.100967>`_

TCR-specific embeddings
=======================


**Important Note**: Most TCR embedding models listed below are primarily trained on alpha/beta TCRs (TRA/TRB sequences), unless otherwise specified. While AMULETY's unified interface accepts gamma/delta TCRs (TRG/TRD), results may be less reliable for these sequences due to limited training data.

TCR-BERT
--------

BERT model pre-trained on 88,403 human TCR alpha and beta sequences (TRA/TRB) from VDJdb and PIRD databases (H, L, HL, LH, H+L chains). Specialized for alpha/beta T-Cell Receptor analysis. This model supports all chain formats and understands chain relationships.

Reference:
Lu T, Zhang Z, Zhu J, et al. Deep learning-based prediction of the T cell receptor–antigen binding specificity. bioRxiv. 2021. `doi:10.1101/2021.11.18.469186 <https://www.biorxiv.org/content/10.1101/2021.11.18.469186v1>`_


TCRT5
-----

T5-based model pre-trained on masked span reconstruction using ~14M CDR3 β sequences from TCRdb and ~780k peptide-pseudosequence pairs from IEDB (H chains only). This model is specialized for TCR beta chains only and supports H chain embedding exclusively. Here we included only the model trained on the TCR sequence only, not the TCR-peptide pairs.

Reference:
Deng K, Guan R, Liu Z, et al. TCRT5: T-cell receptor sequence modeling with T5. bioRxiv. 2024. `doi:10.1101/2024.11.11.623124 <https://doi.org/10.1101/2024.11.11.623124>`_

Immune-specific embeddings
==========================

Immune2Vec
----------

Protein language model trained on immune receptor sequences (H, L, H+L chains). This is a general protein language model without paired chain mechanisms, so HL/LH paired chains will be processed but accuracy may be reduced as the model cannot understand chain pairing structure.

Reference:
Beshnova D, Ye J, Onabolu O, et al. De novo prediction of cancer-associated T cell receptors for noninvasive cancer detection. Science Translational Medicine. 2020;12: eaaz3738. `doi:10.1126/scitranslmed.aaz3738 <https://doi.org/10.1126/scitranslmed.aaz3738>`_


