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


Unified BCR and TCR embeddings
==============================

These models support both B-Cell Receptor (BCR) and T-Cell Receptor (TCR) sequences with automatic receptor type detection and validation.

ProtT5
------

Protein language model pre-trained on UniRef50 sequences. Supports both BCR and TCR data with unified chain mapping (BCR: H/L/HL, TCR: A/B/AB).

Reference:
Elnaggar A, Heinzinger M, Dallago C, et al. ProtTrans: Towards Cracking the Language of Life's Code Through Self-Supervised Deep Learning and High Performance Computing. bioRxiv. 2020. `doi:10.1101/2020.07.12.199554 <https://doi.org/10.1101/2020.07.12.199554>`_

.. typer:: amulety.amulety:app:prott5
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text

ESM2
----

General protein language model pre-trained on 216 million UniRef50 protein sequences. Supports both BCR and TCR sequences.

Reference:
Lin Z, Akin H, Rao R, Hie B, Zhu Z, Lu W, et al. Evolutionary-scale prediction of atomic-level protein structure with a language model. Science. 2023;379: 1123–1130. `doi:10.1126/science.ade2574 <https://doi.org/10.1126/science.ade2574>`_

.. typer:: amulety.amulety:app:esm2
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text


BCR-specific embeddings
=======================

AntiBERTa2
----------

RoFormer model pre-trained on 1.54 billion unpaired and 2.9 million paired human antibody sequences.

Reference:
Leem J, Mitchell LS, Farmery JHR, Barton J, Galson JD. Deciphering the language of antibodies using self-supervised learning. Patterns. 2022;3: 100513. `doi:10.1016/j.patter.2022.100513 <https://doi.org/10.1016/j.patter.2022.100513>`_

.. typer:: amulety.amulety:app:antiberta2
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text

AntiBERTy
----------

Lightweight BERT model pre-trained on 588 million Observed Antibody Space (OAS) heavy and light antibody sequences from multiple species.

Reference:
Ruffolo JA, Gray JJ, Sulam J. Deciphering antibody affinity maturation with language models and weakly supervised learning. arXiv. 2021; 2112.07782. `doi:10.48550/arXiv.2112.07782 <https://doi.org/10.48550/arXiv.2112.07782>`_

.. typer:: amulety.amulety:app:antiberty
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text

BALM-paired
-----------

RoBERTa-based large model pre-trained on 1.34 million concatenated heavy and light chain human antibody sequences. Specialized model for paired chain embeddings.

Reference:
Burbach SM, Briney B. Improving antibody language models with native pairing. Patterns. 2024;5. `doi:10.1016/j.patter.2024.100967 <https://doi.org/10.1016/j.patter.2024.100967>`_

.. typer:: amulety.amulety:app:balm-paired
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text


TCR-specific embeddings
=======================

TCR-BERT
--------

BERT model pre-trained on 88,403 human TCR alpha and beta sequences from VDJdb and PIRD databases. Specialized for T-Cell Receptor analysis.

Reference:
Lu T, Zhang Z, Zhu J, et al. Deep learning-based prediction of the T cell receptor–antigen binding specificity. Nature Machine Intelligence. 2021;3: 864–875. `doi:10.1038/s42256-021-00383-2 <https://doi.org/10.1038/s42256-021-00383-2>`_

.. typer:: amulety.amulety:app:tcr-bert
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text


Custom pre-trained model
========================

.. typer:: amulety.amulety:app:custommodel
    :width: 90
    :show-nested:
    :make-sections:
    :preferred: text


