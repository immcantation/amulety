#!/usr/bin/env python

"""Tests for `amulety` package.
Tests can be run with the command:
python -m unittest test_amulety.py
"""

import os
import unittest

import pandas as pd
import torch

from amulety.amulety import embed


class TestAmulety(unittest.TestCase):
    """Function that runs at start of tests for common resources."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.test_airr_mixed = "AIRR_rearrangement_translated_mixed.tsv"
        self.test_airr_translation = "AIRR_rearrangement_single-cell_testtranslation.tsv"
        self.test_airr_tcr = "AIRR_rearrangement_tcr_test.tsv"
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_airr_mixed_path = os.path.join(self.this_dir, self.test_airr_mixed)
        self.test_airr_translation_path = os.path.join(self.this_dir, self.test_airr_translation)
        self.test_airr_tcr_path = os.path.join(self.this_dir, self.test_airr_tcr)
        self.test_airr_tcr_df = pd.read_table(self.test_airr_tcr_path, delimiter="\t", header=0)
        self.test_mixed = "AIRR_rearrangement_mixed_bcr_tcr_test.tsv"
        self.test_mixed_path = os.path.join(self.this_dir, self.test_mixed)
        self.test_mixed_df = pd.read_table(self.test_mixed_path, delimiter="\t", header=0)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_tcr_bert_A_embedding(self):
        """Test TCR-BERT (alpha chains)."""
        embed(self.test_airr_tcr_path, "A", "TCRBert", "tcr_A_test.pt", batch_size=2)
        assert os.path.exists("tcr_A_test.pt")
        embeddings = torch.load("tcr_A_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcr_A_test.pt")

    def test_tcr_bert_B_embedding(self):
        """Test TCR-BERT (beta chains)."""
        embed(self.test_airr_tcr_path, "B", "TCRBert", "tcr_B_test.pt", batch_size=2)
        assert os.path.exists("tcr_B_test.pt")
        embeddings = torch.load("tcr_B_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 beta chains in test data
        os.remove("tcr_B_test.pt")

    def test_tcr_bert_AB_embedding(self):
        """Test TCR-BERT (alpha-beta pairs)."""
        embed(self.test_airr_tcr_path, "AB", "TCRBert", "tcr_AB_test.pt", batch_size=2)
        assert os.path.exists("tcr_AB_test.pt")
        embeddings = torch.load("tcr_AB_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcr_AB_test.pt")
