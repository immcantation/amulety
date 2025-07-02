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

    def test_tcr_bert_L_embedding(self):
        """Test TCR-BERT (alpha/gamma chains - L chains for TCR)."""
        embed(self.test_airr_tcr_path, "L", "tcr-bert", "tcr_L_test.pt", batch_size=2)
        assert os.path.exists("tcr_L_test.pt")
        embeddings = torch.load("tcr_L_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcr_L_test.pt")

    def test_tcr_bert_H_embedding(self):
        """Test TCR-BERT (beta/delta chains - H chains for TCR)."""
        embed(self.test_airr_tcr_path, "H", "tcr-bert", "tcr_H_test.pt", batch_size=2)
        assert os.path.exists("tcr_H_test.pt")
        embeddings = torch.load("tcr_H_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 beta chains in test data
        os.remove("tcr_H_test.pt")

    def test_tcr_bert_HL_embedding(self):
        """Test TCR-BERT (alpha-beta/gamma-delta pairs - HL pairs for TCR)."""
        embed(self.test_airr_tcr_path, "HL", "tcr-bert", "tcr_HL_test.pt", batch_size=2)
        assert os.path.exists("tcr_HL_test.pt")
        embeddings = torch.load("tcr_HL_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcr_HL_test.pt")

    def test_deep_tcr_L_embedding(self):
        """Test DeepTCR (alpha/gamma chains - L chains for TCR)."""
        embed(self.test_airr_tcr_path, "L", "deep-tcr", "deep_tcr_L_test.pt", batch_size=2)
        assert os.path.exists("deep_tcr_L_test.pt")
        embeddings = torch.load("deep_tcr_L_test.pt")
        assert embeddings.shape[1] == 768  # DeepTCR placeholder (BERT) embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("deep_tcr_L_test.pt")

    def test_deep_tcr_H_embedding(self):
        """Test DeepTCR (beta/delta chains - H chains for TCR)."""
        embed(self.test_airr_tcr_path, "H", "deep-tcr", "deep_tcr_H_test.pt", batch_size=2)
        assert os.path.exists("deep_tcr_H_test.pt")
        embeddings = torch.load("deep_tcr_H_test.pt")
        assert embeddings.shape[1] == 768  # DeepTCR placeholder (BERT) embedding dimension
        assert embeddings.shape[0] == 3  # 3 beta chains in test data
        os.remove("deep_tcr_H_test.pt")

    def test_deep_tcr_HL_embedding(self):
        """Test DeepTCR (alpha-beta/gamma-delta pairs - HL pairs for TCR)."""
        embed(self.test_airr_tcr_path, "HL", "deep-tcr", "deep_tcr_HL_test.pt", batch_size=2)
        assert os.path.exists("deep_tcr_HL_test.pt")
        embeddings = torch.load("deep_tcr_HL_test.pt")
        assert embeddings.shape[1] == 768  # DeepTCR placeholder (BERT) embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("deep_tcr_HL_test.pt")

    def test_tcremp_L_embedding(self):
        """Test TCREMP (alpha/gamma chains - L chains for TCR)."""
        embed(self.test_airr_tcr_path, "L", "tcremp", "tcremp_L_test.pt", batch_size=2)
        assert os.path.exists("tcremp_L_test.pt")
        embeddings = torch.load("tcremp_L_test.pt")
        assert embeddings.shape[1] == 512  # TCREMP embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcremp_L_test.pt")

    def test_tcremp_H_embedding(self):
        """Test TCREMP (beta/delta chains - H chains for TCR)."""
        embed(self.test_airr_tcr_path, "H", "tcremp", "tcremp_H_test.pt", batch_size=2)
        assert os.path.exists("tcremp_H_test.pt")
        embeddings = torch.load("tcremp_H_test.pt")
        assert embeddings.shape[1] == 512  # TCREMP embedding dimension
        assert embeddings.shape[0] == 3  # 3 beta chains in test data
        os.remove("tcremp_H_test.pt")

    def test_tcremp_HL_embedding(self):
        """Test TCREMP (alpha-beta/gamma-delta pairs - HL pairs for TCR)."""
        embed(self.test_airr_tcr_path, "HL", "tcremp", "tcremp_HL_test.pt", batch_size=2)
        assert os.path.exists("tcremp_HL_test.pt")
        embeddings = torch.load("tcremp_HL_test.pt")
        assert embeddings.shape[1] == 512  # TCREMP embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcremp_HL_test.pt")

    def test_trex_L_embedding(self):
        """Test Trex (alpha/gamma chains - L chains for TCR)."""
        embed(self.test_airr_tcr_path, "L", "trex", "trex_L_test.pt", batch_size=2)
        assert os.path.exists("trex_L_test.pt")
        embeddings = torch.load("trex_L_test.pt")
        assert embeddings.shape[1] == 768  # Trex embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("trex_L_test.pt")

    def test_trex_H_embedding(self):
        """Test Trex (beta/delta chains - H chains for TCR)."""
        embed(self.test_airr_tcr_path, "H", "trex", "trex_H_test.pt", batch_size=2)
        assert os.path.exists("trex_H_test.pt")
        embeddings = torch.load("trex_H_test.pt")
        assert embeddings.shape[1] == 768  # Trex embedding dimension
        assert embeddings.shape[0] == 3  # 3 beta chains in test data
        os.remove("trex_H_test.pt")

    def test_trex_HL_embedding(self):
        """Test Trex (alpha-beta/gamma-delta pairs - HL pairs for TCR)."""
        embed(self.test_airr_tcr_path, "HL", "trex", "trex_HL_test.pt", batch_size=2)
        assert os.path.exists("trex_HL_test.pt")
        embeddings = torch.load("trex_HL_test.pt")
        assert embeddings.shape[1] == 768  # Trex embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("trex_HL_test.pt")
