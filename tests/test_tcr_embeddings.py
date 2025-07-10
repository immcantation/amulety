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
        """Test TCR-BERT (alpha/gamma chains - L chains for TCR) - now supported."""
        embed(self.test_airr_tcr_path, "L", "tcr-bert", "tcr_L_test.pt", batch_size=2)
        assert os.path.exists("tcr_L_test.pt")
        embeddings = torch.load("tcr_L_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcr_L_test.pt")

    def test_tcr_bert_H_embedding(self):
        """Test TCR-BERT (beta/delta chains - H chains for TCR) - now supported."""
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

    def test_tcr_bert_LH_embedding(self):
        """Test TCR-BERT (beta-alpha/delta-gamma pairs - LH pairs for TCR with warning)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embed(self.test_airr_tcr_path, "LH", "tcr-bert", "tcr_LH_test.pt", batch_size=2)
            assert os.path.exists("tcr_LH_test.pt")
            embeddings = torch.load("tcr_LH_test.pt")
            assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
            assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
            os.remove("tcr_LH_test.pt")
            # Check that LH warning was issued
            assert len(w) > 0
            assert any("LH (Light-Heavy) chain order detected" in str(warning.message) for warning in w)

    def test_deep_tcr_L_embedding(self):
        """Test DeepTCR (alpha/gamma chains - L chains for TCR)."""
        embed(self.test_airr_tcr_path, "L", "deep-tcr", "deep_tcr_L_test.pt", batch_size=2)
        assert os.path.exists("deep_tcr_L_test.pt")
        embeddings = torch.load("deep_tcr_L_test.pt")
        assert embeddings.shape[1] == 64  # DeepTCR embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("deep_tcr_L_test.pt")

    def test_deep_tcr_H_embedding(self):
        """Test DeepTCR (beta/delta chains - H chains for TCR)."""
        embed(self.test_airr_tcr_path, "H", "deep-tcr", "deep_tcr_H_test.pt", batch_size=2)
        assert os.path.exists("deep_tcr_H_test.pt")
        embeddings = torch.load("deep_tcr_H_test.pt")
        assert embeddings.shape[1] == 64  # DeepTCR embedding dimension
        assert embeddings.shape[0] == 3  # 3 beta chains in test data
        os.remove("deep_tcr_H_test.pt")

    def test_deep_tcr_H_plus_L_embedding(self):
        """Test DeepTCR (both alpha and beta chains separately - H+L for TCR)."""
        embed(self.test_airr_tcr_path, "H+L", "deep-tcr", "deep_tcr_H_plus_L_test.pt", batch_size=2)
        assert os.path.exists("deep_tcr_H_plus_L_test.pt")
        embeddings = torch.load("deep_tcr_H_plus_L_test.pt")
        assert embeddings.shape[1] == 64  # DeepTCR embedding dimension
        assert embeddings.shape[0] == 6  # 3 alpha + 3 beta chains in test data
        os.remove("deep_tcr_H_plus_L_test.pt")

    def test_deep_tcr_HL_chain_validation(self):
        """Test that DeepTCR rejects HL chains (individual chain model)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_tcr_path, "HL", "deep-tcr", "should_fail.pt", batch_size=2)
        self.assertIn("supports individual chains only", str(context.exception))
        self.assertIn("--chain H", str(context.exception))
        self.assertIn("--chain L", str(context.exception))

    def test_tcremp_L_embedding(self):
        """Test TCREMP (alpha/gamma chains - L chains for TCR)."""
        embed(self.test_airr_tcr_path, "L", "tcremp", "tcremp_L_test.pt", batch_size=2)
        assert os.path.exists("tcremp_L_test.pt")
        embeddings = torch.load("tcremp_L_test.pt")
        assert embeddings.shape[1] == 256  # TCREMP temporary embedding dimension (actual TBD)
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcremp_L_test.pt")

    def test_tcremp_H_embedding(self):
        """Test TCREMP (beta/delta chains - H chains for TCR)."""
        embed(self.test_airr_tcr_path, "H", "tcremp", "tcremp_H_test.pt", batch_size=2)
        assert os.path.exists("tcremp_H_test.pt")
        embeddings = torch.load("tcremp_H_test.pt")
        assert embeddings.shape[1] == 256  # TCREMP temporary embedding dimension (actual TBD)
        assert embeddings.shape[0] == 3  # 3 beta chains in test data
        os.remove("tcremp_H_test.pt")

    def test_tcremp_H_plus_L_embedding(self):
        """Test TCREMP (both alpha and beta chains separately - H+L for TCR)."""
        embed(self.test_airr_tcr_path, "H+L", "tcremp", "tcremp_H_plus_L_test.pt", batch_size=2)
        assert os.path.exists("tcremp_H_plus_L_test.pt")
        embeddings = torch.load("tcremp_H_plus_L_test.pt")
        assert embeddings.shape[1] == 256  # TCREMP temporary embedding dimension (actual TBD)
        assert embeddings.shape[0] == 6  # 3 alpha + 3 beta chains in test data
        os.remove("tcremp_H_plus_L_test.pt")

    def test_tcremp_HL_embedding(self):
        """Test TCREMP (alpha-beta/gamma-delta pairs - HL pairs for TCR) - now supported."""
        embed(self.test_airr_tcr_path, "HL", "tcremp", "tcremp_HL_test.pt", batch_size=2)
        assert os.path.exists("tcremp_HL_test.pt")
        embeddings = torch.load("tcremp_HL_test.pt")
        assert embeddings.shape[1] == 256  # TCREMP temporary embedding dimension (actual TBD)
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcremp_HL_test.pt")

    def test_tcremp_LH_embedding(self):
        """Test TCREMP (beta-alpha/delta-gamma pairs - LH pairs for TCR with warning) - now supported."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embed(self.test_airr_tcr_path, "LH", "tcremp", "tcremp_LH_test.pt", batch_size=2)
            assert os.path.exists("tcremp_LH_test.pt")
            embeddings = torch.load("tcremp_LH_test.pt")
            assert embeddings.shape[1] == 256  # TCREMP temporary embedding dimension (actual TBD)
            assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
            os.remove("tcremp_LH_test.pt")
            # Check that LH warning was issued
            assert len(w) > 0
            assert any("LH (Light-Heavy) chain order detected" in str(warning.message) for warning in w)
