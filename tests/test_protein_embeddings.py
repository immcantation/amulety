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

# Skip large model tests on GitHub Actions due to disk space limitations
SKIP_LARGE_MODELS = os.environ.get("GITHUB_ACTIONS") == "true"


class TestAmulety(unittest.TestCase):
    """Function that runs at start of tests for common resources."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.test_airr_sc = "AIRR_rearrangement_translated_single-cell.tsv"
        self.test_airr_bulk = "AIRR_rearrangement_translated_bulk.tsv"
        self.test_airr_mixed = "AIRR_rearrangement_translated_mixed.tsv"
        self.test_airr_translation = "AIRR_rearrangement_single-cell_testtranslation.tsv"
        self.test_airr_tcr = "AIRR_rearrangement_tcr_test.tsv"
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_airr_sc_path = os.path.join(self.this_dir, self.test_airr_sc)
        self.test_airr_sc_df = pd.read_table(self.test_airr_sc_path, delimiter="\t", header=0)
        self.test_airr_bulk_path = os.path.join(self.this_dir, self.test_airr_bulk)
        self.test_airr_mixed_path = os.path.join(self.this_dir, self.test_airr_mixed)
        self.test_airr_translation_path = os.path.join(self.this_dir, self.test_airr_translation)
        self.test_airr_tcr_path = os.path.join(self.this_dir, self.test_airr_tcr)
        self.test_airr_tcr_df = pd.read_table(self.test_airr_tcr_path, delimiter="\t", header=0)
        self.test_mixed = "AIRR_rearrangement_mixed_bcr_tcr_test.tsv"
        self.test_mixed_path = os.path.join(self.this_dir, self.test_mixed)
        self.test_mixed_df = pd.read_table(self.test_mixed_path, delimiter="\t", header=0)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_esm2_sc_HL_embedding(self):
        """Test esm2 (single-cell HL)."""
        embed(self.test_airr_sc_path, "HL", "esm2", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

    def test_esm2_bulk_HL_embedding(self):
        """Test esm2 (bulk HL)."""
        with self.assertRaises(ValueError):
            embed(self.test_airr_bulk_path, "HL", "esm2", "HL_test.pt")

    def test_esm2_sc_H_embedding(self):
        """Test esm2 (single-cell H)."""
        embed(self.test_airr_sc_path, "H", "esm2", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_esm2_bulk_H_embedding(self):
        """Test antiberty (bulk H)."""
        embed(self.test_airr_bulk_path, "H", "esm2", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_esm2_sc_L_embedding(self):
        """Test esm2 (single-cell L)."""
        embed(self.test_airr_sc_path, "L", "esm2", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_esm2_bulk_L_embedding(self):
        """Test esm2 (bulk L)."""
        embed(self.test_airr_bulk_path, "L", "esm2", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_tcr_esm2_A_embedding(self):
        """Test ESM2 with TCR alpha chains (using unified approach)."""
        # Use existing esm2 function with TCR chain mapping: A -> L
        embed(self.test_airr_tcr_path, "L", "esm2", "tcr_esm2_A_test.pt", batch_size=2)
        assert os.path.exists("tcr_esm2_A_test.pt")
        embeddings = torch.load("tcr_esm2_A_test.pt")
        assert embeddings.shape[1] == 1280  # ESM2 embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcr_esm2_A_test.pt")

    def test_tcr_esm2_AB_embedding(self):
        """Test ESM2 with TCR alpha-beta pairs (using unified approach)."""
        # Use existing esm2 function with TCR chain mapping: AB -> HL
        embed(self.test_airr_tcr_path, "HL", "esm2", "tcr_esm2_AB_test.pt", batch_size=2)
        assert os.path.exists("tcr_esm2_AB_test.pt")
        embeddings = torch.load("tcr_esm2_AB_test.pt")
        assert embeddings.shape[1] == 1280  # ESM2 embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcr_esm2_AB_test.pt")

    @unittest.skipIf(SKIP_LARGE_MODELS, "Skipping ProtT5 test on GitHub Actions due to disk space limitations")
    def test_tcr_prott5_A_embedding(self):
        """Test ProtT5 with TCR alpha chains (using unified approach)."""
        # Use generic prott5 function with TCR chain mapping: A -> A (handled internally)
        embed(self.test_airr_tcr_path, "A", "prott5", "tcr_prott5_A_test.pt", batch_size=2)
        assert os.path.exists("tcr_prott5_A_test.pt")
        embeddings = torch.load("tcr_prott5_A_test.pt")
        # ProtT5: 1024 dim, ESM2 fallback: 1280 dim
        assert embeddings.shape[1] in [1024, 1280]
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcr_prott5_A_test.pt")

    @unittest.skipIf(SKIP_LARGE_MODELS, "Skipping ProtT5 test on GitHub Actions due to disk space limitations")
    def test_tcr_prott5_AB_embedding(self):
        """Test ProtT5 with TCR alpha-beta pairs (using unified approach)."""
        # Use generic prott5 function with TCR chain mapping: AB -> AB (handled internally)
        embed(self.test_airr_tcr_path, "AB", "prott5", "tcr_prott5_AB_test.pt", batch_size=2)
        assert os.path.exists("tcr_prott5_AB_test.pt")
        embeddings = torch.load("tcr_prott5_AB_test.pt")
        assert embeddings.shape[1] == 1024  # ProtT5 embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcr_prott5_AB_test.pt")

    @unittest.skipIf(SKIP_LARGE_MODELS, "Skipping ProtT5 test on GitHub Actions due to disk space limitations")
    def test_prott5_bcr_embedding(self):
        """Test ProtT5 with BCR data (backward compatibility)."""
        # Use generic prott5 function with BCR data
        embed(self.test_airr_sc_path, "HL", "prott5", "prott5_bcr_test.pt", batch_size=2)
        assert os.path.exists("prott5_bcr_test.pt")
        embeddings = torch.load("prott5_bcr_test.pt")
        assert embeddings.shape[1] == 1024  # ProtT5 embedding dimension
        assert embeddings.shape[0] == 2  # 2 heavy-light pairs in BCR test data
        os.remove("prott5_bcr_test.pt")
