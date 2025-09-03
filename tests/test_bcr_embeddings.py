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
        self.test_airr_sc = "AIRR_rearrangement_translated_single-cell.tsv"

        self.test_airr_mixed = "AIRR_rearrangement_translated_mixed.tsv"
        self.test_airr_translation = "AIRR_rearrangement_single-cell_testtranslation.tsv"
        self.test_airr_tcr = "AIRR_rearrangement_tcr_test.tsv"
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_airr_sc_path = os.path.join(self.this_dir, self.test_airr_sc)
        self.test_airr_sc_df = pd.read_table(self.test_airr_sc_path, delimiter="\t", header=0)

        self.test_airr_mixed_path = os.path.join(self.this_dir, self.test_airr_mixed)
        self.test_airr_translation_path = os.path.join(self.this_dir, self.test_airr_translation)
        self.test_airr_tcr_path = os.path.join(self.this_dir, self.test_airr_tcr)
        self.test_airr_tcr_df = pd.read_table(self.test_airr_tcr_path, delimiter="\t", header=0)
        self.test_mixed = "AIRR_rearrangement_mixed_bcr_tcr_test.tsv"
        self.test_mixed_path = os.path.join(self.this_dir, self.test_mixed)
        self.test_mixed_df = pd.read_table(self.test_mixed_path, delimiter="\t", header=0)

        # Bulk test data (no cell_id column)
        self.test_bulk = "AIRR_rearrangement_bulk_test.tsv"
        self.test_bulk_path = os.path.join(self.this_dir, self.test_bulk)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    # ablang tests
    def test_ablang_mixed_HL_embedding(self):
        """Test AbLang (mixed bulk sc H+L)."""
        with self.assertWarns(UserWarning) as cm:
            embed(input_airr=self.test_airr_mixed_path, chain="HL", model="ablang", output_file_path="HL_test.pt")
            assert os.path.exists("HL_test.pt")
            embeddings = torch.load("HL_test.pt")
            assert embeddings.shape[1] == 768  # AbLang embedding dimension
            assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
        self.assertIn("trained on individual chains only", str(cm.warning))
        os.remove("HL_test.pt")

    def test_ablang_mixed_H_plus_L_embedding_tsv(self):
        """Test AbLang (mixed bulk sc H+L)."""
        embed(self.test_airr_mixed_path, "H+L", "ablang", "H_plus_L_test.tsv")
        assert os.path.exists("H_plus_L_test.tsv")
        embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 771  # 768 + cell_id + chain + sequence_id
        assert (
            embeddings.shape[0] == 4
        )  # 2 H chain + 2 L chain (only the most abundant L chain per cell kept for single-cell data)
        os.remove("H_plus_L_test.tsv")

    def test_ablang_mixed_H_embedding_tsv(self):
        """Test AbLang H."""
        embed(self.test_airr_mixed_path, "H", "ablang", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 771  # 768 + cell_id + chain + sequence_id
        assert (
            embeddings.shape[0] == 2
        )  # 2 H chain + 2 L chain (only the most abundant L chain per cell kept for single-cell data)
        os.remove("H_test.tsv")

    # antiberty tests
    def test_antiberty_mixed_HL_embedding(self):
        """Test antiberty (mixed bulk sc H+L)."""
        with self.assertWarns(UserWarning) as cm:
            embed(input_airr=self.test_airr_mixed_path, chain="HL", model="antiberty", output_file_path="HL_test.pt")
            assert os.path.exists("HL_test.pt")
            embeddings = torch.load("HL_test.pt")
            assert embeddings.shape[1] == 512
            assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
            os.remove("HL_test.pt")
        self.assertIn("was trained on individual chains only", str(cm.warning))

    def test_antiberty_mixed_H_plus_L_embedding_tsv(self):
        """Test antiberty (mixed bulk sc H+L)."""
        embed(self.test_airr_mixed_path, "H+L", "antiberty", "H_plus_L_test.tsv")
        assert os.path.exists("H_plus_L_test.tsv")
        embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 515  # 512 + cell_id + chain + sequence_id
        assert (
            embeddings.shape[0] == 4
        )  # 2 H chain + 2 L chain (only the most abundant L chain per cell kept for single-cell data)
        os.remove("H_plus_L_test.tsv")

    def test_antiberty_mixed_H_embedding_tsv(self):
        """Test antiberty (mixed bulk sc H)."""
        embed(self.test_airr_mixed_path, "H", "antiberty", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 515  # 512 + cell_id + chain + sequence_id
        assert (
            embeddings.shape[0] == 2
        )  # 2 H chain + 2 L chain (only the most abundant L chain per cell kept for single-cell data)
        os.remove("H_test.tsv")

    # antiberta2 tests
    def test_antiberta2_mixed_HL_embedding(self):
        """Test antiberta2 (mixed bulk sc H+L)."""
        with self.assertWarns(UserWarning) as cm:
            embed(input_airr=self.test_airr_mixed_path, chain="HL", model="antiberta2", output_file_path="HL_test.pt")
            assert os.path.exists("HL_test.pt")
            embeddings = torch.load("HL_test.pt")
            assert embeddings.shape[1] == 1024  # antiberta2 embedding dimension
            assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
        self.assertIn("trained on individual chains only", str(cm.warning))
        os.remove("HL_test.pt")

    def test_antiberta2_mixed_H_plus_L_embedding_tsv(self):
        """Test antiberta2 (mixed bulk sc H+L)."""
        embed(self.test_airr_mixed_path, "H+L", "antiberta2", "H_plus_L_test.tsv")
        assert os.path.exists("H_plus_L_test.tsv")
        embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1027  # 1024 + cell_id + chain + sequence_id
        assert (
            embeddings.shape[0] == 4
        )  # 2 H chain + 2 L chain (only the most abundant L chain per cell kept for single-cell data)
        os.remove("H_plus_L_test.tsv")

    def test_antiberta2_mixed_H_embedding_tsv(self):
        """Test antiberta2 (mixed bulk sc H)."""
        embed(self.test_airr_mixed_path, "H", "antiberta2", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1027  # 1024 + cell_id + chain + sequence_id
        assert (
            embeddings.shape[0] == 2
        )  # 2 H chain + 2 L chain (only the most abundant L chain per cell kept for single-cell data)
        os.remove("H_test.tsv")

    # balm-paired tests
    def test_balm_paired_mixed_HL_embedding(self):
        """Test balm-paired (mixed bulk sc HL)."""
        embed(input_airr=self.test_airr_mixed_path, chain="HL", model="balm-paired", output_file_path="HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 1024  # balm-paired embedding dimension
        assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
        os.remove("HL_test.pt")

    def test_balm_paired_mixed_H_plus_L_embedding_tsv(self):
        """Test balm-paired (mixed bulk sc H+L)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_mixed_path, "H+L", "balm-paired", "H_plus_L_test.tsv")
        self.assertIn("was trained on paired chains", str(context.exception))
        self.assertIn("--chain HL", str(context.exception))

    def test_balm_paired_mixed_H_embedding_tsv(self):
        """Test balm-paired (mixed bulk sc H)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_mixed_path, "H", "balm-paired", "H_test.tsv")
        self.assertIn("was trained on paired chains", str(context.exception))
        self.assertIn("--chain HL", str(context.exception))
