#!/usr/bin/env python

"""Tests for `amulety` package.
Tests can be run with the command:
python -m unittest test_amulety.py
"""

import os
import unittest

import pandas as pd
import pytest
import torch

from amulety.amulety import embed, translate_igblast


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

    def tearDown(self):
        """Tear down test fixtures, if any."""

    ##################
    # amulety tests #
    ##################

    def test_antiberty_sc_HL_embedding(self):
        """Test antiberty (single-cell HL)."""
        embed(input_airr=self.test_airr_sc_path, chain="HL", model="antiberty", output_file_path="HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

    def test_antiberty_mixed_HL_embedding_tsv(self):
        """Test antiberty (mixed BCR/TCR HL)."""
        embed(self.test_airr_mixed_path, "HL", "antiberty", "HL_test.tsv")
        assert os.path.exists("HL_test.tsv")
        embeddings = pd.read_table("HL_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 513
        assert embeddings.shape[0] == 1
        os.remove("HL_test.tsv")

    def test_antiberty_sc_H_embedding(self):
        """Test antiberty (single-cell H)."""
        embed(self.test_airr_sc_path, "H", "antiberty", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_antiberty_mixed_H_embedding_tsv(self):
        """Test antiberty (mixed BCR/TCR H)."""
        embed(self.test_airr_mixed_path, "H", "antiberty", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 514
        assert embeddings.shape[0] == 2
        os.remove("H_test.tsv")

    def test_antiberty_sc_L_embedding(self):
        """Test antiberty (single-cell L)."""
        embed(self.test_airr_sc_path, "L", "antiberty", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_antiberty_mixed_L_embedding_tsv(self):
        """Test antiberty (mixed BCR/TCR L)."""
        embed(self.test_airr_mixed_path, "L", "antiberty", "L_test.tsv")
        assert os.path.exists("L_test.tsv")
        embeddings = pd.read_table("L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 514
        assert embeddings.shape[0] == 2
        os.remove("L_test.tsv")

    def test_antiBERTa2_sc_HL_embedding(self):
        """Test antiBERTa2 (single-cell HL)."""
        embed(self.test_airr_sc_path, "HL", "antiberta2", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

    def test_antiberta2_mixed_HL_embedding_tsv(self):
        """Test antiberta2 (mixed BCR/TCR HL)."""
        embed(self.test_airr_mixed_path, "HL", "antiberta2", "HL_test.tsv")
        assert os.path.exists("HL_test.tsv")
        embeddings = pd.read_table("HL_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1025
        assert embeddings.shape[0] == 1
        os.remove("HL_test.tsv")

    def test_antiBERTa2_sc_H_embedding(self):
        """Test antiBERTa2 (single-cell H)."""
        embed(self.test_airr_sc_path, "H", "antiberta2", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_antiberta2_mixed_H_embedding_tsv(self):
        """Test antiberta2 (mixed BCR/TCR H)."""
        embed(self.test_airr_mixed_path, "H", "antiberta2", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1026
        assert embeddings.shape[0] == 2
        os.remove("H_test.tsv")

    def test_antiBERTa2_sc_L_embedding(self):
        """Test antiBERTa2 (single-cell L)."""
        embed(self.test_airr_sc_path, "L", "antiberta2", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_antiberta2_mixed_L_embedding_tsv(self):
        """Test antiberta2 (mixed BCR/TCR L)."""
        embed(self.test_airr_mixed_path, "L", "antiberta2", "L_test.tsv")
        assert os.path.exists("L_test.tsv")
        embeddings = pd.read_table("L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1026
        assert embeddings.shape[0] == 2
        os.remove("L_test.tsv")

    def test_balm_paired_sc_HL_embedding(self):
        """Test balm-paired (single-cell HL)."""
        embed(self.test_airr_sc_path, "HL", "balm-paired", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

    @pytest.mark.needsigblast  # mark test as needing igblast installation and databases, run with pytest --needsigblast
    def test_translation(self):
        """Test translation for IgBLAST works."""
        translate_igblast(
            self.test_airr_translation_path,
            self.this_dir,
            # ugly hack to get to the igblast_base directory in GitHub actions
            os.path.join(os.path.abspath(os.path.join(os.path.dirname(self.this_dir), os.pardir)), "igblast_base"),
        )
        igblast_outfile = os.path.join(self.this_dir, "AIRR_rearrangement_single-cell_testtranslation_translated.tsv")
        data_out = pd.read_table(igblast_outfile, delimiter="\t")
        assert (data_out["sequence_vdj_aa"] == data_out["sequence_vdj_aa_original"]).all()
        assert (data_out["sequence_alignment_aa"] == data_out["sequence_alignment_aa_original"]).all()
        assert (data_out["sequence_aa"] == data_out["sequence_aa_original"]).all()
        os.remove(igblast_outfile)
