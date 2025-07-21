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

        # Bulk test data (no cell_id column)
        self.test_bulk = "AIRR_rearrangement_bulk_test.tsv"
        self.test_bulk_path = os.path.join(self.this_dir, self.test_bulk)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    ##################
    # amulety tests #
    ##################

    def test_antiberty_sc_H_plus_L_embedding(self):
        """Test antiberty (single-cell H+L)."""
        embed(input_airr=self.test_airr_sc_path, chain="H+L", model="antiberty", output_file_path="H_plus_L_test.pt")
        assert os.path.exists("H_plus_L_test.pt")
        embeddings = torch.load("H_plus_L_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 4  # 2 H chains + 2 L chains
        os.remove("H_plus_L_test.pt")

    def test_antiberty_mixed_H_plus_L_embedding_tsv(self):
        """Test antiberty (mixed BCR/TCR H+L)."""
        embed(self.test_airr_mixed_path, "H+L", "antiberty", "H_plus_L_test.tsv")
        assert os.path.exists("H_plus_L_test.tsv")
        embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 515  # 512 + cell_id + chain + sequence_id
        assert embeddings.shape[0] == 2  # 1 H chain + 1 L chain (only single-cell data processed)
        os.remove("H_plus_L_test.tsv")

    def test_antiberty_HL_chain_validation(self):
        """Test that antiberty rejects HL chain (individual chain model)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_sc_path, "HL", "antiberty", "should_fail.pt")
        self.assertIn("supports individual chains only", str(context.exception))
        self.assertIn("--chain H", str(context.exception))
        self.assertIn("--chain L", str(context.exception))

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

    def test_ablang_sc_H_embedding(self):
        """Test AbLang (single-cell H)."""
        embed(self.test_airr_sc_path, "H", "ablang", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 768  # AbLang embedding dimension
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_ablang_sc_L_embedding(self):
        """Test AbLang (single-cell L)."""
        embed(self.test_airr_sc_path, "L", "ablang", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 768  # AbLang embedding dimension
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_ablang_sc_H_plus_L_embedding(self):
        """Test AbLang (single-cell H+L)."""
        embed(self.test_airr_sc_path, "H+L", "ablang", "H_plus_L_test.pt")
        assert os.path.exists("H_plus_L_test.pt")
        embeddings = torch.load("H_plus_L_test.pt")
        assert embeddings.shape[1] == 768  # AbLang embedding dimension
        assert embeddings.shape[0] == 4  # 2 H chains + 2 L chains
        os.remove("H_plus_L_test.pt")

    def test_ablang_HL_chain_validation(self):
        """Test that AbLang rejects HL chains (individual chain model)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_sc_path, "HL", "ablang", "should_fail.pt")
        self.assertIn("supports individual chains only", str(context.exception))
        self.assertIn("--chain H", str(context.exception))
        self.assertIn("--chain L", str(context.exception))

    def test_ablang_LH_chain_validation(self):
        """Test that AbLang rejects LH chains (individual chain model)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_sc_path, "LH", "ablang", "should_fail.pt")
        self.assertIn("supports individual chains only", str(context.exception))
        self.assertIn("--chain H", str(context.exception))
        self.assertIn("--chain L", str(context.exception))

    def test_antiBERTa2_sc_H_plus_L_embedding(self):
        """Test antiBERTa2 (single-cell H+L)."""
        embed(self.test_airr_sc_path, "H+L", "antiberta2", "H_plus_L_test.pt")
        assert os.path.exists("H_plus_L_test.pt")
        embeddings = torch.load("H_plus_L_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 4  # 2 H chains + 2 L chains
        os.remove("H_plus_L_test.pt")

    def test_antiberta2_mixed_H_plus_L_embedding_tsv(self):
        """Test antiberta2 (mixed BCR/TCR H+L)."""
        embed(self.test_airr_mixed_path, "H+L", "antiberta2", "H_plus_L_test.tsv")
        assert os.path.exists("H_plus_L_test.tsv")
        embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1027  # 1024 + cell_id + chain + sequence_id
        assert embeddings.shape[0] == 2  # 1 H chain + 1 L chain (only single-cell data processed)
        os.remove("H_plus_L_test.tsv")

    def test_antiberta2_HL_chain_validation(self):
        """Test that antiberta2 rejects HL chain (individual chain model)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_sc_path, "HL", "antiberta2", "should_fail.pt")
        self.assertIn("supports individual chains only", str(context.exception))
        self.assertIn("--chain H", str(context.exception))
        self.assertIn("--chain L", str(context.exception))

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

    def test_balm_paired_sc_LH_embedding(self):
        """Test balm-paired (single-cell LH with warning)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embed(self.test_airr_sc_path, "LH", "balm-paired", "LH_test.pt")
            assert os.path.exists("LH_test.pt")
            embeddings = torch.load("LH_test.pt")
            assert embeddings.shape[1] == 1024
            assert embeddings.shape[0] == 2
            os.remove("LH_test.pt")
            # Check that LH warning was issued
            assert len(w) > 0
            assert any("LH (Light-Heavy) chain order detected" in str(warning.message) for warning in w)

    def test_balm_paired_H_chain_validation(self):
        """Test that balm-paired rejects individual H chains (paired-only model)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_sc_path, "H", "balm-paired", "should_fail.pt")
        self.assertIn("requires paired chains", str(context.exception))
        self.assertIn("--chain HL", str(context.exception))

    def test_balm_paired_L_chain_validation(self):
        """Test that balm-paired rejects individual L chains (paired-only model)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_sc_path, "L", "balm-paired", "should_fail.pt")
        self.assertIn("requires paired chains", str(context.exception))
        self.assertIn("--chain HL", str(context.exception))

    def test_balm_paired_H_plus_L_chain_validation(self):
        """Test that balm-paired rejects H+L chains (paired-only model)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_sc_path, "H+L", "balm-paired", "should_fail.pt")
        self.assertIn("requires paired chains", str(context.exception))
        self.assertIn("--chain HL", str(context.exception))

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

    ##################
    # Bulk AIRR tests #
    ##################

    def test_antiberty_bulk_H_embedding(self):
        """Test antiberty with bulk H chain data."""
        embed(self.test_bulk_path, "H", "antiberty", "bulk_H_test.pt")
        assert os.path.exists("bulk_H_test.pt")
        embeddings = torch.load("bulk_H_test.pt")
        assert embeddings.shape[1] == 512  # AntiBERTy embedding dimension
        assert embeddings.shape[0] == 4  # 4 H chains in bulk test data (2 BCR + 2 TCR)
        os.remove("bulk_H_test.pt")

    def test_antiberty_bulk_L_embedding(self):
        """Test antiberty with bulk L chain data."""
        embed(self.test_bulk_path, "L", "antiberty", "bulk_L_test.pt")
        assert os.path.exists("bulk_L_test.pt")
        embeddings = torch.load("bulk_L_test.pt")
        assert embeddings.shape[1] == 512  # AntiBERTy embedding dimension
        assert embeddings.shape[0] == 4  # 4 L chains in bulk test data (2 BCR + 2 TCR)
        os.remove("bulk_L_test.pt")

    def test_antiberta2_bulk_H_embedding(self):
        """Test antiberta2 with bulk H chain data."""
        embed(self.test_bulk_path, "H", "antiberta2", "bulk_H_test.pt")
        assert os.path.exists("bulk_H_test.pt")
        embeddings = torch.load("bulk_H_test.pt")
        assert embeddings.shape[1] == 1024  # AntiBERTa2 embedding dimension
        assert embeddings.shape[0] == 4  # 4 H chains in bulk test data
        os.remove("bulk_H_test.pt")

    def test_ablang_bulk_H_embedding(self):
        """Test AbLang with bulk H chain data."""
        embed(self.test_bulk_path, "H", "ablang", "bulk_H_test.pt")
        assert os.path.exists("bulk_H_test.pt")
        embeddings = torch.load("bulk_H_test.pt")
        assert embeddings.shape[1] == 768  # AbLang embedding dimension
        assert embeddings.shape[0] == 4  # 4 H chains in bulk test data
        os.remove("bulk_H_test.pt")

    def test_bulk_invalid_chain_types(self):
        """Test that bulk data rejects paired chain types."""
        # Test that HL is not allowed for bulk data - should fail on model compatibility first
        with pytest.raises(ValueError, match="antiberty model supports individual chains only"):
            embed(self.test_bulk_path, "HL", "antiberty", "should_fail.pt")

        # Test that H+L is not allowed for bulk data
        with pytest.raises(ValueError, match='chain = "H\\+L" invalid for bulk mode'):
            embed(self.test_bulk_path, "H+L", "antiberty", "should_fail.pt")
