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

from amulety.amulety import antiberta2, antiberty, balm_paired, esm2, prott5, tcr_bert, translate_igblast


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
        self.test_airr_bulk_path = os.path.join(self.this_dir, self.test_airr_bulk)
        self.test_airr_mixed_path = os.path.join(self.this_dir, self.test_airr_mixed)
        self.test_airr_translation_path = os.path.join(self.this_dir, self.test_airr_translation)
        self.test_airr_tcr_path = os.path.join(self.this_dir, self.test_airr_tcr)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    ##################
    # amulety tests #
    ##################

    def test_antiberty_sc_HL_embedding(self):
        """Test antiberty (single-cell HL)."""
        antiberty(self.test_airr_sc_path, "HL", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

    def test_antiberty_mixed_HL_embedding_tsv(self):
        """Test antiberty (single-cell and bulk HL)."""
        antiberty(self.test_airr_mixed_path, "HL", "HL_test.tsv")
        assert os.path.exists("HL_test.tsv")
        embeddings = pd.read_table("HL_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 513
        assert embeddings.shape[0] == 1
        os.remove("HL_test.tsv")

    def test_antiberty_bulk_HL_embedding(self):
        """Test antiberty (bulk HL)."""
        with self.assertRaises(ValueError):
            antiberty(self.test_airr_bulk_path, "HL", "HL_test.pt")

    def test_antiberty_sc_H_embedding(self):
        """Test antiberty (single-cell H)."""
        antiberty(self.test_airr_sc_path, "H", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_antiberty_mixed_H_embedding_tsv(self):
        """Test antiberty (single-cell and bulk H)."""
        antiberty(self.test_airr_mixed_path, "H", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 514
        assert embeddings.shape[0] == 2
        os.remove("H_test.tsv")

    def test_antiberty_bulk_H_embedding(self):
        """Test antiberty (bulk H)."""
        antiberty(self.test_airr_bulk_path, "H", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_antiberty_sc_L_embedding(self):
        """Test antiberty (single-cell L)."""
        antiberty(self.test_airr_sc_path, "L", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_antiberty_mixed_L_embedding_tsv(self):
        """Test antiberty (single-cell and bulk L)."""
        antiberty(self.test_airr_mixed_path, "L", "L_test.tsv")
        assert os.path.exists("L_test.tsv")
        embeddings = pd.read_table("L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 514
        assert embeddings.shape[0] == 2
        os.remove("L_test.tsv")

    def test_antiberty_bulk_L_embedding(self):
        """Test antiberty (bulk L)."""
        antiberty(self.test_airr_bulk_path, "L", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_esm2_sc_HL_embedding(self):
        """Test esm2 (single-cell HL)."""
        esm2(self.test_airr_sc_path, "HL", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

    def test_esm2_bulk_HL_embedding(self):
        """Test esm2 (bulk HL)."""
        with self.assertRaises(ValueError):
            esm2(self.test_airr_bulk_path, "HL", "HL_test.pt")

    def test_esm2_sc_H_embedding(self):
        """Test esm2 (single-cell H)."""
        esm2(self.test_airr_sc_path, "H", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_esm2_bulk_H_embedding(self):
        """Test antiberty (bulk H)."""
        esm2(self.test_airr_bulk_path, "H", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_esm2_sc_L_embedding(self):
        """Test esm2 (single-cell L)."""
        esm2(self.test_airr_sc_path, "L", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_esm2_bulk_L_embedding(self):
        """Test esm2 (bulk L)."""
        esm2(self.test_airr_bulk_path, "L", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_antiBERTa2_sc_HL_embedding(self):
        """Test antiBERTa2 (single-cell HL)."""
        antiberta2(self.test_airr_sc_path, "HL", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

    def test_antiberta2_mixed_HL_embedding_tsv(self):
        """Test antiberta2 (single-cell and bulk HL)."""
        antiberta2(self.test_airr_mixed_path, "HL", "HL_test.tsv")
        assert os.path.exists("HL_test.tsv")
        embeddings = pd.read_table("HL_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1025
        assert embeddings.shape[0] == 1
        os.remove("HL_test.tsv")

    def test_antiBERTa2_bulk_HL_embedding(self):
        """Test antiBERTa2 (bulk HL)."""
        with self.assertRaises(ValueError):
            antiberta2(self.test_airr_bulk_path, "HL", "HL_test.pt")

    def test_antiBERTa2_sc_H_embedding(self):
        """Test antiBERTa2 (single-cell H)."""
        antiberta2(self.test_airr_sc_path, "H", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_antiberta2_mixed_H_embedding_tsv(self):
        """Test antiberta2 (single-cell and bulk H)."""
        antiberta2(self.test_airr_mixed_path, "H", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1026
        assert embeddings.shape[0] == 2
        os.remove("H_test.tsv")

    def test_antiBERTa2_bulk_H_embedding(self):
        """Test antiBERTa2 (bulk H)."""
        antiberta2(self.test_airr_bulk_path, "H", "H_test.pt")
        assert os.path.exists("H_test.pt")
        embeddings = torch.load("H_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("H_test.pt")

    def test_antiBERTa2_sc_L_embedding(self):
        """Test antiBERTa2 (single-cell L)."""
        antiberta2(self.test_airr_sc_path, "L", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_antiBERTa2_bulk_L_embedding(self):
        """Test antiBERTa2 (bulk L)."""
        antiberta2(self.test_airr_bulk_path, "L", "L_test.pt")
        assert os.path.exists("L_test.pt")
        embeddings = torch.load("L_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("L_test.pt")

    def test_antiberta2_mixed_L_embedding_tsv(self):
        """Test antiberta2 (single-cell and bulk L)."""
        antiberta2(self.test_airr_mixed_path, "L", "L_test.tsv")
        assert os.path.exists("L_test.tsv")
        embeddings = pd.read_table("L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1026
        assert embeddings.shape[0] == 2
        os.remove("L_test.tsv")

    def test_balm_paired_sc_HL_embedding(self):
        """Test antiBERTa2 (single-cell HL)."""
        balm_paired(self.test_airr_sc_path, "HL", "HL_test.pt")
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

    ################
    # TCR tests    #
    ################

    def test_tcr_bert_A_embedding(self):
        """Test TCR-BERT (alpha chains)."""
        tcr_bert(self.test_airr_tcr_path, "A", "tcr_A_test.pt", batch_size=2)
        assert os.path.exists("tcr_A_test.pt")
        embeddings = torch.load("tcr_A_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcr_A_test.pt")

    def test_tcr_bert_B_embedding(self):
        """Test TCR-BERT (beta chains)."""
        tcr_bert(self.test_airr_tcr_path, "B", "tcr_B_test.pt", batch_size=2)
        assert os.path.exists("tcr_B_test.pt")
        embeddings = torch.load("tcr_B_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 beta chains in test data
        os.remove("tcr_B_test.pt")

    def test_tcr_bert_AB_embedding(self):
        """Test TCR-BERT (alpha-beta pairs)."""
        tcr_bert(self.test_airr_tcr_path, "AB", "tcr_AB_test.pt", batch_size=2)
        assert os.path.exists("tcr_AB_test.pt")
        embeddings = torch.load("tcr_AB_test.pt")
        assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcr_AB_test.pt")

    def test_tcr_esm2_A_embedding(self):
        """Test ESM2 with TCR alpha chains (using unified approach)."""
        # Use existing esm2 function with TCR chain mapping: A -> L
        esm2(self.test_airr_tcr_path, "L", "tcr_esm2_A_test.pt", batch_size=2)
        assert os.path.exists("tcr_esm2_A_test.pt")
        embeddings = torch.load("tcr_esm2_A_test.pt")
        assert embeddings.shape[1] == 1280  # ESM2 embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcr_esm2_A_test.pt")

    def test_tcr_esm2_AB_embedding(self):
        """Test ESM2 with TCR alpha-beta pairs (using unified approach)."""
        # Use existing esm2 function with TCR chain mapping: AB -> HL
        esm2(self.test_airr_tcr_path, "HL", "tcr_esm2_AB_test.pt", batch_size=2)
        assert os.path.exists("tcr_esm2_AB_test.pt")
        embeddings = torch.load("tcr_esm2_AB_test.pt")
        assert embeddings.shape[1] == 1280  # ESM2 embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcr_esm2_AB_test.pt")

    def test_tcr_prott5_A_embedding(self):
        """Test ProtT5 with TCR alpha chains (using unified approach)."""
        # Use generic prott5 function with TCR chain mapping: A -> A (handled internally)
        prott5(self.test_airr_tcr_path, "A", "tcr_prott5_A_test.pt", batch_size=2)
        assert os.path.exists("tcr_prott5_A_test.pt")
        embeddings = torch.load("tcr_prott5_A_test.pt")
        assert embeddings.shape[1] == 1024  # ProtT5 embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcr_prott5_A_test.pt")

    def test_tcr_prott5_AB_embedding(self):
        """Test ProtT5 with TCR alpha-beta pairs (using unified approach)."""
        # Use generic prott5 function with TCR chain mapping: AB -> AB (handled internally)
        prott5(self.test_airr_tcr_path, "AB", "tcr_prott5_AB_test.pt", batch_size=2)
        assert os.path.exists("tcr_prott5_AB_test.pt")
        embeddings = torch.load("tcr_prott5_AB_test.pt")
        assert embeddings.shape[1] == 1024  # ProtT5 embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcr_prott5_AB_test.pt")

    def test_prott5_bcr_embedding(self):
        """Test ProtT5 with BCR data (backward compatibility)."""
        # Use generic prott5 function with BCR data
        prott5(self.test_airr_sc_path, "HL", "prott5_bcr_test.pt", batch_size=2)
        assert os.path.exists("prott5_bcr_test.pt")
        embeddings = torch.load("prott5_bcr_test.pt")
        assert embeddings.shape[1] == 1024  # ProtT5 embedding dimension
        assert embeddings.shape[0] == 2  # 2 heavy-light pairs in BCR test data
        os.remove("prott5_bcr_test.pt")

    def test_tcr_unified_chain_mapping(self):
        """Test that TCR chains are correctly mapped to BCR chain system."""
        from amulety.utils import process_airr

        # Test alpha chains (A -> L)
        result_alpha = process_airr(self.test_airr_tcr_path, "L")
        assert result_alpha.shape[0] == 3  # 3 alpha chains

        # Test beta chains (B -> H)
        result_beta = process_airr(self.test_airr_tcr_path, "H")
        assert result_beta.shape[0] == 3  # 3 beta chains

        # Test alpha-beta pairs (AB -> HL)
        result_pairs = process_airr(self.test_airr_tcr_path, "HL")
        assert result_pairs.shape[0] == 3  # 3 alpha-beta pairs

    def test_tcr_concatenation_order(self):
        """Test that TCR concatenation follows Beta+Alpha order."""
        import pandas as pd

        from amulety.utils import process_airr

        # Get concatenated pairs
        result = process_airr(self.test_airr_tcr_path, "HL")

        # Read original data to verify order
        data = pd.read_table(self.test_airr_tcr_path)

        # Check first pair
        if result.shape[0] > 0:
            seq = result.iloc[0]["sequence_vdj_aa"]
            assert "<cls><cls>" in seq

            parts = seq.split("<cls><cls>")
            # First part should be beta (TRB), second part should be alpha (TRA)
            beta_seq = data[data.locus == "TRB"].iloc[0]["sequence_vdj_aa"]
            alpha_seq = data[data.locus == "TRA"].iloc[0]["sequence_vdj_aa"]

            assert parts[0] == beta_seq  # Beta first
            assert parts[1] == alpha_seq  # Alpha second

    def test_tcr_backward_compatibility(self):
        """Test that adding TCR support doesn't break BCR functionality."""
        from amulety.utils import process_airr

        # Test with existing BCR data
        result_bcr_h = process_airr(self.test_airr_sc_path, "H")
        result_bcr_l = process_airr(self.test_airr_sc_path, "L")
        result_bcr_hl = process_airr(self.test_airr_sc_path, "HL")

        # Should still work as before
        assert result_bcr_h.shape[0] == 2  # 2 heavy chains
        assert result_bcr_l.shape[0] == 2  # 2 light chains
        assert result_bcr_hl.shape[0] == 2  # 2 heavy-light pairs
