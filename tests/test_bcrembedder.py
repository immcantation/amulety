#!/usr/bin/env python

"""Tests for `bcrembed` package.
Tests can be run with the command:
python -m unittest test_bcrembedder.py
"""

import unittest
import os
import torch
from bcrembed.__main__ import antiberty, antiberta2, esm2
from bcrembed.utils import translate_igblast
import pandas as pd

class TestBcrembedder(unittest.TestCase):
    """Function that runs at start of tests for common resources.
    """

    def setUp(self):
        """Set up test fixtures, if any."""
        self.test_airr_sc = "AIRR_rearrangement_translated_single-cell.tsv"
        self.test_airr_bulk = "AIRR_rearrangement_translated_bulk.tsv"
        self.test_airr_mixed = "AIRR_rearrangement_translated_mixed.tsv"
        self.test_airr_translation = "AIRR_rearrangement_single-cell_testtranslation.tsv"
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_airr_sc_path = os.path.join(self.this_dir, self.test_airr_sc)
        self.test_airr_bulk_path = os.path.join(self.this_dir, self.test_airr_bulk)
        self.test_airr_mixed_path = os.path.join(self.this_dir, self.test_airr_mixed)
        self.test_airr_translation_path = os.path.join(self.this_dir, self.test_airr_translation)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    ##################
    # BCRembed tests #
    ##################

    def test_antiberty_sc_HL_embedding(self):
        """Test antiberty (single-cell HL)."""
        antiberty(self.test_airr_sc_path, "HL", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

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

    def test_translation(self):
        """Test translation for IgBLAST works."""
        translate_igblast(self.test_airr_translation_path,
                                 self.this_dir, "../igblast_base")
        igblast_outfile = os.path.join(self.this_dir, "AIRR_rearrangement_single-cell_testtranslation_translated.tsv")
        data_out = pd.read_table(igblast_outfile, delimiter="\t")
        assert (data_out["sequence_vdj_aa"]==data_out["sequence_vdj_aa_original"]).all()
        assert (data_out["sequence_alignment_aa"]==data_out["sequence_alignment_aa_original"]).all()
        assert (data_out["sequence_aa"]==data_out["sequence_aa_original"]).all()
        os.remove(igblast_outfile)

