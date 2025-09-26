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
from amulety.utils import process_airr


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

    # TCR-BERT tests (mixed data)
    def test_tcr_bert_mixed_HL_embedding(self):
        """Test TCR-BERT (mixed bulk sc HL)."""
        try:
            embed(input_airr=self.test_mixed_path, chain="HL", model="tcr-bert", output_file_path="HL_test.pt")
            assert os.path.exists("HL_test.pt")
            embeddings = torch.load("HL_test.pt")
            assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
            assert embeddings.shape[0] == 3  # 3 TCR cells with paired H and L chains
            os.remove("HL_test.pt")
            os.remove("HL_test_metadata.tsv")
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCR-BERT model loading failed: {e}")
            else:
                raise

    def test_tcr_bert_mixed_H_embedding_tsv(self):
        """Test TCR-BERT (mixed bulk sc H)."""
        try:
            embed(self.test_mixed_path, "H", "tcr-bert", "H_test.tsv")
            assert os.path.exists("H_test.tsv")
            embeddings = pd.read_table("H_test.tsv", delimiter="\t")
            assert embeddings.shape[1] == 769  # 768 + id
            assert embeddings.shape[0] == 3  # 3 H chains (TRB chains from 3 TCR cells)
            os.remove("H_test.tsv")
            os.remove("H_test_metadata.tsv")
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCR-BERT model loading failed: {e}")
            else:
                raise

    def test_tcr_bert_mixed_H_plus_L_embedding_tsv(self):
        """Test TCR-BERT (mixed bulk sc H+L)."""
        try:
            embed(self.test_mixed_path, "H+L", "tcr-bert", "H_plus_L_test.tsv")
            assert os.path.exists("H_plus_L_test.tsv")
            embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
            assert embeddings.shape[1] == 769  # 768 + id
            assert embeddings.shape[0] == 6  # 3 H chains (TRB) + 3 L chains (TRA) from 3 TCR cells
            os.remove("H_plus_L_test.tsv")
            os.remove("H_plus_L_test_metadata.tsv")
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCR-BERT model loading failed: {e}")
            else:
                raise

    def test_tcr_bert_mixed_H_embedding_residue_level(self):
        """Test TCR-BERT mixed bulk sc H with residue-level embeddings."""
        embed(self.test_mixed_path, "H", "tcr-bert", "H_residue_test.pt", residue_level=True)
        assert os.path.exists("H_residue_test.pt")
        embeddings = torch.load("H_residue_test.pt")
        assert embeddings.shape[0] == 3  # 3 H chains (TRB chains from 3 TCR cells)
        assert embeddings.shape[1] == 64  # max sequence length (padded)
        assert embeddings.shape[2] == 768  # TCR-BERT embedding dimension
        os.remove("H_residue_test.pt")
        os.remove("H_residue_test_metadata.tsv")

    # TCRT5 tests (mixed data)
    def test_tcrt5_mixed_H_embedding_tsv(self):
        """Test TCRT5 (mixed bulk sc H)."""
        try:
            embed(self.test_mixed_path, "H", "tcrt5", "tcrt5_H_test.tsv")
            assert os.path.exists("tcrt5_H_test.tsv")
            embeddings = pd.read_table("tcrt5_H_test.tsv", delimiter="\t")
            assert embeddings.shape[1] == 257  # 256 + id
            assert embeddings.shape[0] == 3  # 3 H chains (TRB chains from 3 TCR cells)
            os.remove("tcrt5_H_test.tsv")
            os.remove("tcrt5_H_test_metadata.tsv")
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCRT5 model loading failed: {e}")
            else:
                raise

    def test_tcrt5_mixed_L_embedding_should_warn(self):
        """Test TCRT5 with L chains should warn (only supports H chains)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embed(self.test_mixed_path, "L", "tcrt5", "tcrt5_L_test.tsv")

            warning_msg = str(w[0].message)
            assert "TCRT5 model was trained on" in warning_msg
            assert "beta chains for TCR" in warning_msg
            os.remove("tcrt5_L_test.tsv")
            os.remove("tcrt5_L_test_metadata.tsv")

    def test_tcrt5_mixed_HL_embedding_should_warn(self):
        """Test TCRT5 with HL chains should warn (only supports H chains)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embed(self.test_mixed_path, "HL", "tcrt5", "tcrt5_HL_test.tsv")

            warning_msg = str(w[0].message)
            assert "TCRT5 model was trained on" in warning_msg
            assert "beta chains for TCR" in warning_msg
            os.remove("tcrt5_HL_test.tsv")
            os.remove("tcrt5_HL_test_metadata.tsv")

    def test_tcrt5_mixed_H_embedding_residue_level(self):
        """Test TCRT5 mixed bulk sc H with residue-level embeddings."""
        embed(self.test_mixed_path, "H", "tcrt5", "tcrt5_H_residue_test.pt", residue_level=True)
        assert os.path.exists("tcrt5_H_residue_test.pt")
        embeddings = torch.load("tcrt5_H_residue_test.pt")
        assert embeddings.shape[0] == 3  # 3 H chains (TRB chains from 3 TCR cells)
        assert embeddings.shape[1] == 20  # TCRT5 embedding dimension
        assert embeddings.shape[2] == 256  # max sequence length (padded)
        os.remove("tcrt5_H_residue_test.pt")
        os.remove("tcrt5_H_residue_test_metadata.tsv")

    def test_custom_duplicate_column_tcr(self):
        """Test that we can pass any column name as duplicate_col for TCR data selection."""
        # Create test data with realistic selection columns that users might have
        test_data = pd.DataFrame(
            {
                "sequence_id": ["TCR_001", "TCR_002", "TCR_003", "TCR_004", "TCR_005", "TCR_006", "TCR_007"],
                "sequence_vdj_aa": [
                    "CASSLAPGATNEKLFF",
                    "CAVNTGNQFYF",
                    "CASSLVGQGAYEQYF",
                    "CAVRDMEYGNKLVF",
                    "CASSLPGQGAYEQYF",
                    "CAVKDSNYQLIW",
                    "CASSLAPGATNEKLFF",
                ],
                "cdr3_aa": [
                    "CASSLAPGATNEKLFF",
                    "CAVNTGNQFYF",
                    "CASSLVGQGAYEQYF",
                    "CAVRDMEYGNKLVF",
                    "CASSLPGQGAYEQYF",
                    "CAVKDSNYQLIW",
                    "CASSLAPGATNEKLFF",
                ],
                "locus": ["TRB", "TRA", "TRB", "TRA", "TRB", "TRA", "TRA"],
                "cell_id": ["cell_1", "cell_1", "cell_2", "cell_2", "cell_3", "cell_3", "cell_3"],
                "duplicate_count": [10, 8, 15, 12, 20, 18, 20],  # Default column
                "v_call": [
                    "TRBV7-9*01",
                    "TRAV8-4*01",
                    "TRBV7-2*01",
                    "TRAV13-1*01",
                    "TRBV7-2*01",
                    "TRAV12-1*01",
                    "TRAV8-4*01",
                ],
                "j_call": [
                    "TRBJ2-1*01",
                    "TRAJ49*01",
                    "TRBJ2-7*01",
                    "TRAJ56*01",
                    "TRBJ2-7*01",
                    "TRAJ33*01",
                    "TRAJ49*01",
                ],
                "umi_count": [100, 90, 150, 120, 200, 180, 1],  # UMI counts - realistic user column
                "read_count": [95, 85, 145, 115, 195, 175, 1],  # Read counts - another realistic column
            }
        )

        # Test with default duplicate_count column
        result_default, _ = process_airr(test_data, "HL", duplicate_col="duplicate_count")
        assert len(result_default) == 3  # 3 cells with HL pairs
        assert "CASSLAPGATNEKLFF" in result_default[2]

        # Test with umi_count column (realistic user column)
        result_umi, _ = process_airr(test_data, "HL", duplicate_col="umi_count")
        assert len(result_umi) == 3  # 3 cells with HL pairs

        # Test with read_count column (another realistic user column)
        result_reads, _ = process_airr(test_data, "HL", duplicate_col="read_count")
        assert len(result_reads) == 3  # 3 cells with HL pairs

        # Test error when duplicate_col doesn't exist
        with self.assertRaises(ValueError) as context:
            process_airr(test_data, "HL", duplicate_col="nonexistent_column")

        error_msg = str(context.exception)
        assert "nonexistent_column" in error_msg

        # Test error when duplicate_col is not numeric
        test_data_non_numeric = test_data.copy()
        test_data_non_numeric["text_col"] = ["A", "B", "C", "D", "E", "F", "G"]
        with self.assertRaises(ValueError) as context:
            process_airr(test_data_non_numeric, "HL", duplicate_col="text_col")

        error_msg = str(context.exception)
        assert "must be numeric" in error_msg

        print("PASS: Custom duplicate column test for TCR data successful")
