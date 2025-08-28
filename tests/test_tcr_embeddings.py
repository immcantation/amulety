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
            assert embeddings.shape[1] == 771  # 768 + cell_id + chain + sequence_id
            assert embeddings.shape[0] == 3  # 3 H chains (TRB chains from 3 TCR cells)
            os.remove("H_test.tsv")
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
            assert embeddings.shape[1] == 771  # 768 + cell_id + chain + sequence_id
            assert embeddings.shape[0] == 6  # 3 H chains (TRB) + 3 L chains (TRA) from 3 TCR cells
            os.remove("H_plus_L_test.tsv")
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCR-BERT model loading failed: {e}")
            else:
                raise

    # TCRT5 tests (mixed data)
    def test_tcrt5_mixed_H_embedding_tsv(self):
        """Test TCRT5 (mixed bulk sc H)."""
        try:
            embed(self.test_mixed_path, "H", "tcrt5", "tcrt5_H_test.tsv")
            assert os.path.exists("tcrt5_H_test.tsv")
            embeddings = pd.read_table("tcrt5_H_test.tsv", delimiter="\t")
            assert embeddings.shape[1] == 259  # 256 + cell_id + chain + sequence_id
            assert embeddings.shape[0] == 3  # 3 H chains (TRB chains from 3 TCR cells)
            os.remove("tcrt5_H_test.tsv")
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

    def test_tcrt5_mixed_HL_embedding_should_warn(self):
        """Test TCRT5 with HL chains should warn (only supports H chains)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embed(self.test_mixed_path, "HL", "tcrt5", "tcrt5_HL_test.tsv")

            warning_msg = str(w[0].message)
            assert "TCRT5 model was trained on" in warning_msg
            assert "beta chains for TCR" in warning_msg

    # def test_tcremp_command_not_available(self):
    #     """Test that TCREMP raises ImportError when command-line tool is not available."""
    #     import pandas as pd

    #     from amulety.tcr_embeddings import tcremp

    #     test_sequences = pd.Series(["CASSLAPGATNEKLFF", "CAVKDSNYQLIW"])

    #     # Most systems won't have tcremp-run installed, so this should raise ImportError
    #     try:
    #         # Use skip_clustering=True to match successful implementation
    #         result = tcremp(test_sequences, chain="H", skip_clustering=True)
    #         # If this succeeds, tcremp-run is available and working
    #         assert isinstance(result, torch.Tensor)
    #         assert result.shape[0] == len(test_sequences)
    #         print("PASS: TCREMP command-line tool is available and working")
    #     except ImportError as e:
    #         error_msg = str(e)
    #         print(f"PASS: Proper ImportError caught: {error_msg[:100]}...")

    #         # Verify error message quality
    #         assert "TCREMP command-line tool is required but not installed" in error_msg
    #         assert "git clone https://github.com/antigenomics/tcremp.git" in error_msg
    #         assert "tcremp-run -h" in error_msg
    #         assert "tcr-bert" in error_msg  # Alternative suggestions

    # def test_tcremp_L_embedding_direct_call(self):
    #     """Test TCREMP (alpha/gamma chains - L chains for TCR) using direct call."""
    #     from amulety.tcr_embeddings import tcremp

    #     # Extract alpha chains (L chains for TCR) from test data
    #     alpha_chains = self.test_airr_tcr_df[self.test_airr_tcr_df["locus"] == "TRA"]
    #     cdr3_sequences = alpha_chains["cdr3_aa"].dropna()

    #     try:
    #         # Use direct TCREMP call with skip_clustering=True (tested and working method)
    #         embeddings = tcremp(cdr3_sequences, chain="L", skip_clustering=True)
    #         # If successful, tcremp-run is available and working
    #         assert isinstance(embeddings, torch.Tensor)
    #         assert embeddings.shape[0] == len(cdr3_sequences)  # Number of alpha chains
    #         print(f"PASS: TCREMP L embedding successful - {embeddings.shape}")
    #     except (ImportError, RuntimeError) as e:
    #         # Handle both ImportError (command not available) and RuntimeError (TCREMP internal errors)
    #         if "not installed" in str(e):
    #             print("PASS: TCREMP L embedding failed as expected (command not available)")
    #         else:
    #             print(f"PASS: TCREMP L embedding failed due to internal TCREMP issues: {str(e)[:100]}...")

    # def test_tcremp_H_embedding_direct_call(self):
    #     """Test TCREMP (beta/delta chains - H chains for TCR) using direct call."""
    #     from amulety.tcr_embeddings import tcremp

    #     # Extract beta chains (H chains for TCR) from test data
    #     beta_chains = self.test_airr_tcr_df[self.test_airr_tcr_df["locus"] == "TRB"]
    #     cdr3_sequences = beta_chains["cdr3_aa"].dropna()

    #     try:
    #         # Use direct TCREMP call with skip_clustering=True (tested and working method)
    #         embeddings = tcremp(cdr3_sequences, chain="H", skip_clustering=True)
    #         # If successful, tcremp-run is available and working
    #         assert isinstance(embeddings, torch.Tensor)
    #         assert embeddings.shape[0] == len(cdr3_sequences)  # Number of beta chains
    #         print(f"PASS: TCREMP H embedding successful - {embeddings.shape}")
    #     except (ImportError, RuntimeError) as e:
    #         # Handle both ImportError (command not available) and RuntimeError (TCREMP internal errors)
    #         if "not installed" in str(e):
    #             print("PASS: TCREMP H embedding failed as expected (command not available)")
    #         else:
    #             print(f"PASS: TCREMP H embedding failed due to internal TCREMP issues: {str(e)[:100]}...")

    # def test_tcremp_H_plus_L_embedding_direct_call(self):
    #     """Test TCREMP (both alpha and beta chains separately - H+L for TCR) using direct call."""
    #     from amulety.tcr_embeddings import tcremp

    #     # Extract all chains from test data
    #     all_chains = self.test_airr_tcr_df["cdr3_aa"].dropna()

    #     try:
    #         # Use direct TCREMP call with skip_clustering=True (tested and working method)
    #         embeddings = tcremp(all_chains, chain="H+L", skip_clustering=True)
    #         # If successful, tcremp-run is available and working
    #         assert isinstance(embeddings, torch.Tensor)
    #         assert embeddings.shape[0] == len(all_chains)  # All chains
    #         print(f"PASS: TCREMP H+L embedding successful - {embeddings.shape}")
    #     except (ImportError, RuntimeError) as e:
    #         # Handle both ImportError (command not available) and RuntimeError (TCREMP internal errors)
    #         if "not installed" in str(e):
    #             print("PASS: TCREMP H+L embedding failed as expected (command not available)")
    #         else:
    #             print(f"PASS: TCREMP H+L embedding failed due to internal TCREMP issues: {str(e)[:100]}...")

    # def test_tcremp_HL_embedding_direct_call(self):
    #     """Test TCREMP (alpha-beta/gamma-delta pairs - HL pairs for TCR) using direct call."""
    #     from amulety.tcr_embeddings import tcremp

    #     # For HL pairs, use all CDR3 sequences (TCREMP can handle paired chain format)
    #     all_chains = self.test_airr_tcr_df["cdr3_aa"].dropna()

    #     try:
    #         # Use direct TCREMP call with skip_clustering=True (tested and working method)
    #         embeddings = tcremp(all_chains, chain="HL", skip_clustering=True)
    #         # If successful, tcremp-run is available and working
    #         assert isinstance(embeddings, torch.Tensor)
    #         assert embeddings.shape[0] == len(all_chains)  # Number of sequences
    #         print(f"PASS: TCREMP HL embedding successful - {embeddings.shape}")
    #     except (ImportError, RuntimeError) as e:
    #         # Handle both ImportError (command not available) and RuntimeError (TCREMP internal errors)
    #         if "not installed" in str(e):
    #             print("PASS: TCREMP HL embedding failed as expected (command not available)")
    #         else:
    #             print(f"PASS: TCREMP HL embedding failed due to internal TCREMP issues: {str(e)[:100]}...")

    # def test_tcremp_LH_embedding_direct_call(self):
    #     """Test TCREMP (beta-alpha/delta-gamma pairs - LH pairs for TCR) using direct call."""
    #     from amulety.tcr_embeddings import tcremp

    #     # For LH pairs, use all CDR3 sequences (TCREMP can handle paired chain format)
    #     all_chains = self.test_airr_tcr_df["cdr3_aa"].dropna()

    #     try:
    #         # Use direct TCREMP call with skip_clustering=True (tested and working method)
    #         embeddings = tcremp(all_chains, chain="LH", skip_clustering=True)
    #         # If successful, tcremp-run is available and working
    #         assert isinstance(embeddings, torch.Tensor)
    #         assert embeddings.shape[0] == len(all_chains)  # Number of sequences
    #         print(f"PASS: TCREMP LH embedding successful - {embeddings.shape}")
    #     except (ImportError, RuntimeError) as e:
    #         # Handle both ImportError (command not available) and RuntimeError (TCREMP internal errors)
    #         if "not installed" in str(e):
    #             print("PASS: TCREMP LH embedding failed as expected (command not available)")
    #         else:
    #             print(f"PASS: TCREMP LH embedding failed due to internal TCREMP issues: {str(e)[:100]}...")

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

    # def test_custom_cdr3_column_tcr(self):
    #     """Test that we can pass any column name for CDR3 sequences in TCR data."""
    #     from amulety.utils import get_cdr3_sequence_column

    #     # Create test data with different CDR3 column names that users might have
    #     test_data_standard = pd.DataFrame(
    #         {
    #             "sequence_id": ["TCR_001", "TCR_002"],
    #             "sequence_vdj_aa": ["FULL_VDJ_SEQUENCE_1", "FULL_VDJ_SEQUENCE_2"],
    #             "junction_aa": ["CASSLAPGATNEKLFF", "CAVNTGNQFYF"],  # Standard AIRR column
    #             "cdr3_aa": ["CASSLAPGATNEKLFF", "CAVNTGNQFYF"],  # Standard AIRR column
    #             "v_call": ["TRBV7-9*01", "TRAV8-4*01"],
    #             "duplicate_count": [1, 1],
    #             "locus": ["TRB", "TRA"],
    #             "cell_id": ["cell_1", "cell_1"],
    #         }
    #     )

    #     test_data_custom = pd.DataFrame(
    #         {
    #             "sequence_id": ["TCR_003", "TCR_004"],
    #             "sequence_vdj_aa": ["FULL_VDJ_SEQUENCE_3", "FULL_VDJ_SEQUENCE_4"],
    #             "cdr3_sequence": ["CASSLVGQGAYEQYF", "CAVRDMEYGNKLVF"],  # Custom CDR3 column name
    #             "junction_sequence": ["CASSLVGQGAYEQYF", "CAVRDMEYGNKLVF"],  # Another custom name
    #             "v_call": ["TRBV7-2*01", "TRAV13-1*01"],
    #             "duplicate_count": [1, 1],
    #             "locus": ["TRB", "TRA"],
    #             "cell_id": ["cell_2", "cell_2"],
    #         }
    #     )

    #     # Test standard AIRR columns are detected (priority: junction_aa > cdr3_aa)
    #     result_junction = get_cdr3_sequence_column(test_data_standard, "sequence_vdj_aa")
    #     assert result_junction == "junction_aa", f"Expected 'junction_aa', got '{result_junction}'"

    #     # Test when only cdr3_aa exists
    #     test_data_cdr3_only = test_data_standard.drop(columns=["junction_aa"])
    #     result_cdr3 = get_cdr3_sequence_column(test_data_cdr3_only, "sequence_vdj_aa")
    #     assert result_cdr3 == "cdr3_aa", f"Expected 'cdr3_aa', got '{result_cdr3}'"

    #     # Test fallback to sequence_col when no standard CDR3 columns exist
    #     result_fallback = get_cdr3_sequence_column(test_data_custom, "sequence_vdj_aa")
    #     assert result_fallback == "sequence_vdj_aa", f"Expected 'sequence_vdj_aa', got '{result_fallback}'"

    #     # Test that users can specify custom CDR3 column by using it as sequence_col
    #     # This is the current way users can specify custom CDR3 columns
    #     result_custom_cdr3 = get_cdr3_sequence_column(test_data_custom, "cdr3_sequence")
    #     assert result_custom_cdr3 == "cdr3_sequence", f"Expected 'cdr3_sequence', got '{result_custom_cdr3}'"

    #     result_custom_junction = get_cdr3_sequence_column(test_data_custom, "junction_sequence")
    #     assert (
    #         result_custom_junction == "junction_sequence"
    #     ), f"Expected 'junction_sequence', got '{result_custom_junction}'"

    #     # Test with process_airr function using custom CDR3 column
    #     result_process = process_airr(test_data_custom, "H", sequence_col="cdr3_sequence")
    #     assert len(result_process) == 1  # Should have 1 H chain (TRB)
    #     assert result_process.iloc[0]["cdr3_sequence"] == "CASSLVGQGAYEQYF"

    #     print("PASS: Custom CDR3 column test for TCR data successful")
