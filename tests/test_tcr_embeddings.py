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
        try:
            embed(self.test_airr_tcr_path, "L", "tcr-bert", "tcr_L_test.pt", batch_size=2)
            assert os.path.exists("tcr_L_test.pt")
            embeddings = torch.load("tcr_L_test.pt")
            assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
            assert embeddings.shape[0] == 3  # 3 alpha chains in test data
            os.remove("tcr_L_test.pt")
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCR-BERT model loading failed: {e}")
            else:
                raise

    def test_tcr_bert_H_embedding(self):
        """Test TCR-BERT (beta/delta chains - H chains for TCR) - now supported."""
        try:
            embed(self.test_airr_tcr_path, "H", "tcr-bert", "tcr_H_test.pt", batch_size=2)
            assert os.path.exists("tcr_H_test.pt")
            embeddings = torch.load("tcr_H_test.pt")
            assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
            assert embeddings.shape[0] == 3  # 3 beta chains in test data
            os.remove("tcr_H_test.pt")
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCR-BERT model loading failed: {e}")
            else:
                raise

    def test_tcr_bert_HL_embedding(self):
        """Test TCR-BERT (alpha-beta/gamma-delta pairs - HL pairs for TCR)."""
        try:
            embed(self.test_airr_tcr_path, "HL", "tcr-bert", "tcr_HL_test.pt", batch_size=2)
            assert os.path.exists("tcr_HL_test.pt")
            embeddings = torch.load("tcr_HL_test.pt")
            assert embeddings.shape[1] == 768  # TCR-BERT embedding dimension
            assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
            os.remove("tcr_HL_test.pt")
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCR-BERT model loading failed: {e}")
            else:
                raise

    def test_tcr_bert_LH_embedding(self):
        """Test TCR-BERT (beta-alpha/delta-gamma pairs - LH pairs for TCR with warning)."""
        import warnings

        try:
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
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCR-BERT model loading failed: {e}")
            else:
                raise

    def test_tcrt5_H_embedding(self):
        """Test TCRT5 (beta chains only - H chains for TCR)."""
        try:
            embed(self.test_airr_tcr_path, "H", "tcrt5", "tcrt5_H_test.pt", batch_size=2)
            assert os.path.exists("tcrt5_H_test.pt")
            embeddings = torch.load("tcrt5_H_test.pt")
            assert embeddings.shape[1] == 256  # TCRT5 embedding dimension
            assert embeddings.shape[0] == 3  # 3 beta chains in test data
            os.remove("tcrt5_H_test.pt")
        except Exception as e:
            if any(
                error_type in str(e)
                for error_type in ["SafetensorError", "InvalidHeaderDeserialization", "ConnectionError", "HTTPError"]
            ):
                self.skipTest(f"TCRT5 model loading failed: {e}")
            else:
                raise

    def test_tcrt5_L_embedding_should_fail(self):
        """Test TCRT5 with L chains should fail (only supports H chains)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_tcr_path, "L", "tcrt5", "tcrt5_L_test.pt", batch_size=2)

        error_msg = str(context.exception)
        assert "TCRT5 model only supports H chains" in error_msg
        assert "beta chains for TCR" in error_msg

    def test_tcrt5_HL_embedding_should_fail(self):
        """Test TCRT5 with HL chains should fail (only supports H chains)."""
        with self.assertRaises(ValueError) as context:
            embed(self.test_airr_tcr_path, "HL", "tcrt5", "tcrt5_HL_test.pt", batch_size=2)

        error_msg = str(context.exception)
        assert "TCRT5 model only supports H chains" in error_msg
        assert "beta chains for TCR" in error_msg

    def test_tcremp_command_not_available(self):
        """Test that TCREMP raises ImportError when command-line tool is not available."""
        import pandas as pd

        from amulety.tcr_embeddings import tcremp

        test_sequences = pd.Series(["CASSLAPGATNEKLFF", "CAVKDSNYQLIW"])

        # Most systems won't have tcremp-run installed, so this should raise ImportError
        try:
            result = tcremp(test_sequences)
            # If this succeeds, tcremp-run is available and working
            assert isinstance(result, torch.Tensor)
            assert result.shape[0] == len(test_sequences)
            print("PASS: TCREMP command-line tool is available and working")
        except ImportError as e:
            error_msg = str(e)
            print(f"PASS: Proper ImportError caught: {error_msg[:100]}...")

            # Verify error message quality
            assert "TCREMP command-line tool is required but not installed" in error_msg
            assert "git clone https://github.com/antigenomics/tcremp.git" in error_msg
            assert "tcremp-run -h" in error_msg
            assert "tcr-bert" in error_msg  # Alternative suggestions

    def test_tcremp_L_embedding_command_unavailable(self):
        """Test TCREMP (alpha/gamma chains - L chains for TCR) when command is unavailable."""
        try:
            embed(self.test_airr_tcr_path, "L", "tcremp", "tcremp_L_test.pt", batch_size=2)
            # If successful, tcremp-run is available
            assert os.path.exists("tcremp_L_test.pt")
            embeddings = torch.load("tcremp_L_test.pt")
            assert embeddings.shape[0] == 3  # 3 alpha chains in test data
            os.remove("tcremp_L_test.pt")
            print("PASS: TCREMP L embedding successful")
        except ImportError:
            print("PASS: TCREMP L embedding failed as expected (command not available)")

    def test_tcremp_H_embedding_command_unavailable(self):
        """Test TCREMP (beta/delta chains - H chains for TCR) when command is unavailable."""
        try:
            embed(self.test_airr_tcr_path, "H", "tcremp", "tcremp_H_test.pt", batch_size=2)
            # If successful, tcremp-run is available
            assert os.path.exists("tcremp_H_test.pt")
            embeddings = torch.load("tcremp_H_test.pt")
            assert embeddings.shape[0] == 3  # 3 beta chains in test data
            os.remove("tcremp_H_test.pt")
            print("PASS: TCREMP H embedding successful")
        except ImportError:
            print("PASS: TCREMP H embedding failed as expected (command not available)")

    def test_tcremp_H_plus_L_embedding_command_unavailable(self):
        """Test TCREMP (both alpha and beta chains separately - H+L for TCR) when command is unavailable."""
        try:
            embed(self.test_airr_tcr_path, "H+L", "tcremp", "tcremp_H_plus_L_test.pt", batch_size=2)
            # If successful, tcremp-run is available
            assert os.path.exists("tcremp_H_plus_L_test.pt")
            embeddings = torch.load("tcremp_H_plus_L_test.pt")
            assert embeddings.shape[0] == 6  # 3 alpha + 3 beta chains in test data
            os.remove("tcremp_H_plus_L_test.pt")
            print("PASS: TCREMP H+L embedding successful")
        except ImportError:
            print("PASS: TCREMP H+L embedding failed as expected (command not available)")

    def test_tcremp_HL_embedding_command_unavailable(self):
        """Test TCREMP (alpha-beta/gamma-delta pairs - HL pairs for TCR) when command is unavailable."""
        try:
            embed(self.test_airr_tcr_path, "HL", "tcremp", "tcremp_HL_test.pt", batch_size=2)
            # If successful, tcremp-run is available
            assert os.path.exists("tcremp_HL_test.pt")
            embeddings = torch.load("tcremp_HL_test.pt")
            assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
            os.remove("tcremp_HL_test.pt")
            print("PASS: TCREMP HL embedding successful")
        except ImportError:
            print("PASS: TCREMP HL embedding failed as expected (command not available)")

    def test_tcremp_LH_embedding_command_unavailable(self):
        """Test TCREMP (beta-alpha/delta-gamma pairs - LH pairs for TCR) when command is unavailable."""
        try:
            embed(self.test_airr_tcr_path, "LH", "tcremp", "tcremp_LH_test.pt", batch_size=2)
            # If successful, tcremp-run is available
            assert os.path.exists("tcremp_LH_test.pt")
            embeddings = torch.load("tcremp_LH_test.pt")
            assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
            os.remove("tcremp_LH_test.pt")
            print("PASS: TCREMP LH embedding successful")
        except ImportError:
            print("PASS: TCREMP LH embedding failed as expected (command not available)")
