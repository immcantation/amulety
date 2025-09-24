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
# Also allow users to skip locally by setting SKIP_LARGE_MODELS=true
SKIP_LARGE_MODELS = (
    os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("SKIP_LARGE_MODELS", "").lower() == "true"
)


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

    # esm2 tests
    def test_esm2_mixed_HL_embedding(self):
        """Test esm2 (mixed bulk sc HL)."""
        import warnings

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                embed(input_airr=self.test_airr_mixed_path, chain="HL", model="esm2", output_file_path="HL_test.pt")
                assert os.path.exists("HL_test.pt")
                embeddings = torch.load("HL_test.pt")
                assert embeddings.shape[1] == 1280  # ESM2 embedding dimension
                assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
                os.remove("HL_test.pt")
                os.remove("HL_test_metadata.tsv")
                # Check that protein language model warning was issued
                assert len(w) > 0
                warning_messages = [str(warning.message) for warning in w]
                assert any(
                    "does not have mechanisms to understand paired chain relationships" in msg
                    for msg in warning_messages
                )
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_esm2_mixed_H_plus_L_embedding_tsv(self):
        """Test esm2 (mixed bulk sc H+L)."""
        try:
            embed(self.test_airr_mixed_path, "H+L", "esm2", "H_plus_L_test.tsv")
            assert os.path.exists("H_plus_L_test.tsv")
            embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
            assert embeddings.shape[1] == 1281  # 1280 + id
            assert embeddings.shape[0] == 5  # 2 H chain + 3 L chain
            os.remove("H_plus_L_test.tsv")
            os.remove("H_plus_L_test_metadata.tsv")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_esm2_mixed_H_embedding_tsv(self):
        """Test esm2 (mixed bulk sc H)."""
        try:
            embed(self.test_airr_mixed_path, "H", "esm2", "H_test.tsv")
            assert os.path.exists("H_test.tsv")
            embeddings = pd.read_table("H_test.tsv", delimiter="\t")
            assert embeddings.shape[1] == 1281  # 1280 + id
            assert (
                embeddings.shape[0] == 2
            )  # 2 H chain + 2 L chain (only the most abundant L chain per cell kept for single-cell data)
            os.remove("H_test.tsv")
            os.remove("H_test_metadata.tsv")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_esm2_mixed_H_embedding_residue_level(self):
        """Test esm2 mixed bulk sc H with residue-level embeddings."""
        try:
            embed(self.test_airr_mixed_path, "H", "esm2", "H_residue_test.pt", residue_level=True)
            assert os.path.exists("H_residue_test.pt")
            embeddings = torch.load("H_residue_test.pt")
            assert embeddings.shape[0] == 2  # 2 H chains (TRB chains from 2 TCR cells)
            assert embeddings.shape[1] == 512  # max sequence length (padded)
            assert embeddings.shape[2] == 1280  # ESM2 embedding dimension
            os.remove("H_residue_test.pt")
            os.remove("H_residue_test_metadata.tsv")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    # prott5 tests
    @unittest.skipIf(SKIP_LARGE_MODELS, "Skipping ProtT5 test on GitHub Actions due to disk space limitations")
    def test_prott5_mixed_HL_embedding(self):
        """Test prott5 (mixed bulk sc HL)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            embed(input_airr=self.test_airr_mixed_path, chain="HL", model="prott5", output_file_path="HL_test.pt")
            assert os.path.exists("HL_test.pt")
            embeddings = torch.load("HL_test.pt")
            assert embeddings.shape[1] == 1024  # ProtT5 embedding dimension
            assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
            os.remove("HL_test.pt")
            os.remove("HL_test_metadata.tsv")
            # Check that protein language model warning was issued
            assert len(w) > 0
            warning_messages = [str(warning.message) for warning in w]
            assert any(
                "does not have mechanisms to understand paired chain relationships" in msg for msg in warning_messages
            )

    @unittest.skipIf(SKIP_LARGE_MODELS, "Skipping ProtT5 test on GitHub Actions due to disk space limitations")
    def test_prott5_mixed_H_plus_L_embedding_tsv(self):
        """Test prott5 (mixed bulk sc H+L)."""
        embed(self.test_airr_mixed_path, "H+L", "prott5", "H_plus_L_test.tsv")
        assert os.path.exists("H_plus_L_test.tsv")
        embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1025  # 1024 + id
        assert embeddings.shape[0] == 5  # 2 H chain + 3 L chain
        os.remove("H_plus_L_test.tsv")
        os.remove("H_plus_L_test_metadata.tsv")

    @unittest.skipIf(SKIP_LARGE_MODELS, "Skipping ProtT5 test on GitHub Actions due to disk space limitations")
    def test_prott5_mixed_H_embedding_tsv(self):
        """Test prott5 (mixed bulk sc H)."""
        embed(self.test_airr_mixed_path, "H", "prott5", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1025  # 1024 + id
        assert (
            embeddings.shape[0] == 2
        )  # 2 H chain + 2 L chain (only the most abundant L chain per cell kept for single-cell data)
        os.remove("H_test.tsv")
        os.remove("H_test_metadata.tsv")

    @unittest.skipIf(SKIP_LARGE_MODELS, "Skipping ProtT5 test on GitHub Actions due to disk space limitations")
    def test_prott5_mixed_H_embedding_residue_level(self):
        """Test prott5 mixed bulk sc H with residue-level embeddings."""
        embed(self.test_airr_mixed_path, "H", "prott5", "H_residue_test.pt", residue_level=True)
        assert os.path.exists("H_residue_test.pt")
        embeddings = torch.load("H_residue_test.pt")
        assert embeddings.shape[0] == 2  # 2 H chains (TRB chains from 2 TCR cells)
        assert embeddings.shape[1] == 1024  # max sequence length (padded)
        assert embeddings.shape[2] == 1024  # ProtT5 embedding dimension
        os.remove("H_residue_test.pt")
        os.remove("H_residue_test_metadata.tsv")

    # custom model tests
    def test_custom_mixed_HL_embedding(self):
        """Test custom model (mixed bulk sc HL)."""
        import warnings

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                embed(
                    input_airr=self.test_airr_mixed_path,
                    chain="HL",
                    model="custom",
                    output_file_path="HL_test.pt",
                    model_path="facebook/esm2_t33_650M_UR50D",
                    embedding_dimension=1280,
                    max_length=512,
                )
                assert os.path.exists("HL_test.pt")
                embeddings = torch.load("HL_test.pt")
                assert embeddings.shape[1] == 1280  # Custom ESM2 embedding dimension
                assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
                os.remove("HL_test.pt")
                os.remove("HL_test_metadata.tsv")
                # Check that protein language model warning was issued
                assert len(w) > 0
                warning_messages = [str(warning.message) for warning in w]
                assert any("might not understand paired chain relationships" in msg for msg in warning_messages)
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"Custom model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_custom_mixed_H_plus_L_embedding_tsv(self):
        """Test custom model (mixed bulk sc H+L)."""
        try:
            embed(
                self.test_airr_mixed_path,
                "H+L",
                "custom",
                "H_plus_L_test.tsv",
                model_path="facebook/esm2_t33_650M_UR50D",
                embedding_dimension=1280,
                max_length=512,
            )
            assert os.path.exists("H_plus_L_test.tsv")
            embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
            assert embeddings.shape[1] == 1281  # 1280 + id
            assert embeddings.shape[0] == 5  # 2 H chain + 3 L chain
            os.remove("H_plus_L_test.tsv")
            os.remove("H_plus_L_test_metadata.tsv")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"Custom model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_custom_mixed_H_embedding_tsv(self):
        """Test custom model (mixed bulk sc H)."""
        try:
            embed(
                self.test_airr_mixed_path,
                "H",
                "custom",
                "H_test.tsv",
                model_path="facebook/esm2_t33_650M_UR50D",
                embedding_dimension=1280,
                max_length=512,
            )
            assert os.path.exists("H_test.tsv")
            embeddings = pd.read_table("H_test.tsv", delimiter="\t")
            assert embeddings.shape[1] == 1281  # 1280 + id
            assert (
                embeddings.shape[0] == 2
            )  # 2 H chain + 2 L chain (only the most abundant L chain per cell kept for single-cell data)
            os.remove("H_test.tsv")
            os.remove("H_test_metadata.tsv")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"Custom model loading failed (corrupted cache): {e}")
            else:
                raise
