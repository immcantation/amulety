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

    def test_esm2_finetuned_custom_model(self):
        """Test fine-tuned ESM2 with custom model name (using base ESM2 as example)."""
        from amulety.amulety import embed_airr

        # Test with base ESM2 model as a fine-tuned model example
        result = embed_airr(
            self.test_airr_sc_df,
            "H",
            "esm2-custom",
            output_type="pickle",
            custom_model_name="facebook/esm2_t33_650M_UR50D",  # Use base model as example
            batch_size=2,
        )
        assert result.shape[1] == 1280  # ESM2 embedding dimension
        assert result.shape[0] == 2  # 2 heavy chains in test data

    def test_immune2vec_embedding(self):
        """Test Immune2Vec embedding (will skip if dependencies not available)."""
        from amulety.amulety import embed_airr

        try:
            # Test Immune2Vec with TCR data
            result = embed_airr(self.test_airr_sc_df, "H", "immune2vec", output_type="pickle", batch_size=2)
            assert result.shape[1] == 100  # Default Immune2Vec embedding dimension
            assert result.shape[0] == 2  # 2 heavy chains in test data
            print("PASS: Immune2Vec test passed")
        except ImportError as e:
            error_msg = str(e)
            print("SKIP: Skipping Immune2Vec test due to missing dependencies")
            print(f"   Error: {error_msg}")

            # Verify that the error message contains installation instructions
            assert "gensim" in error_msg or "immune2vec" in error_msg
            assert "pip install" in error_msg

            # Different error messages for different missing dependencies
            if "gensim" in error_msg and "immune2vec" not in error_msg:
                # Only gensim error
                assert "conda install" in error_msg
                print("   Detected gensim missing error")
            else:
                # Immune2vec error (may also mention gensim)
                assert "git clone" in error_msg or "github" in error_msg
                print("   Detected immune2vec missing error")

            self.skipTest("Immune2Vec dependencies not available - this is expected behavior")
        except Exception as e:
            print(f"FAIL: Immune2Vec test failed with unexpected error: {e}")
            raise

    def test_immune2vec_error_messages(self):
        """Test that Immune2Vec provides helpful error messages when dependencies are missing."""
        import pandas as pd

        from amulety.protein_embeddings import immune2vec

        test_sequences = pd.Series(["CASSLAPGATNEKLFF", "CAVKDSNYQLIW"])

        try:
            result = immune2vec(test_sequences)
            # If this succeeds, dependencies are available
            assert result.shape[1] == 100
            print("PASS: Immune2Vec dependencies are available")
        except ImportError as e:
            error_msg = str(e)
            print(f"PASS: Proper ImportError caught: {error_msg[:100]}...")

            # Verify error message quality - check for either gensim or immune2vec instructions
            assert "gensim" in error_msg or "immune2vec" in error_msg
            assert "pip install" in error_msg

            # Different error messages for different missing dependencies
            if "gensim" in error_msg:
                assert "conda install" in error_msg
                print("PASS: Gensim error message contains proper installation instructions")
            else:
                assert "git clone" in error_msg or "github" in error_msg
                assert "sys.path.append" in error_msg or "PYTHONPATH" in error_msg
                print("PASS: Immune2Vec error message contains proper installation instructions")
        except Exception as e:
            print(f"FAIL: Unexpected error type: {type(e).__name__}: {e}")
            raise
