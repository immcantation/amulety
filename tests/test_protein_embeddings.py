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

    def test_esm2_sc_HL_embedding(self):
        """Test esm2 (single-cell HL with protein language model warning)."""
        import warnings

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                embed(self.test_airr_sc_path, "HL", "esm2", "HL_test.pt")
                assert os.path.exists("HL_test.pt")
                embeddings = torch.load("HL_test.pt")
                assert embeddings.shape[1] == 1280
                assert embeddings.shape[0] == 2
                os.remove("HL_test.pt")
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

    def test_esm2_sc_H_embedding(self):
        """Test esm2 (single-cell H)."""
        try:
            embed(self.test_airr_sc_path, "H", "esm2", "H_test.pt")
            assert os.path.exists("H_test.pt")
            embeddings = torch.load("H_test.pt")
            assert embeddings.shape[1] == 1280
            assert embeddings.shape[0] == 2
            os.remove("H_test.pt")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_esm2_sc_L_embedding(self):
        """Test esm2 (single-cell L)."""
        try:
            embed(self.test_airr_sc_path, "L", "esm2", "L_test.pt")
            assert os.path.exists("L_test.pt")
            embeddings = torch.load("L_test.pt")
            assert embeddings.shape[1] == 1280
            assert embeddings.shape[0] == 2
            os.remove("L_test.pt")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_esm2_sc_LH_embedding(self):
        """Test esm2 (single-cell LH with warnings for both LH order and protein language model)."""
        import warnings

        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                embed(self.test_airr_sc_path, "LH", "esm2", "LH_test.pt")
                assert os.path.exists("LH_test.pt")
                embeddings = torch.load("LH_test.pt")
                assert embeddings.shape[1] == 1280
                assert embeddings.shape[0] == 2
                os.remove("LH_test.pt")
                # Check that both LH order warning and protein language model warning were issued
                assert len(w) >= 2
                warning_messages = [str(warning.message) for warning in w]
                assert any("LH (Light-Heavy) chain order detected" in msg for msg in warning_messages)
                assert any(
                    "does not have mechanisms to understand paired chain relationships" in msg
                    for msg in warning_messages
                )
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_esm2_sc_H_plus_L_embedding(self):
        """Test esm2 (single-cell H+L)."""
        try:
            embed(self.test_airr_sc_path, "H+L", "esm2", "H_plus_L_test.pt")
            assert os.path.exists("H_plus_L_test.pt")
            embeddings = torch.load("H_plus_L_test.pt")
            assert embeddings.shape[1] == 1280
            assert embeddings.shape[0] == 4  # 2 H chains + 2 L chains
            os.remove("H_plus_L_test.pt")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_tcr_esm2_L_embedding(self):
        """Test ESM2 with TCR alpha chains (using unified approach)."""
        try:
            # Use existing esm2 function with TCR chain mapping: alpha -> L
            embed(self.test_airr_tcr_path, "L", "esm2", "tcr_esm2_L_test.pt", batch_size=2)
            assert os.path.exists("tcr_esm2_L_test.pt")
            embeddings = torch.load("tcr_esm2_L_test.pt")
            assert embeddings.shape[1] == 1280  # ESM2 embedding dimension
            assert embeddings.shape[0] == 3  # 3 alpha chains in test data
            os.remove("tcr_esm2_L_test.pt")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    def test_tcr_esm2_HL_embedding(self):
        """Test ESM2 with TCR alpha-beta pairs (using unified approach)."""
        try:
            # Use existing esm2 function with TCR chain mapping: alpha-beta -> HL
            embed(self.test_airr_tcr_path, "HL", "esm2", "tcr_esm2_HL_test.pt", batch_size=2)
            assert os.path.exists("tcr_esm2_HL_test.pt")
            embeddings = torch.load("tcr_esm2_HL_test.pt")
            assert embeddings.shape[1] == 1280  # ESM2 embedding dimension
            assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
            os.remove("tcr_esm2_HL_test.pt")
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"ESM2 model loading failed (corrupted cache): {e}")
            else:
                raise

    @unittest.skipIf(SKIP_LARGE_MODELS, "Skipping ProtT5 test on GitHub Actions due to disk space limitations")
    def test_tcr_prott5_L_embedding(self):
        """Test ProtT5 with TCR alpha chains (using unified L notation)."""
        # Use generic prott5 function with TCR chain mapping: alpha -> L (unified approach)
        embed(self.test_airr_tcr_path, "L", "prott5", "tcr_prott5_L_test.pt", batch_size=2)
        assert os.path.exists("tcr_prott5_L_test.pt")
        embeddings = torch.load("tcr_prott5_L_test.pt")
        # ProtT5: 1024 dim, ESM2 fallback: 1280 dim
        assert embeddings.shape[1] in [1024, 1280]
        assert embeddings.shape[0] == 3  # 3 alpha chains in test data
        os.remove("tcr_prott5_L_test.pt")

    @unittest.skipIf(SKIP_LARGE_MODELS, "Skipping ProtT5 test on GitHub Actions due to disk space limitations")
    def test_tcr_prott5_HL_embedding(self):
        """Test ProtT5 with TCR alpha-beta pairs (using unified HL notation)."""
        # Use generic prott5 function with TCR chain mapping: alpha-beta -> HL (unified approach)
        embed(self.test_airr_tcr_path, "HL", "prott5", "tcr_prott5_HL_test.pt", batch_size=2)
        assert os.path.exists("tcr_prott5_HL_test.pt")
        embeddings = torch.load("tcr_prott5_HL_test.pt")
        assert embeddings.shape[1] == 1024  # ProtT5 embedding dimension
        assert embeddings.shape[0] == 3  # 3 alpha-beta pairs in test data
        os.remove("tcr_prott5_HL_test.pt")

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

    def test_custom_model_with_esm2(self):
        """Test custom model functionality using ESM2 as example."""
        from amulety.amulety import embed_airr

        # Test custom model with ESM2 parameters
        result = embed_airr(
            self.test_airr_sc_df,
            "H",
            "custom",
            output_type="pickle",
            model_path="facebook/esm2_t33_650M_UR50D",  # Use base ESM2 model as custom model example
            embedding_dimension=1280,
            max_length=512,
            batch_size=2,
        )
        assert result.shape[1] == 1280  # ESM2 embedding dimension
        assert result.shape[0] == 2  # 2 heavy chains in test data

    def test_finetuned_esm2_custommodel(self):
        """Test fine-tuned ESM2 model using custommodel function."""
        from amulety.amulety import embed_airr

        # Test fine-tuned ESM2 model: AmelieSchreiber/esm2_t6_8M_UR50D-finetuned-localization
        # This model is based on ESM2-t6 (8M parameters) with 320-dimensional embeddings
        result = embed_airr(
            self.test_airr_sc_df,
            "H",
            "custom",
            output_type="pickle",
            model_path="AmelieSchreiber/esm2_t6_8M_UR50D-finetuned-localization",
            embedding_dimension=320,  # ESM2-t6 uses 320-dim embeddings
            max_length=512,
            batch_size=2,
        )
        assert result.shape[1] == 320  # ESM2-t6 embedding dimension
        assert result.shape[0] == 2  # 2 heavy chains in test data

    def test_auto_receptor_type_validation(self):
        """Test automatic receptor type validation functionality."""
        from amulety.amulety import embed_airr

        try:
            # Test 1: BCR data with BCR model (should work)
            result = embed_airr(
                self.test_airr_sc_df,
                "H",
                "antiberta2",
                batch_size=2,
            )
            assert result.shape[1] == 1024  # AntiBERTa2 embedding dimension
            assert result.shape[0] == 2  # 2 heavy chains in test data

            # Test 2: TCR data with TCR model (should work)
            result = embed_airr(
                self.test_airr_tcr_df,
                "H",
                "tcr-bert",
                batch_size=2,
            )
            assert result.shape[1] == 768  # TCR-BERT embedding dimension
            assert result.shape[0] == 3  # 3 beta chains in TCR test data

            # Test 3: BCR data with protein model (should work)
            result = embed_airr(
                self.test_airr_sc_df,
                "H",
                "esm2",
                batch_size=2,
            )
            assert result.shape[1] == 1280  # ESM2 embedding dimension
            assert result.shape[0] == 2  # 2 heavy chains in test data

            # Test 4: TCR data with BCR model (should fail with clear error)
            with self.assertRaises(ValueError) as context:
                embed_airr(
                    self.test_airr_tcr_df,
                    "H",
                    "antiberta2",
                    batch_size=2,
                )
            self.assertIn("is a BCR-specific model", str(context.exception))
            self.assertIn("no BCR data", str(context.exception))

            # Test 5: BCR data with TCR model (should fail with clear error)
            with self.assertRaises(ValueError) as context:
                embed_airr(
                    self.test_airr_sc_df,
                    "H",
                    "tcr-bert",
                    batch_size=2,
                )
            self.assertIn("is a TCR-specific model", str(context.exception))
            self.assertIn("no TCR data", str(context.exception))
        except Exception as e:
            if "SafetensorError" in str(e) or "InvalidHeaderDeserialization" in str(e):
                self.skipTest(f"Model loading failed (corrupted cache): {e}")
            else:
                raise

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

            # Verify error message quality - check for installation instructions
            assert (
                "Immune2Vec package is required but not installed" in error_msg
                or "Gensim library is required" in error_msg
            )
            assert "pip install" in error_msg
            assert "bitbucket.org/yaarilab/immune2vec_model" in error_msg

            # Different error messages for different missing dependencies
            if "Gensim library is required" in error_msg:
                assert "pip install gensim" in error_msg
                assert "git clone" in error_msg
                print("PASS: Gensim error message contains proper installation instructions")
            else:
                assert "git clone" in error_msg
                assert "sys.path.append" in error_msg
                print("PASS: Immune2Vec error message contains proper installation instructions")
        except Exception as e:
            print(f"FAIL: Unexpected error type: {type(e).__name__}: {e}")
            raise
