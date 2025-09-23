"""
Test chain validation functionality for AMULETY embedding models.
Tests the new chain classification system with proper error handling and warnings.
"""

import unittest
import warnings

import pandas as pd

from amulety.amulety import embed_airr


class TestChainValidation(unittest.TestCase):
    """Test chain validation for different model types."""

    def setUp(self):
        """Set up test data."""
        # Create test BCR data
        self.bcr_data = pd.DataFrame(
            {
                "sequence_id": ["bcr1", "bcr2", "bcr3", "bcr4"],
                "cell_id": ["cell1", "cell1", "cell2", "cell2"],
                "sequence_vdj_aa": [
                    "EVQLVESGGGLVQPGGSLRLSCAASGFTFS",
                    "DIQMTQSPSSLSASVGDRVTITC",
                    "EVQLVESGGGLVQPGGSLRLSCAASGFTFS",
                    "DIQMTQSPSSLSASVGDRVTITC",
                ],
                "v_call": ["IGHV1-69*01", "IGLV2-14*01", "IGHV1-69*01", "IGLV2-14*01"],
                "duplicate_count": [10, 8, 15, 12],
            }
        )

        # Create test TCR data
        self.tcr_data = pd.DataFrame(
            {
                "sequence_id": ["tcr1", "tcr2", "tcr3", "tcr4"],
                "cell_id": ["cell1", "cell1", "cell2", "cell2"],
                "sequence_vdj_aa": ["CASSLAPGATNEKLFF", "CAASRDDKIIF", "CASSLAPGATNEKLFF", "CAASRDDKIIF"],
                "v_call": ["TRBV19*01", "TRAV1-2*01", "TRBV19*01", "TRAV1-2*01"],
                "duplicate_count": [15, 12, 18, 10],
            }
        )

    def test_paired_only_models_accept_hl_lh(self):
        """Test that paired-only models accept HL and LH chains only."""
        paired_only_models = ["balm-paired"]
        paired_chains = ["HL", "LH"]

        for model in paired_only_models:
            data = self.bcr_data if model == "balm-paired" else self.tcr_data

            for chain in paired_chains:
                try:
                    _, _ = embed_airr(data, chain, model, output_type="pickle")
                    # For BALM-paired, result might be None due to download failure, but no exception should be raised
                    # The important thing is that the chain validation passes (no ValueError)
                    print(f"{model} accepts {chain} chains (validation passed)")
                except ValueError as e:
                    # This should not happen - paired-only models should accept HL/LH
                    self.fail(f"{model} should accept {chain} chains, but got validation error: {e}")

    def test_paired_only_models_reject_individual_chains(self):
        """Test that paired-only models reject individual chains."""
        paired_only_models = ["balm-paired"]
        individual_chains = ["H", "L", "H+L"]

        for model in paired_only_models:
            data = self.bcr_data if model == "balm-paired" else self.tcr_data

            for chain in individual_chains:
                with self.assertWarns(UserWarning) as context:
                    embed_airr(data, chain, model, output_type="pickle")

                warning_msg = str(context.warning)
                self.assertIn("was trained on paired chains", warning_msg)
                self.assertIn("--chain HL", warning_msg)

    def test_flexible_paired_models_accept_all_chains(self):
        """Test that flexible paired models accept all chain types."""
        flexible_paired_models = ["tcr-bert"]  # Added tcremp - supports all chain types
        all_chains = ["H", "L", "HL", "LH", "H+L"]

        for model in flexible_paired_models:
            data = self.tcr_data  # Both are TCR models

            for chain in all_chains:
                result = embed_airr(data, chain, model, output_type="pickle")
                self.assertIsNotNone(result)

    def test_h_chain_only_models_accept_h_only(self):
        """Test that H-chain-only models accept only H chains."""
        h_chain_only_models = ["tcrt5"]  # TCRT5 only supports H chains (beta chains for TCR)

        for model in h_chain_only_models:
            data = self.tcr_data  # TCR data for TCR models

            # Test that H chain is accepted
            try:
                result = embed_airr(data, "H", model, output_type="pickle")
                self.assertIsNotNone(result)
            except Exception as e:
                self.fail(f"{model} should accept H chains, but got error: {e}")

    def test_h_chain_only_models_warn_for_other_chains(self):
        """Test that H-chain-only models warn for non-H chains but continue processing."""
        h_chain_only_models = ["tcrt5"]  # TCRT5 only supports H chains (beta chains for TCR)
        invalid_chains = ["L", "HL", "LH", "H+L"]

        for model in h_chain_only_models:
            data = self.tcr_data  # TCR data for TCR models

            for chain in invalid_chains:
                with self.assertWarns(UserWarning) as context:
                    embed_airr(data, chain, model, output_type="pickle")

                warning_msg = str(context.warning)
                self.assertIn("TCRT5 model was trained on", warning_msg)

    def test_protein_language_models_warn_for_paired_chains(self):
        """Test that protein language models warn for paired chains."""
        protein_models = ["esm2", "prott5", "immune2vec"]
        paired_chains = ["HL", "LH"]

        for model in protein_models:
            for chain in paired_chains:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    try:
                        result = embed_airr(self.bcr_data, chain, model, output_type="pickle")
                        self.assertIsNotNone(result)
                        # Check that protein language model warning was issued
                        warning_messages = [str(warning.message) for warning in w]
                        self.assertTrue(
                            any(
                                "does not have mechanisms to understand paired chain relationships" in msg
                                for msg in warning_messages
                            )
                        )
                    except ImportError as e:
                        # Skip models that require additional dependencies not installed in test environment
                        print(f"(warning) Skipping {model} with {chain} chains - missing dependencies: {e}")
                        continue
                    except Exception as e:
                        self.fail(f"{model} should accept {chain} chains with warning, but got error: {e}")

    def test_individual_chain_models_accept_h_l_h_plus_l(self):
        """Test that individual chain models accept H, L, and H+L chains."""
        individual_models = [
            "ablang",
            "antiberty",
            "antiberta2",
        ]  # Removed tcremp - supports paired chains, removed trex - R language package
        valid_chains = ["H", "L", "H+L"]

        for model in individual_models:
            data = self.bcr_data if model in ["ablang", "antiberty", "antiberta2"] else self.tcr_data

            for chain in valid_chains:
                try:
                    result = embed_airr(data, chain, model, output_type="pickle")
                    self.assertIsNotNone(result)
                except Exception as e:
                    self.fail(f"{model} should accept {chain} chains, but got error: {e}")

    def test_individual_chain_models_reject_paired_chains(self):
        """Test that individual chain models reject paired chains."""
        individual_models = [
            "ablang",
            "antiberty",
            "antiberta2",
        ]  # Removed tcremp - supports paired chains, removed trex - R language package
        paired_chains = ["HL", "LH"]

        for model in individual_models:
            data = self.bcr_data if model in ["ablang", "antiberty", "antiberta2"] else self.tcr_data

            for chain in paired_chains:
                with self.assertWarns(UserWarning) as context:
                    embed_airr(data, chain, model, output_type="pickle")

                warning_msg = str(context.warning)
                self.assertIn("was trained on individual chains only", warning_msg)
                self.assertIn("--chain H", warning_msg)
                self.assertIn("--chain L", warning_msg)

    def test_protein_language_models_accept_individual_chains(self):
        """Test that protein language models accept individual chains without warning."""
        protein_models = ["esm2", "prott5", "immune2vec"]
        individual_chains = ["H", "L", "H+L"]

        for model in protein_models:
            for chain in individual_chains:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    try:
                        result = embed_airr(self.bcr_data, chain, model, output_type="pickle")
                        self.assertIsNotNone(result)
                        # Should not have protein language model warnings for individual chains
                        warning_messages = [str(warning.message) for warning in w]
                        self.assertFalse(
                            any(
                                "does not have mechanisms to understand paired chain relationships" in msg
                                for msg in warning_messages
                            )
                        )
                    except ImportError as e:
                        # Skip models that require additional dependencies not installed in test environment
                        print(f"(warning) Skipping {model} with {chain} chains - missing dependencies: {e}")
                        continue
                    except Exception as e:
                        self.fail(f"{model} should accept {chain} chains without warning, but got error: {e}")

    def test_lh_warning_message_content(self):
        """Test that LH warning message is correctly formatted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                embed_airr(self.bcr_data, "LH", "esm2", output_type="pickle")
                # Check that LH warning was issued with correct content
                self.assertTrue(len(w) > 0)
                lh_warnings = [
                    warning for warning in w if "LH (Light-Heavy) chain order detected" in str(warning.message)
                ]
                self.assertTrue(len(lh_warnings) > 0)

                warning_msg = str(lh_warnings[0].message)
                self.assertIn("Most paired models are trained on HL (Heavy-Light) order", warning_msg)
                self.assertIn("may result in reduced accuracy", warning_msg)
                self.assertIn("Consider using --chain_mode HL", warning_msg)
            except Exception as e:
                self.fail(f"ESM2 should accept LH chains with warning, but got error: {e}")

    def test_invalid_chain_parameters(self):
        """Test that invalid chain parameters are rejected."""
        invalid_chains = ["X", "HH", "LL", "ABC", "H+H", "L+L"]

        for chain in invalid_chains:
            with self.assertRaises(ValueError) as context:
                embed_airr(self.bcr_data, chain, "esm2", output_type="pickle")

            error_msg = str(context.exception)
            self.assertIn("must be one of", error_msg)
            self.assertIn("['H', 'L', 'HL', 'LH', 'H+L']", error_msg)

    def test_data_chain_parameter_validation(self):
        """Test that data and chain parameter consistency is validated."""
        # Test H chain parameter with only light chain data
        light_only_data = pd.DataFrame(
            {
                "sequence_id": ["seq1"],
                "cell_id": ["cell1"],
                "sequence_vdj_aa": ["DIQMTQSPSSLSASVGDRVTITC"],
                "v_call": ["IGLV2-14*01"],  # Light chain only
                "duplicate_count": [10],
            }
        )

        with self.assertRaises(ValueError) as context:
            embed_airr(light_only_data, "H", "esm2", output_type="pickle")

        error_msg = str(context.exception)
        self.assertIn("Chain parameter 'H' requires heavy chain data", error_msg)
        self.assertIn("no heavy chain loci found", error_msg)
        self.assertIn("Use --chain L", error_msg)

        # Test L chain parameter with only heavy chain data
        heavy_only_data = pd.DataFrame(
            {
                "sequence_id": ["seq1"],
                "cell_id": ["cell1"],
                "sequence_vdj_aa": ["EVQLVESGGGLVQPGGSLRLSCAASGFTFS"],
                "v_call": ["IGHV1-69*01"],  # Heavy chain only
                "duplicate_count": [10],
            }
        )

        with self.assertRaises(ValueError) as context:
            embed_airr(heavy_only_data, "L", "esm2", output_type="pickle")

        error_msg = str(context.exception)
        self.assertIn("Chain parameter 'L' requires light chain data", error_msg)
        self.assertIn("no light chain loci found", error_msg)
        self.assertIn("Use --chain H", error_msg)

        # Test HL chain parameter with only heavy chain data
        with self.assertRaises(ValueError) as context:
            embed_airr(heavy_only_data, "HL", "esm2", output_type="pickle")

        error_msg = str(context.exception)
        self.assertIn("Chain parameter 'HL' requires light chain data", error_msg)
        self.assertIn("no light chain loci found", error_msg)
        self.assertIn("Use --chain H", error_msg)

    def test_chain_validation_error_messages(self):
        """Test that error messages are informative and helpful."""
        # Test individual chain model with paired chain
        with self.assertWarns(UserWarning) as cm:
            embed_airr(self.bcr_data, "HL", "antiberty", output_type="pickle")
        msg = str(cm.warning)
        self.assertIn("was trained on individual chains only", msg)
        self.assertIn("--chain H", msg)
        self.assertIn("--chain L", msg)


if __name__ == "__main__":
    unittest.main()
