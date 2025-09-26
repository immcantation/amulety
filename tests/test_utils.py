#!/usr/bin/env python

"""Tests for `amulety` package.
Tests can be run with the command:
python -m unittest test_amulety.py
"""

import os
import unittest

import pandas as pd
import pytest

from amulety.amulety import translate_igblast
from amulety.utils import concatenate_heavylight, process_airr


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
        self.test_bulk_df = pd.read_table(self.test_bulk_path, delimiter="\t", header=0)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_bulk_data_columns(self):
        """Test that bulk data processing returns correct columns."""
        _, result = process_airr(self.test_bulk_df, "H")

        # Verify sequence_id column is preserved
        assert "BCR_bulk_001" in result["sequence_id"].values
        assert "BCR_bulk_002" in result["sequence_id"].values
        assert "TCR_bulk_001" in result["sequence_id"].values
        assert "TCR_bulk_002" in result["sequence_id"].values

    def test_bulk_data_invalid_chain_types(self):
        """Test that bulk data rejects paired chain types (HL, LH) but allows H+L."""
        # Test that HL is not allowed for bulk data
        with pytest.raises(ValueError, match="invalid for bulk mode"):
            process_airr(self.test_bulk_df, "HL")

        # Test that LH is not allowed for bulk data
        with pytest.raises(ValueError, match="invalid for bulk mode"):
            process_airr(self.test_bulk_df, "LH")

        # Test that H+L is now allowed for bulk data (separate heavy and light chains)
        try:
            _, result = process_airr(self.test_bulk_df, "H+L")
            assert result.shape[0] > 0  # Should return some data
            assert "chain" in result.columns  # Should have chain column
        except Exception as e:
            pytest.fail(f"H+L should be allowed for bulk data, but got error: {e}")

    def test_bulk_data_processing_h_chain(self):
        """Test bulk data processing for H chains (heavy chains)."""
        # Test BCR H chain processing
        _, result_bcr_h = process_airr(self.test_bulk_df, "H", receptor_type="BCR")
        assert result_bcr_h.shape[0] == 2  # 2 BCR heavy chains
        assert all(result_bcr_h["v_call"].str.contains("IGH"))  # BCR heavy chain signatures

        # Test TCR H chain processing (beta chains)
        _, result_tcr_h = process_airr(self.test_bulk_df, "H", receptor_type="TCR")
        assert result_tcr_h.shape[0] == 2  # 2 TCR beta chains
        # The sequence column should now be consistently named as sequence_vdj_aa
        assert all(result_tcr_h["v_call"].str.contains("TRB"))  # TCR beta chain signatures

        # Test unified processing (all H chains)
        _, result_all_h = process_airr(self.test_bulk_df, "H", receptor_type="all")
        assert result_all_h.shape[0] == 4  # 2 BCR + 2 TCR heavy chains

    def test_bulk_data_processing_l_chain(self):
        """Test bulk data processing for L chains (light chains)."""
        # Test BCR L chain processing
        _, result_bcr_l = process_airr(self.test_bulk_df, "L", receptor_type="BCR")
        assert result_bcr_l.shape[0] == 2  # 2 BCR light chains
        assert all(result_bcr_l["v_call"].str.contains("IGL|IGK"))  # IGL signature

        # Test TCR L chain processing (alpha chains)
        _, result_tcr_l = process_airr(self.test_bulk_df, "L", receptor_type="TCR")
        assert result_tcr_l.shape[0] == 2  # 2 TCR alpha chains
        # The sequence column should now be consistently named as sequence_vdj_aa
        assert all(result_tcr_l["v_call"].str.contains("TRA"))  # TCR alpha chain signatures

        # Test unified processing (all L chains)
        _, result_all_l = process_airr(self.test_bulk_df, "L", receptor_type="all")
        assert result_all_l.shape[0] == 4  # 2 BCR + 2 TCR light chains

    def test_custom_selection_column(self):
        """Test concatenate_heavylight with custom selection column."""
        # Create test data with custom selection column
        test_data = pd.DataFrame(
            {
                "cell_id": ["cell1", "cell1", "cell1", "cell2", "cell2"],
                "locus": ["IGH", "IGL", "IGK", "IGH", "IGL"],
                "sequence_vdj_aa": ["HEAVY1", "LIGHT1", "LIGHT2", "HEAVY2", "LIGHT3"],
                "duplicate_count": [10, 5, 8, 15, 12],
                "custom_score": [100, 50, 90, 200, 150],  # Custom numeric column
            }
        )

        # Add chain mapping
        test_data["chain"] = test_data["locus"].apply(lambda x: "H" if x == "IGH" else "L")

        # Test with default duplicate_count column
        result_default = concatenate_heavylight(test_data, "sequence_vdj_aa", "cell_id")
        # Should select LIGHT2 for cell1 (duplicate_count=8 > 5) and LIGHT3 for cell2
        assert "HEAVY1<cls><cls>LIGHT2" in result_default["sequence_vdj_aa"].values
        assert "HEAVY2<cls><cls>LIGHT3" in result_default["sequence_vdj_aa"].values

        # Test with custom selection column
        result_custom = concatenate_heavylight(test_data, "sequence_vdj_aa", "cell_id", "custom_score")
        # Should select LIGHT2 for cell1 (custom_score=90 > 50) and LIGHT3 for cell2
        assert "HEAVY1<cls><cls>LIGHT2" in result_custom["sequence_vdj_aa"].values
        assert "HEAVY2<cls><cls>LIGHT3" in result_custom["sequence_vdj_aa"].values

        # Test error when selection column doesn't exist
        with pytest.raises(ValueError, match="Column\\(s\\) \\['nonexistent'\\] is/are not present"):
            concatenate_heavylight(test_data, "sequence_vdj_aa", "cell_id", "nonexistent")

        # Test error when selection column is not numeric
        test_data_non_numeric = test_data.copy()
        test_data_non_numeric["text_col"] = ["A", "B", "C", "D", "E"]
        with pytest.raises(ValueError, match="Selection column 'text_col' must be numeric"):
            concatenate_heavylight(test_data_non_numeric, "sequence_vdj_aa", "cell_id", "text_col")

    def test_mixed_bcr_tcr_data(self):
        """Test processing mixed BCR and TCR data with different receptor_type settings."""

        # Test with receptor_type="all" (should work and include all sequences)
        _, result_all = process_airr(self.test_mixed_df, "H", receptor_type="all")
        assert result_all.shape[0] == 6  # 3 IGH + 3 TRB chains

        # Test with receptor_type="BCR" (should warn and filter out TCR chains)
        _, result_bcr = process_airr(self.test_mixed_df, "H", receptor_type="BCR")
        assert result_bcr.shape[0] == 3  # Only IGH chains

        # Test with receptor_type="TCR" (should warn and filter out BCR chains)
        _, result_tcr = process_airr(self.test_mixed_df, "H", receptor_type="TCR")
        assert result_tcr.shape[0] == 3  # Only TRB chains

        # Test light chains
        _, result_all_light = process_airr(self.test_mixed_df, "L", receptor_type="all")
        assert result_all_light.shape[0] == 6  # 2 IGL + 1 IGK + 3 TRA chains

        # Test paired chains with receptor_type="all"
        _, result_pairs = process_airr(self.test_mixed_df, "HL", receptor_type="all")
        assert result_pairs.shape[0] == 6  # 3 BCR pairs + 3 TCR pairs

    def test_mixed_data_with_unified_models_data_processing(self):
        """Test that unified models can process mixed BCR+TCR data (data processing only)."""

        # Test that process_airr works correctly with mixed data for unified models
        # This tests the data processing pipeline without running expensive model inference

        # Test ProtT5 data processing (uses receptor_type="all")
        _, result_h = process_airr(self.test_mixed_df, "H", receptor_type="all")
        assert result_h.shape[0] == 6  # 3 BCR + 3 TCR heavy chains

        _, result_l = process_airr(self.test_mixed_df, "L", receptor_type="all")
        assert result_l.shape[0] == 6  # 2 IGL + 1 IGK + 3 TRA light chains

        _, result_hl = process_airr(self.test_mixed_df, "HL", receptor_type="all")
        assert result_hl.shape[0] == 6  # 3 BCR + 3 TCR pairs

        # Verify that sequences contain both BCR and TCR data
        # The sequence column should now be consistently named as sequence_vdj_aa
        assert any("EVQL" in seq for seq in result_h["sequence_vdj_aa"])  # BCR signature
        assert any("CASS" in seq for seq in result_h["sequence_vdj_aa"])  # TCR signature

    def test_receptor_type_validation(self):
        """Test receptor type validation functionality."""

        # Test BCR file with correct receptor_type
        _, result = process_airr(self.test_airr_sc_df, "H", receptor_type="BCR")
        assert result.shape[0] == 2

        # Test TCR file with correct receptor_type
        _, result = process_airr(self.test_airr_tcr_df, "H", receptor_type="TCR")
        assert result.shape[0] == 3

        # Test BCR file with wrong receptor_type (should fail)
        with pytest.raises(ValueError, match="No TCR chains.*found in data"):
            process_airr(self.test_airr_sc_df, "H", receptor_type="TCR")

        # Test TCR file with wrong receptor_type (should fail)
        with pytest.raises(ValueError, match="No BCR chains.*found in data"):
            process_airr(self.test_airr_tcr_df, "H", receptor_type="BCR")

        # Test both files with receptor_type="all" (should work)
        _, result_bcr = process_airr(self.test_airr_sc_df, "H", receptor_type="all")
        _, result_tcr = process_airr(self.test_airr_tcr_df, "H", receptor_type="all")
        assert result_bcr.shape[0] == 2
        assert result_tcr.shape[0] == 3

        # Test invalid receptor_type
        with pytest.raises(ValueError, match="receptor_type must be one of"):
            process_airr(self.test_airr_sc_df, "H", receptor_type="invalid")

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

    def test_tcr_backward_compatibility(self):
        """Test that adding TCR support doesn't break BCR functionality."""

        # Test with existing BCR data
        _, result_bcr_h = process_airr(self.test_airr_sc_df, "H", receptor_type="BCR")
        _, result_bcr_l = process_airr(self.test_airr_sc_df, "L", receptor_type="BCR")
        _, result_bcr_hl = process_airr(self.test_airr_sc_df, "HL", receptor_type="BCR")

        # Should still work as before
        assert result_bcr_h.shape[0] == 2  # 2 heavy chains
        assert result_bcr_l.shape[0] == 2  # 2 light chains
        assert result_bcr_hl.shape[0] == 2  # 2 heavy-light pairs

    def test_tcr_concatenation_order(self):
        """Test that TCR concatenation follows Beta+Alpha order."""

        # Get concatenated pairs
        _, result = process_airr(self.test_airr_tcr_df, "HL", receptor_type="TCR")

        # Read original data to verify order
        data = pd.read_table(self.test_airr_tcr_path)

        # Check first pair
        if result.shape[0] > 0:
            # The sequence column should now be consistently named as sequence_vdj_aa
            seq = result.iloc[0]["sequence_vdj_aa"]
            assert "<cls><cls>" in seq

            parts = seq.split("<cls><cls>")
            # First part should be beta (TRB), second part should be alpha (TRA)
            # Use cdr3_aa from original data since that's what gets concatenated for TCR
            beta_seq = data[data.locus == "TRB"].iloc[0]["cdr3_aa"]
            alpha_seq = data[data.locus == "TRA"].iloc[0]["cdr3_aa"]

            assert parts[0] == beta_seq  # Beta first
            assert parts[1] == alpha_seq  # Alpha second

    def test_tcr_unified_chain_mapping(self):
        """Test that TCR chains are correctly mapped to BCR chain system."""

        # Test alpha chains (A -> L)
        _, result_alpha = process_airr(self.test_airr_tcr_df, "L", receptor_type="TCR")
        assert result_alpha.shape[0] == 3  # 3 alpha chains

        # Test beta chains (B -> H)
        _, result_beta = process_airr(self.test_airr_tcr_df, "H", receptor_type="TCR")
        assert result_beta.shape[0] == 3  # 3 beta chains

        # Test alpha-beta pairs (AB -> HL)
        _, result_pairs = process_airr(self.test_airr_tcr_df, "HL", receptor_type="TCR")
        assert result_pairs.shape[0] == 3  # 3 alpha-beta pairs
