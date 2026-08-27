#!/usr/bin/env python

"""Tests for `amulety` package.
Tests can be run with the command:
python -m unittest test_amulety.py
"""

import os
import sys
import unittest
from importlib.util import find_spec

import numpy as np
import pandas as pd
import torch

from amulety.amulety import embed, embed_airr
from amulety.bcr_embeddings import BCRDistEncoder, setup_bcr_cdr_strings, vec_bcrdist

HAS_PWSEQDIST = find_spec("pwseqdist") is not None


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

        self.test_vec_bcrdist_path = os.path.join(
            self.this_dir,
            "translated_cdr",
            "AIRR_rearrangement_single-cell_testtranslation_translated.tsv",
        )
        self.vec_bcrdist_df = pd.read_table(self.test_vec_bcrdist_path, delimiter="\t", header=0)

        # Bulk test data (no cell_id column)
        self.test_bulk = "AIRR_rearrangement_bulk_test.tsv"
        self.test_bulk_path = os.path.join(self.this_dir, self.test_bulk)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def _bcrdist_reference_data(self):
        """Use unique observed CDR triplets from the translated AIRR fixture."""
        return self.vec_bcrdist_df.drop_duplicates(["cdr1_aa", "cdr2_aa", "cdr3_aa"]).reset_index(drop=True)

    # ablang tests
    @unittest.skipIf(sys.version_info >= (3, 14), "AbLang unavailable on Python >=3.14 (numba constraint)")
    def test_ablang_mixed_HL_embedding(self):
        """Test AbLang (mixed bulk sc HL)."""
        with self.assertWarns(UserWarning) as cm:
            embed(input_airr=self.test_airr_mixed_path, chain="HL", model="ablang", output_file_path="HL_test.pt")
            assert os.path.exists("HL_test.pt")
            embeddings = torch.load("HL_test.pt")
            assert embeddings.shape[1] == 768
            assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
        self.assertIn("trained on individual chains only", str(cm.warning))
        os.remove("HL_test.pt")
        os.remove("HL_test_metadata.tsv")

    @unittest.skipIf(sys.version_info >= (3, 14), "AbLang unavailable on Python >=3.14 (numba constraint)")
    def test_ablang_mixed_H_plus_L_embedding_tsv(self):
        """Test AbLang (mixed bulk sc H+L)."""
        embed(self.test_airr_mixed_path, "H+L", "ablang", "H_plus_L_test.tsv")
        assert os.path.exists("H_plus_L_test.tsv")
        embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 769  # 768 + id column
        assert embeddings.shape[0] == 5  # 2 H chain + 3 L chain
        os.remove("H_plus_L_test.tsv")
        os.remove("H_plus_L_test_metadata.tsv")

    @unittest.skipIf(sys.version_info >= (3, 14), "AbLang unavailable on Python >=3.14 (numba constraint)")
    def test_ablang_mixed_H_embedding_tsv(self):
        """Test AbLang H."""
        embed(self.test_airr_mixed_path, "H", "ablang", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 769  # 768 + id column
        assert embeddings.shape[0] == 2  # 2 H chain
        os.remove("H_test.tsv")
        os.remove("H_test_metadata.tsv")

    @unittest.skipIf(sys.version_info >= (3, 14), "AbLang unavailable on Python >=3.14 (numba constraint)")
    def test_ablang_residue_level(self):
        """Test AbLang residue-level embeddings."""
        embed(self.test_airr_mixed_path, "H", "ablang", "H_test_residue.pt", residue_level=True)
        assert os.path.exists("H_test_residue.pt")

        embeddings = torch.load("H_test_residue.pt")
        assert embeddings.shape[1] == 162  # max seq length for ablang
        assert embeddings.shape[2] == 768  # 768
        assert embeddings.shape[0] == 2  # 2 H chain
        os.remove("H_test_residue.pt")
        os.remove("H_test_residue_metadata.tsv")

    # antiberty tests
    def test_antiberty_mixed_HL_embedding(self):
        """Test antiberty (mixed bulk sc H+L)."""
        with self.assertWarns(UserWarning) as cm:
            embed(input_airr=self.test_airr_mixed_path, chain="HL", model="antiberty", output_file_path="HL_test.pt")
            assert os.path.exists("HL_test.pt")
            embeddings = torch.load("HL_test.pt")
            assert embeddings.shape[1] == 512
            assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
            os.remove("HL_test.pt")
            os.remove("HL_test_metadata.tsv")
        self.assertIn("was trained on individual chains only", str(cm.warning))

    def test_antiberty_mixed_H_plus_L_embedding_tsv(self):
        """Test antiberty (mixed bulk sc H+L)."""
        embed(self.test_airr_mixed_path, "H+L", "antiberty", "H_plus_L_test.tsv")
        assert os.path.exists("H_plus_L_test.tsv")
        embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 513  # antiberty embedding dimension + id column
        assert embeddings.shape[0] == 5  # 2 H chain + 3 L chain
        os.remove("H_plus_L_test.tsv")
        os.remove("H_plus_L_test_metadata.tsv")

    def test_antiberty_mixed_H_embedding_tsv(self):
        """Test antiberty (mixed bulk sc H)."""
        embed(self.test_airr_mixed_path, "H", "antiberty", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 513  # 512 + id column
        assert embeddings.shape[0] == 2  # 2 H chain
        os.remove("H_test.tsv")
        os.remove("H_test_metadata.tsv")

    def test_antiberty_mixed_H_embedding_h5ad(self):
        """Test antiberty (mixed bulk sc H) with h5ad output."""
        embed(self.test_airr_mixed_path, "H", "antiberty", "H_test.h5ad")
        assert os.path.exists("H_test.h5ad")
        import anndata

        embeddings = anndata.read_h5ad("H_test.h5ad")
        assert embeddings.X.shape[1] == 512  # Antiberty embedding dimension
        assert embeddings.X.shape[0] == 2  # 2 H chain
        os.remove("H_test.h5ad")

    def test_antiberty_residue_level(self):
        """Test AbLang residue-level embeddings."""
        embed(self.test_airr_mixed_path, "H", "antiberty", "H_test_residue.pt", residue_level=True)
        assert os.path.exists("H_test_residue.pt")

        embeddings = torch.load("H_test_residue.pt")
        assert embeddings.shape[1] == 510  # max seq length for antiberty
        assert embeddings.shape[2] == 512  # antiberty embedding dimension
        assert embeddings.shape[0] == 2  # 2 H chain
        os.remove("H_test_residue.pt")
        os.remove("H_test_residue_metadata.tsv")

    # antiberta2 tests
    def test_antiberta2_mixed_HL_embedding(self):
        """Test antiberta2 (mixed bulk sc H+L)."""
        with self.assertWarns(UserWarning) as cm:
            embed(input_airr=self.test_airr_mixed_path, chain="HL", model="antiberta2", output_file_path="HL_test.pt")
            assert os.path.exists("HL_test.pt")
            embeddings = torch.load("HL_test.pt")
            assert embeddings.shape[1] == 1024
            assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
        self.assertIn("trained on individual chains only", str(cm.warning))
        os.remove("HL_test.pt")
        os.remove("HL_test_metadata.tsv")

    def test_antiberta2_mixed_H_plus_L_embedding_tsv(self):
        """Test antiberta2 (mixed bulk sc H+L)."""
        embed(self.test_airr_mixed_path, "H+L", "antiberta2", "H_plus_L_test.tsv")
        assert os.path.exists("H_plus_L_test.tsv")
        embeddings = pd.read_table("H_plus_L_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1025  # 1024 + id column
        assert embeddings.shape[0] == 5  # 2 H chain + 3 L chain
        os.remove("H_plus_L_test.tsv")
        os.remove("H_plus_L_test_metadata.tsv")

    def test_antiberta2_mixed_H_embedding_tsv(self):
        """Test antiberta2 (mixed bulk sc H)."""
        embed(self.test_airr_mixed_path, "H", "antiberta2", "H_test.tsv")
        assert os.path.exists("H_test.tsv")
        embeddings = pd.read_table("H_test.tsv", delimiter="\t")
        assert embeddings.shape[1] == 1025  # 1024 + id column
        assert embeddings.shape[0] == 2  # 2 H chain
        os.remove("H_test.tsv")
        os.remove("H_test_metadata.tsv")

    def test_antiberta2_residue_level(self):
        """Test antiberta2 residue-level embeddings."""
        embed(self.test_airr_mixed_path, "H", "antiberta2", "H_test_residue.pt", residue_level=True)
        assert os.path.exists("H_test_residue.pt")

        embeddings = torch.load("H_test_residue.pt")
        assert embeddings.shape[1] == 256  # max seq length for antiberta2
        assert embeddings.shape[2] == 1024  # antiberta2 embedding dimension
        assert embeddings.shape[0] == 2  # 2 H chain
        os.remove("H_test_residue.pt")
        os.remove("H_test_residue_metadata.tsv")

    # balm-paired tests
    def test_balm_paired_mixed_HL_embedding(self):
        """Test balm-paired (mixed bulk sc HL)."""
        embed(input_airr=self.test_airr_mixed_path, chain="HL", model="balm-paired", output_file_path="HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 1024  # balm-paired embedding dimension
        assert embeddings.shape[0] == 1  # Just one cell with paired H and L chains
        os.remove("HL_test.pt")
        os.remove("HL_test_metadata.tsv")

    def test_balm_paired_mixed_H_plus_L_embedding_tsv(self):
        """Test balm-paired (mixed bulk sc H+L)."""
        with self.assertWarns(UserWarning) as cm:
            embed(self.test_airr_mixed_path, "H+L", "balm-paired", "H_plus_L_test.tsv")
        self.assertIn("was trained on paired chains", str(cm.warning))
        self.assertIn("--chain HL", str(cm.warning))
        os.remove("H_plus_L_test.tsv")
        os.remove("H_plus_L_test_metadata.tsv")

    def test_balm_paired_mixed_H_embedding_tsv(self):
        """Test balm-paired (mixed bulk sc H)."""
        with self.assertWarns(UserWarning) as cm:
            embed(self.test_airr_mixed_path, "H", "balm-paired", "H_test.tsv")
        self.assertIn("was trained on paired chains", str(cm.warning))
        self.assertIn("--chain HL", str(cm.warning))
        os.remove("H_test.tsv")
        os.remove("H_test_metadata.tsv")

    def test_balm_paired_residue_level(self):
        """Test balm-paired residue-level embeddings."""
        embed(self.test_airr_mixed_path, "HL", "balm-paired", "HL_test_residue.pt", residue_level=True)
        assert os.path.exists("HL_test_residue.pt")

        embeddings = torch.load("HL_test_residue.pt")
        assert embeddings.shape[1] == 510  # max seq length for balm-paired
        assert embeddings.shape[2] == 1024  # balm-paired embedding dimension
        assert embeddings.shape[0] == 1  # 1 H chain
        os.remove("HL_test_residue.pt")
        os.remove("HL_test_residue_metadata.tsv")

    ########################################################
    # vec-bcrdist tests
    def test_vec_bcrdist_reads_observed_cdrs(self):
        """Use AIRR CDR1/2/3 rather than inferred V-gene CDR strings."""
        cdrs = setup_bcr_cdr_strings(self.vec_bcrdist_df)
        self.assertEqual(cdrs[0], ("GFTFSSYW", "IKQDGTEK", "ARENLPSFFYYDSSAYLPEATFDF"))

    def test_vec_bcrdist_handles_cdr3_equal_to_trim_length(self):
        """A five-residue CDR3 has an empty trimmed interior and becomes gaps."""
        cdrs = self.vec_bcrdist_df.copy()
        cdrs.loc[cdrs.index[0], "cdr3_aa"] = "VVNQF"
        self.assertEqual(setup_bcr_cdr_strings(cdrs)[0][2], "VVNQF")
        self.assertEqual(BCRDistEncoder()._trim_and_gap_cdr3("VVNQF"), "-" * 32)

        cdrs.loc[cdrs.index[0], "cdr3_aa"] = "VVNQ"
        with self.assertRaisesRegex(ValueError, r"cdr3_aa.*length < 5"):
            setup_bcr_cdr_strings(cdrs)

    def test_vec_bcrdist_allows_missing_cdr1_cdr2_as_gaps(self):
        """Missing/empty CDR1/2 are kept and encoded as all-gap regions."""
        cdrs = self.vec_bcrdist_df.copy()
        cdrs.loc[cdrs.index[0], "cdr1_aa"] = pd.NA
        cdrs.loc[cdrs.index[0], "cdr2_aa"] = ""
        parsed = setup_bcr_cdr_strings(cdrs)
        self.assertEqual(parsed[0][0], "")
        self.assertEqual(parsed[0][1], "")

        encoder = BCRDistEncoder().fit(cdrs)
        self.assertEqual(encoder._trim_and_gap("", encoder.cdr1_num_pos), "-" * encoder.cdr1_num_pos)
        self.assertEqual(encoder._trim_and_gap("", encoder.cdr2_num_pos), "-" * encoder.cdr2_num_pos)
        embedding = encoder.transform(cdrs.iloc[[0]])
        self.assertEqual(embedding.shape, (1, 512))

    def test_vec_bcrdist_rejects_missing_cdr3_and_noncanonical(self):
        """CDR3 must be present; noncanonical residues are still rejected."""
        cdrs = self.vec_bcrdist_df.copy()
        cdrs.loc[cdrs.index[0], "cdr3_aa"] = pd.NA
        with self.assertRaisesRegex(ValueError, r"cdr3_aa.*missing or empty"):
            setup_bcr_cdr_strings(cdrs)

        cdrs = self.vec_bcrdist_df.copy()
        cdrs.loc[cdrs.index[0], "cdr1_aa"] = "GFTFXSYW"
        with self.assertRaisesRegex(ValueError, r"unsupported amino acid"):
            setup_bcr_cdr_strings(cdrs)

    def test_vec_bcrdist_changes_when_observed_cdr_changes(self):
        """A mutation in an observed CDR changes the receptor embedding."""
        embeddings = vec_bcrdist(self.vec_bcrdist_df)
        self.assertEqual(embeddings.shape, (5, 512))
        self.assertEqual(embeddings.dtype, torch.float32)
        self.assertFalse(torch.equal(embeddings[0], embeddings[1]))

    def test_vec_bcrdist_encoder_is_deterministic(self):
        """The fixed-seed AA MDS encoding is reproducible."""
        first = BCRDistEncoder().fit_transform(self.vec_bcrdist_df)
        second = BCRDistEncoder().fit_transform(self.vec_bcrdist_df)
        np.testing.assert_allclose(first, second)

    @unittest.skipUnless(HAS_PWSEQDIST, "requires optional dependency: pwseqdist")
    def test_vec_bcrdist_pwseqdist_reference_values(self):
        """Regenerate the six BCRdist reference values with pwseqdist.nb_tcrdist."""
        import pwseqdist as pw

        def region_distance(first, second, weight, n_trim, c_trim, fixed_gap):
            return weight * pw.metrics.nb_tcrdist(
                pw.matrices.seq2vec(first),
                pw.matrices.seq2vec(second),
                distance_matrix=pw.matrices.tcr_nb_distance_matrix,
                dist_weight=1,
                gap_penalty=4,
                ntrim=n_trim,
                ctrim=c_trim,
                fixed_gappos=fixed_gap,
            )

        def bcrdist(first, second):
            return (
                region_distance(first.cdr1_aa, second.cdr1_aa, 1, 0, 0, True)
                + region_distance(first.cdr2_aa, second.cdr2_aa, 1, 0, 0, True)
                + region_distance(first.cdr3_aa, second.cdr3_aa, 3, 3, 2, False)
            )

        data = self._bcrdist_reference_data()
        reference_bcrdist = np.array(
            [
                bcrdist(data.iloc[first], data.iloc[second])
                for first in range(len(data))
                for second in range(first + 1, len(data))
            ]
        )

        np.testing.assert_array_equal(
            reference_bcrdist,
            np.array([283.0, 257.0, 285.0, 148.0, 98.0, 155.0]),
        )

    def test_vec_bcrdist_embed_airr_filters_single_chain(self):
        """AMULETY passes only the requested heavy or light chain to vecBCRdist."""
        embedding, metadata = embed_airr(self.vec_bcrdist_df, chain="H", model="vec-bcrdist")
        self.assertEqual(embedding.shape, (2, 512))
        self.assertListEqual(metadata["locus"].tolist(), ["IGH", "IGH"])

    def test_vec_bcrdist_embed_airr_rejects_paired_input(self):
        """Paired heavy--light mode is outside the current vecBCRdist contract."""
        with self.assertRaisesRegex(ValueError, "single-chain"):
            embed_airr(self.vec_bcrdist_df, chain="H+L", model="vec-bcrdist")
