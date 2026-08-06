#!/usr/bin/env python

"""Tests for `amulety` package.
Tests can be run with the command:
python -m unittest test_amulety.py
"""

import os
import sys
import unittest
from importlib.util import find_spec, module_from_spec, spec_from_file_location

import numpy as np
import pandas as pd
import torch

from amulety.amulety import embed, embed_airr
from amulety.bcr_embeddings import BCRDistEncoder, setup_bcr_cdr_strings, vec_bcrdist


HAS_PWSEQDIST = find_spec("pwseqdist") is not None
CONVBENCH_TCRDIST3_PATH = os.environ.get("CONVBENCH_TCRDIST3_PATH")
HAS_CONVBENCH_REFERENCE = (
    CONVBENCH_TCRDIST3_PATH is not None
    and os.path.isfile(CONVBENCH_TCRDIST3_PATH)
    and find_spec("tcrdist") is not None
    and HAS_PWSEQDIST
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

        self.bcr_cdr_data = pd.DataFrame(
            {
                "sequence_id": ["heavy_1", "heavy_2", "light_1"],
                "sequence_vdj_aa": ["AAAA", "AAAA", "AAAA"],
                "v_call": ["IGHV1-1*01", "IGHV1-1*01", "IGKV1-1*01"],
                "locus": ["IGH", "IGH", "IGK"],
                "cdr1_aa": ["GFTFSSYW", "GFTFSTYW", "QGISNS"],
                "cdr2_aa": ["IKQDGTEK", "IKQDGTEK", "TAS"],
                "cdr3_aa": ["ARENLPSFFYYDSSAYLPEATFDF", "ARENLPSFFYYDSSAYLPEATFDF", "QQYYSSVMYT"],
            }
        )

        # Bulk test data (no cell_id column)
        self.test_bulk = "AIRR_rearrangement_bulk_test.tsv"
        self.test_bulk_path = os.path.join(self.this_dir, self.test_bulk)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    @staticmethod
    def _bcrdist_reference_data():
        """Four translated BCRs used for the exact pwseqdist reference values."""
        return pd.DataFrame(
            {
                "cdr1_aa": ["GFTFSSYW", "SSNIGNNY", "GGSINSGDYS", "QGISNS"],
                "cdr2_aa": ["IKQDGTEK", "ENN", "IYHSGST", "TAS"],
                "cdr3_aa": [
                    "ARENLPSFFYYDSSAYLPEATFDF",
                    "GTWDSSLSAGV",
                    "ARSTVNIWGTFEY",
                    "QQYYSSVMYT",
                ],
            }
        )

    @staticmethod
    def _load_convbench_tcrdist3(path):
        """Load the pinned ConvBench reference script without adding it to AMULETY."""
        spec = spec_from_file_location("convbench_tcrdist3_reference", path)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

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

    # vec-bcrdist tests
    def test_vec_bcrdist_reads_observed_cdrs(self):
        """Use AIRR CDR1/2/3 rather than inferred V-gene CDR strings."""
        cdrs = setup_bcr_cdr_strings(self.bcr_cdr_data)
        self.assertEqual(cdrs[0], ("GFTFSSYW", "IKQDGTEK", "ARENLPSFFYYDSSAYLPEATFDF"))

    def test_vec_bcrdist_changes_when_observed_cdr_changes(self):
        """A mutation in an observed CDR changes the receptor embedding."""
        embeddings = vec_bcrdist(self.bcr_cdr_data)
        self.assertEqual(embeddings.shape, (3, 512))
        self.assertEqual(embeddings.dtype, torch.float32)
        self.assertFalse(torch.equal(embeddings[0], embeddings[1]))

    def test_vec_bcrdist_encoder_is_deterministic(self):
        """The fixed-seed AA MDS encoding is reproducible."""
        first = BCRDistEncoder().fit_transform(self.bcr_cdr_data)
        second = BCRDistEncoder().fit_transform(self.bcr_cdr_data)
        np.testing.assert_allclose(first, second)

    @unittest.skipUnless(HAS_CONVBENCH_REFERENCE, "requires pinned ConvBench tcrdist3 reference dependencies")
    def test_vec_bcrdist_matches_convbench_bcrdist(self):
        """Compare vecBCRdist directly with ConvBench's v2cdr=False BCRdist matrix."""
        data = self._bcrdist_reference_data()
        convbench_input = pd.DataFrame(
            {
                "id_col": [f"reference_{index}" for index in data.index],
                "subject_id": "reference_subject",
                # This mirrors ConvBench BCR input. With v2cdr=False, observed
                # CDR columns below, rather than V-gene inference, are used.
                "v_b_gene": ["IGHV3-7*01", "IGLV1-51*02", "IGHV4-30-2*01", "IGKV1-NL1*01"],
                "j_b_gene": ["IGHJ3*01", "IGLJ3*02", "IGHJ4*02", "IGKJ2*01"],
                "cdr1_b_aa": data["cdr1_aa"],
                "cdr2_b_aa": data["cdr2_aa"],
                "cdr3_b_aa": data["cdr3_aa"],
                "pmhc_b_aa": "",
                "seq_id": [f"s{index}" for index in data.index],
                "count": 1,
            }
        )
        convbench = self._load_convbench_tcrdist3(CONVBENCH_TCRDIST3_PATH)
        reference = convbench.run_tcrdist3(
            seqs=convbench_input,
            id_col="id_col",
            cpus=1,
            chunk_size=1000,
            radius=1000,
            v2cdr=False,
        ).rw_beta.toarray()
        vectors = BCRDistEncoder().fit_transform(data)
        upper_triangle = np.triu_indices(len(data), k=1)
        reference_bcrdist = reference[upper_triangle]
        squared_l2 = np.sum((vectors[:, None] - vectors[None, :]) ** 2, axis=2)[upper_triangle]

        self.assertTrue(np.all(reference_bcrdist > 0))
        self.assertGreater(np.corrcoef(reference_bcrdist, squared_l2)[0, 1], 0.98)
        self.assertLess(np.mean(np.abs(reference_bcrdist - squared_l2) / reference_bcrdist), 0.1)

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
        embedding, metadata = embed_airr(self.bcr_cdr_data, chain="H", model="vec-bcrdist")
        self.assertEqual(embedding.shape, (2, 512))
        self.assertListEqual(metadata["locus"].tolist(), ["IGH", "IGH"])

    def test_vec_bcrdist_embed_airr_rejects_paired_input(self):
        """Paired heavy--light mode is outside the current vecBCRdist contract."""
        with self.assertRaisesRegex(ValueError, "single-chain"):
            embed_airr(self.bcr_cdr_data, chain="H+L", model="vec-bcrdist")
