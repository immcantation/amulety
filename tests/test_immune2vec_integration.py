import os
import tempfile
import unittest

import pandas as pd
import torch

from amulety.protein_embeddings import immune2vec


class TestImmune2VecIntegration(unittest.TestCase):
    """Test Immune2Vec functionality with custom installation paths."""

    @classmethod
    def setUpClass(cls):
        """Set up test data and check for Immune2Vec availability."""
        # Create test data
        cls.test_data = pd.DataFrame(
            {
                "sequence_id": ["seq1", "seq2", "seq3", "seq4"],
                "cell_id": ["cell1", "cell1", "cell2", "cell2"],
                "sequence_vdj_aa": [
                    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
                    "DIQMTQSPSSLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGGGTKVEIK",
                    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYYMHWVRQAPGQGLEWMGGINPSNGGTNFNEKFKNRVTITADESTSTAYMELSSLRSEDTAVYYCAR",
                    "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASSRATGIPDRFSGSGSGTDFTLTISRLEPEDFAVYYCQQYGSSPLTFGGGTKVEI",
                ],
                "chain": ["H", "L", "H", "L"],
                "duplicate_count": [1, 1, 1, 1],
                "v_call": ["IGHV1-69*01", "IGKV1-39*01", "IGHV3-23*01", "IGKV2-28*01"],
                "locus": ["IGH", "IGK", "IGH", "IGK"],
            }
        )

        # Create temporary test file
        cls.temp_dir = tempfile.mkdtemp()
        cls.test_file_path = os.path.join(cls.temp_dir, "test_data.tsv")
        cls.test_data.to_csv(cls.test_file_path, sep="\t", index=False)

        # Get Immune2Vec path from environment or use default search
        cls.installation_path = os.environ.get("INSTALLATION_PATH")
        if cls.installation_path:
            print(f"Using Immune2Vec path from environment: {cls.installation_path}")
        else:
            print("No INSTALLATION_PATH set, will use default search paths")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files."""
        import shutil

        if os.path.exists(cls.temp_dir):
            shutil.rmtree(cls.temp_dir)

    def test_immune2vec_with_custom_path_programmatic(self):
        """Test immune2vec function with custom path parameter."""
        sequences = pd.Series(
            [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
                "DIQMTQSPSSLSASVGDRVTITCRASQSISSWLAWYQQKPGKAPKLLIYKASSLESGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQYNSYPLTFGGGTKVEIK",
            ]
        )

        # Test with custom path
        embeddings = immune2vec(
            sequences=sequences,
            cache_dir=self.temp_dir,
            batch_size=2,
            n_dim=50,  # Smaller dimension for faster testing
            installation_path=self.installation_path,
        )

        # Verify results
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], 2)
        self.assertEqual(embeddings.shape[1], 50)
        self.assertEqual(embeddings.dtype, torch.float32)

        print(f"Programmatic test passed: {embeddings.shape}")

    def test_immune2vec_cli_with_custom_path_H_chain(self):
        """Test CLI interface with custom path for H chain."""
        # Import embed function and test directly
        from amulety.amulety import embed_airr

        embeddings, _ = embed_airr(
            airr=self.test_data,
            chain="H",
            model="immune2vec",
            cache_dir=self.temp_dir,
            batch_size=2,
            output_type="pickle",
            installation_path=self.installation_path,
        )

        # Verify results
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], 2)  # 2 H chains
        self.assertEqual(embeddings.shape[1], 100)  # Default dimension

        print(f"CLI H chain test passed: {embeddings.shape}")

    def test_immune2vec_cli_with_custom_path_HL_pairs(self):
        """Test CLI interface with custom path for HL pairs."""
        from amulety.amulety import embed_airr

        embeddings, _ = embed_airr(
            airr=self.test_data,
            chain="HL",
            model="immune2vec",
            cache_dir=self.temp_dir,
            batch_size=2,
            output_type="pickle",
            installation_path=self.installation_path,
        )

        # Verify results
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], 2)  # 2 cells with HL pairs
        self.assertEqual(embeddings.shape[1], 100)  # Default dimension

        print(f"CLI HL pairs test passed: {embeddings.shape}")

    def test_immune2vec_cli_with_custom_path_H_plus_L(self):
        """Test CLI interface with custom path for H+L separate chains."""
        from amulety.amulety import embed_airr

        embeddings, _ = embed_airr(
            airr=self.test_data,
            chain="H+L",
            model="immune2vec",
            cache_dir=self.temp_dir,
            batch_size=2,
            output_type="pickle",
            installation_path=self.installation_path,
        )

        # Verify results
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], 4)  # 2H + 2L chains
        self.assertEqual(embeddings.shape[1], 100)  # Default dimension

        print(f"CLI H+L test passed: {embeddings.shape}")

    def test_immune2vec_error_handling_invalid_path(self):
        """Test error handling with invalid path."""
        sequences = pd.Series(
            [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS"
            ]
        )

        with self.assertRaises(ImportError) as context:
            immune2vec(
                sequences=sequences,
                cache_dir=self.temp_dir,
                batch_size=1,
                n_dim=50,
                installation_path="/invalid/immune2vec/installation/path",
            )

        # Verify error message mentions both installation options
        error_msg = str(context.exception)
        self.assertIn("Option A", error_msg)
        self.assertIn("Option B", error_msg)
        self.assertIn("installation_path", error_msg)

        print("Error handling test passed")

    def test_immune2vec_without_custom_path(self):
        """Test immune2vec without custom path (should use default search)."""
        sequences = pd.Series(
            [
                "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMHWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS"
            ]
        )

        # This should work if immune2vec is in the default search paths
        embeddings = immune2vec(
            sequences=sequences,
            cache_dir=self.temp_dir,
            batch_size=1,
            n_dim=50,
            installation_path=None,  # Explicitly test None
        )

        # Verify results
        self.assertIsInstance(embeddings, torch.Tensor)
        self.assertEqual(embeddings.shape[0], 1)
        self.assertEqual(embeddings.shape[1], 50)

        print(f"Default path test passed: {embeddings.shape}")


if __name__ == "__main__":
    unittest.main()
