#!/usr/bin/env python3
"""
Test script to verify Immune2Vec integration fixes.
This script tests the improved error handling and NaN detection.
"""

import sys

import pandas as pd

# Add the current directory to Python path for testing
sys.path.insert(0, ".")

# Add Immune2Vec path
immune2vec_path = "/Users/jiangwengyao/Downloads/Wang2024_Bioinformatics/immune2vec_model"
if immune2vec_path not in sys.path:
    sys.path.append(immune2vec_path)
    print(f"Added Immune2Vec path: {immune2vec_path}")


def test_immune2vec_basic():
    """Test basic Immune2Vec functionality with sample sequences."""

    # Sample immune receptor sequences (BCR/TCR-like)
    test_sequences = [
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDGDYWGQGTLVTVSS",
        "EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
        "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDGDYWGQGTLVTVSS",
        "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
    ]

    sequences = pd.Series(test_sequences)

    print("Testing Immune2Vec with sample sequences...")
    print(f"Number of sequences: {len(sequences)}")
    print(f"Sample sequence: {sequences.iloc[0][:50]}...")

    try:
        from amulety.protein_embeddings import immune2vec

        # Test with small parameters for quick testing
        embeddings = immune2vec(
            sequences=sequences,
            n_dim=50,  # Smaller dimension for testing
            n_gram=3,
            window=10,
            min_count=1,
            workers=1,
            cache_dir="/tmp/immune2vec_test",
        )

        print(f"Success! Generated embeddings shape: {embeddings.shape}")
        print(f"Embedding dtype: {embeddings.dtype}")
        print(f"Contains NaN: {torch.isnan(embeddings).any().item()}")
        print(f"Contains Inf: {torch.isinf(embeddings).any().item()}")
        print(f"Min value: {embeddings.min().item():.6f}")
        print(f"Max value: {embeddings.max().item():.6f}")
        print(f"Mean value: {embeddings.mean().item():.6f}")

        return True

    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure Immune2Vec is properly installed:")
        print("1. git clone https://bitbucket.org/yaarilab/immune2vec_model.git")
        print("2. Add to Python path: sys.path.append('/path/to/immune2vec_model')")
        return False

    except Exception as e:
        print(f"Error during embedding: {e}")
        return False


def test_edge_cases():
    """Test edge cases that might cause NaN values."""

    print("\nTesting edge cases...")

    # Test with very short sequences
    short_sequences = pd.Series(["A", "AC", "ACD"])

    # Test with empty sequences (should be filtered out)
    mixed_sequences = pd.Series(
        [
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
            "",  # Empty sequence
            "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDGDYWGQGTLVTVSS",
            "   ",  # Whitespace only
            "EVQLVESGGGLVKPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
        ]
    )

    try:
        from amulety.protein_embeddings import immune2vec

        print("Testing with short sequences...")
        embeddings_short = immune2vec(
            sequences=short_sequences,
            n_dim=20,
            n_gram=2,  # Smaller n-gram for short sequences
            window=5,
            min_count=1,
            workers=1,
        )
        print(f"Short sequences result: {embeddings_short.shape}")

        print("Testing with mixed sequences (including empty)...")
        embeddings_mixed = immune2vec(sequences=mixed_sequences, n_dim=30, n_gram=3, window=10, min_count=1, workers=1)
        print(f"Mixed sequences result: {embeddings_mixed.shape}")

        return True

    except Exception as e:
        print(f"Edge case test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Immune2Vec Integration Fixes")
    print("=" * 60)

    # Import torch for testing
    try:
        import torch

        print("PyTorch available")
    except ImportError:
        print("PyTorch not available - some tests may fail")
        sys.exit(1)

    # Run basic test
    basic_success = test_immune2vec_basic()

    # Run edge case tests if basic test passes
    if basic_success:
        edge_success = test_edge_cases()

        if edge_success:
            print("\n" + "=" * 60)
            print("All tests passed! Immune2Vec integration appears to be working correctly.")
            print("=" * 60)
        else:
            print("\n" + "=" * 60)
            print("Basic test passed but edge cases failed.")
            print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Basic test failed. Please check Immune2Vec installation.")
        print("=" * 60)
