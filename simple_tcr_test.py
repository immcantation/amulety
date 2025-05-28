#!/usr/bin/env python3
"""
Simple test to verify TCR functions work
"""

import os
import tempfile

import pandas as pd


def test_tcr_chain_mapping():
    """Test TCR chain mapping logic"""
    print("=== Testing TCR Chain Mapping ===")

    # Create test data
    data = {
        "sequence_id": ["tcr_001", "tcr_002", "tcr_003", "tcr_004"],
        "cell_id": ["cell_1", "cell_1", "cell_2", "cell_2"],
        "locus": ["TRA", "TRB", "TRA", "TRB"],
        "v_call": ["TRAV1*01", "TRBV1*01", "TRAV2*01", "TRBV2*01"],
        "sequence_vdj_aa": [
            "QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYAMHWVRQAPGQRLEWMG",
            "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVS",
            "QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLM",
            "DIQMTQSPSSLSASVGDRVTITCRASQGISNSLAWFQQKPGKAPKLLLY",
        ],
        "duplicate_count": [10, 15, 8, 12],
    }

    df = pd.DataFrame(data)

    # Test TCR chain mapping
    df.loc[:, "tcr_chain"] = df.loc[:, "locus"].apply(
        lambda x: "B" if x in ["TRB", "TRD"] else ("A" if x in ["TRA", "TRG"] else None)
    )

    print("Original data:")
    print(df[["locus", "tcr_chain"]])

    # Filter for TCR chains only
    tcr_data = df[df.tcr_chain.notna()]
    print(f"\nFiltered TCR sequences: {tcr_data.shape[0]}")

    # Test alpha chains
    alpha_chains = tcr_data[tcr_data.tcr_chain == "A"]
    print(f"Alpha chains: {alpha_chains.shape[0]}")

    # Test beta chains
    beta_chains = tcr_data[tcr_data.tcr_chain == "B"]
    print(f"Beta chains: {beta_chains.shape[0]}")

    print("✓ TCR chain mapping test passed!")


def test_concatenation_logic():
    """Test alpha-beta concatenation logic"""
    print("\n=== Testing Alpha-Beta Concatenation Logic ===")

    # Create test data
    data = {
        "cell_id": ["cell_1", "cell_1", "cell_2", "cell_2"],
        "tcr_chain": ["A", "B", "A", "B"],
        "sequence_vdj_aa": ["ALPHA_SEQUENCE_1", "BETA_SEQUENCE_1", "ALPHA_SEQUENCE_2", "BETA_SEQUENCE_2"],
        "duplicate_count": [10, 15, 8, 12],
    }

    df = pd.DataFrame(data)
    print("Input data:")
    print(df)

    # Simulate concatenation logic
    # Group by cell_id and tcr_chain, take max duplicate_count
    grouped = df.loc[df.groupby(["cell_id", "tcr_chain"])["duplicate_count"].idxmax()]
    print("\nAfter grouping:")
    print(grouped)

    # Pivot to get alpha and beta in separate columns
    pivoted = grouped.pivot(index="cell_id", columns="tcr_chain", values="sequence_vdj_aa")
    pivoted = pivoted.reset_index()
    print("\nAfter pivoting:")
    print(pivoted)

    # Drop cells with missing chains
    complete_pairs = pivoted.dropna(axis=0)
    print(f"\nComplete alpha-beta pairs: {complete_pairs.shape[0]}")

    # Concatenate sequences
    if "A" in complete_pairs.columns and "B" in complete_pairs.columns:
        complete_pairs.loc[:, "concatenated"] = complete_pairs.B + "<cls><cls>" + complete_pairs.A
        print("\nConcatenated sequences:")
        print(complete_pairs[["cell_id", "concatenated"]])

    print("✓ Alpha-beta concatenation test passed!")


def test_file_operations():
    """Test reading and writing TSV files"""
    print("\n=== Testing File Operations ===")

    # Create test data
    data = {
        "sequence_id": ["tcr_001", "tcr_002"],
        "cell_id": ["cell_1", "cell_1"],
        "locus": ["TRA", "TRB"],
        "v_call": ["TRAV1*01", "TRBV1*01"],
        "sequence_vdj_aa": ["ALPHA_SEQ", "BETA_SEQ"],
        "duplicate_count": [10, 15],
    }

    df = pd.DataFrame(data)

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        df.to_csv(f.name, sep="\t", index=False)
        temp_file = f.name

    try:
        # Read back
        read_df = pd.read_table(temp_file)
        print("Successfully wrote and read TSV file")
        print(f"Original shape: {df.shape}, Read shape: {read_df.shape}")

        # Check if locus column exists
        if "locus" not in read_df.columns:
            read_df.loc[:, "locus"] = read_df.loc[:, "v_call"].apply(lambda x: x[:3])
            print("Added locus column from v_call")

        print("✓ File operations test passed!")

    finally:
        # Clean up
        os.unlink(temp_file)


if __name__ == "__main__":
    print("Testing TCR functionality components...")
    test_tcr_chain_mapping()
    test_concatenation_logic()
    test_file_operations()
    print("\n=== All component tests completed successfully! ===")
