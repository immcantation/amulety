#!/usr/bin/env python

"""Tests for `bcrembed` package.
Tests can be run with the command:
python -m unittest test_bcrembedder.py
"""


import unittest
import os
import torch
from bcrembed.__main__ import antiberty,antiberta2,esm2
import bcrembed.utils


class TestBcrembedder(unittest.TestCase):
    """Function that runs at start of tests for common resources.
    """

    def setUp(self):
        """Set up test fixtures, if any."""
        self.test_airr = "AIRR_rearrangement_translated.tsv"
        self.this_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_airr_path = os.path.join(self.this_dir, self.test_airr)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    ##################
    # BCRembed tests #
    ##################

    def test_antiberty_embedding(self):
        """Test something."""
        antiberty(self.test_airr_path, "HL", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 512
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

    def test_esm2_embedding(self):
        """Test esm2"""
        esm2(self.test_airr_path, "HL", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 1280
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")

    def test_antiberta2_embedding(self):
        """Test antiberta2"""
        antiberta2(self.test_airr_path, "HL", "HL_test.pt")
        assert os.path.exists("HL_test.pt")
        embeddings = torch.load("HL_test.pt")
        assert embeddings.shape[1] == 1024
        assert embeddings.shape[0] == 2
        os.remove("HL_test.pt")




