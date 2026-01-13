"""
LZGraphs Test Suite
==================

A comprehensive test suite for the LZGraphs package, organized by component:

- test_naive_lzgraph.py: Tests for NaiveLZGraph class
- test_ndp_lzgraph.py: Tests for NDPLZGraph (Nucleotide-position Dependent) class
- test_aap_lzgraph.py: Tests for AAPLZGraph (Amino Acid Positional) class
- test_metrics.py: Tests for diversity metrics (K1000, LZCentrality, entropy)
- test_utilities.py: Tests for utility functions (LZ decomposition, kmer dictionary)

Running Tests
-------------
From the project root directory:

    # Run all tests
    python -m pytest tests/ -v

    # Run specific test module
    python -m pytest tests/test_aap_lzgraph.py -v

    # Run with coverage
    python -m pytest tests/ --cov=LZGraphs --cov-report=html

Test Data
---------
Test data files (ExampleData1.csv, ExampleData2.csv, ExampleData3.csv) are located
in the tests/ directory and contain sample TCRB repertoire data for testing.
"""
