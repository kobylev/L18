"""
Test Runner

Run all unit tests for the Logistic Regression implementation.

Usage:
    python run_tests.py              # Run all tests
    python run_tests.py -v           # Verbose output
    python run_tests.py TestClassName # Run specific test class
"""

import sys
import unittest

# Discover and run all tests
if __name__ == '__main__':
    # Create test loader
    loader = unittest.TestLoader()

    # Discover all tests in current directory
    suite = loader.discover('.', pattern='test_*.py')

    # Run tests with appropriate verbosity
    verbosity = 2 if '-v' in sys.argv else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)

    # Run the tests
    result = runner.run(suite)

    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)
