"""
Utilities

Helper functions for file operations and general utilities.
"""

import os


def create_output_directory(output_dir: str = 'output') -> str:
    """
    Create output directory if it doesn't exist.

    Args:
        output_dir: Name of the output directory

    Returns:
        Path to the output directory
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}/")
    return output_dir
