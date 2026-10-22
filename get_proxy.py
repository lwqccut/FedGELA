"""Compatibility module for the original training code.

The project historically imported this module with a wildcard, but all proxy
implementations now live in :mod:`model`.  Keeping this module makes the
published entry points importable without changing model behavior.
"""

