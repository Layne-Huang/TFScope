"""Thin re-export of AF3 functions from multiflow/evaluation/AF3.py."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../multiflow/evaluation'))
from AF3 import AF3_folding, extract_sequences_from_pdb  # noqa: F401
