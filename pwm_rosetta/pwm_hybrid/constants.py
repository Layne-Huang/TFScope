"""Constants shared across the pwm_hybrid package."""

DNA_BASES = {'A', 'C', 'G', 'T'}

# PyRosetta uses different DNA residue names
DNA_RESIDUES = {"DA", "DC", "DG", "DT", "ADE", "THY", "GUA", "CYT"}

METAL_RES = {"ZN", "MG", "CA", "MN", "FE", "CU", "CO", "NI"}

# Mapping from both naming conventions to single-letter bases
BASE_MAPPING = {
    'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T',
    'ADE': 'A', 'CYT': 'C', 'GUA': 'G', 'THY': 'T'
}

# Watson-Crick base pairing (complementary strands)
COMPLEMENTARY = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}

# Full names for PyRosetta mutations
BASE_TO_FULL = {'A': 'ADE', 'C': 'CYT', 'G': 'GUA', 'T': 'THY'}
