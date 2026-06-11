try:
    from .folding import AF3_folding, extract_sequences_from_pdb
    HAS_AF3 = True
    AF3_IMPORT_ERROR = None
except ImportError as e:
    HAS_AF3 = False
    AF3_IMPORT_ERROR = str(e)
    AF3_folding = None
    extract_sequences_from_pdb = None
