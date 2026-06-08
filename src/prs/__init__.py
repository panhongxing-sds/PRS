"""PRS: Perturbation Reliability Score for LLM uncertainty and failure detection."""

from prs.ase.prs import compute_prs, compute_prs_from_row, enrich_row_with_prs

__version__ = "0.1.0"
__all__ = ["compute_prs", "compute_prs_from_row", "enrich_row_with_prs", "__version__"]
