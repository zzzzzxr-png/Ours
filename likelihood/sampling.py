"""SRDTrans spatial-neighbor masking (upstream sampling.py)."""

import os
import sys

from .backbone_factory import ensure_srdtrans_repo_on_path

ensure_srdtrans_repo_on_path()

from sampling import generate_mask_pair, generate_subimages  # noqa: E402

__all__ = ['generate_mask_pair', 'generate_subimages']
