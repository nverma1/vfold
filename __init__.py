"""
Package for KV alignment and analysis.
Primary focus: attn_head_align for attention head alignment.
"""

from .attn_head_align import (
    add_hooks_to_model,
    apply_v_alignment,
    do_alignment,
    get_wikitext2,
    prepare_dataloader,
)

__all__ = [
    "add_hooks_to_model",
    "apply_v_alignment",
    "do_alignment",
    "get_wikitext2",
    "prepare_dataloader",
]

