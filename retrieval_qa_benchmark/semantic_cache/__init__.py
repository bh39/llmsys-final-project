
"""Semantic cache module for efficient retrieval."""

from .semantic_cache import SemanticCache
from .eviction_policies import EvictionPolicy, FIFOPolicy, LRUPolicy, LFUPolicy

__all__ = [
    'SemanticCache',
    'EvictionPolicy',
    'FIFOPolicy',
    'LRUPolicy',
    'LFUPolicy',
]