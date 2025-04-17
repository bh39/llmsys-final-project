"""MyScale searcher with semantic cache."""

from typing import Any, List, Optional, Tuple
import numpy as np
from loguru import logger

# Import the base searcher instead of BaseContextTransform
from retrieval_qa_benchmark.transforms.searchers.base import BaseSearcher, Entry
from retrieval_qa_benchmark.transforms.searchers.myscale import MyScaleSearcher
from retrieval_qa_benchmark.utils.registry import REGISTRY
from retrieval_qa_benchmark.utils.profiler import PROFILER

from retrieval_qa_benchmark.semantic_cache import SemanticCache
import time


class MyScaleSearcherWithCache(MyScaleSearcher):
    """Extension of MyScaleSearcher with semantic caching capabilities."""

    cache_threshold: float = 0.35
    """Threshold for semantic similarity in cache lookups"""
    cache_max_size: int = 1000
    """Maximum number of items in the cache"""
    cache_policy: str = "LRU"
    """Eviction policy for the cache"""
    index_type: str = "L2"
    """Index type for the cache"""
    enable_cache: bool = True
    """Whether to enable caching"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # Initialize cache if enabled
        if self.enable_cache:
            # Get the embedding dimension from the model
            embedding_dim = self.model.get_sentence_embedding_dimension()

            self.semantic_cache = SemanticCache(
                threshold=self.cache_threshold,
                max_size=self.cache_max_size,
                eviction_policy=self.cache_policy,
                index_type=self.index_type,
                embedding_dim=embedding_dim
            )
            # logger.info(f"Initialized semantic cache with policy {self.cache_policy}")
        else:
            self.semantic_cache = None
            # logger.info("Semantic cache is disabled")

    # @PROFILER.profile_function("database.MyScaleSearcherWithCache.search.profile")
    # def search(
    #     self,
    #     query_list: list,
    #     num_selected: int,
    #     context: Optional[List[List[str]]] = None,
    # ) -> Tuple[List[List[float]], List[List[Entry]]]:
    #     """Search for documents matching the queries, using cache when available."""
    #     assert len(query_list) == 1, "MyScale currently does not support batch mode."

    #     # Generate embedding for the query only once
    #     query = query_list[0]
    #     query_embedding = self.model.encode(query)

    #     # Try to get results from cache if enabled
    #     if self.enable_cache and self.semantic_cache:
    #         distances, entries = self.semantic_cache.search(query_embedding)
    #         if distances is not None and entries is not None:
    #             logger.info("Query served from cache")
    #             return [distances], [entries]

    #     # If cache miss or cache disabled, perform actual search
    #     # But avoid calling super().search() which would re-encode the query
    #     # logger.info("Cache miss or disabled, performing database search")

    #     # The following code is adapted from MyScaleSearcher.search
    #     if context is not None and context not in [[], [None]]:
    #         logger.warning("Ignoring context data in myscale search...")

    #     # Use the embedding we already computed instead of re-encoding
    #     emb_list = [query_embedding]

    #     query_sql = f"""SELECT d, title, text FROM {self.table_name}
    #         ORDER BY distance(emb,
    #             [{','.join(map(str, emb_list[0].tolist()))}]) AS d
    #         LIMIT {self.num_filtered if self.two_staged else num_selected}
    #         """

    #     if self.two_staged:
    #         self.ke_model.extract_keywords_from_text(query)
    #         from retrieval_qa_benchmark.transforms.searchers.myscale import is_sql_safe
    #         terms = [w for w in self.ke_model.get_ranked_phrases() if is_sql_safe(w)][
    #             : self.kw_topk
    #         ]
    #         terms_pattern = [f"(?i){x}" for x in terms]
    #         query_sql = (
    #             f"SELECT tempt.text AS text, tempt.title AS title, "
    #             f"distance1 + distance2 + tempt.d AS d "
    #             f"FROM ({query_sql}) tempt "
    #             f"ORDER BY "
    #             f"length(multiMatchAllIndices(arrayStringConcat("
    #             f"[tempt.title, tempt.text], ' '), {terms_pattern})) "
    #             f"AS distance1 DESC, "
    #             f"log(1 + countMatches(arrayStringConcat([tempt.title, "
    #             f"tempt.text], ' '), '(?i)({'|'.join(terms)})')) "
    #             f"AS distance2, d DESC LIMIT {num_selected}"
    #         )

    #     result = self.retrieve(query_sql)
    #     entry_list = [
    #         [
    #             Entry(rank=i, paragraph_id=i, title=r["title"], paragraph=r["text"])
    #             for i, r in enumerate(result)
    #         ]
    #     ]
    #     D_list = [[float(r["d"]) for r in result]]

    #     # Add results to cache if enabled
    #     if self.enable_cache and self.semantic_cache:
    #         try:
    #             self.semantic_cache.add(query, query_embedding, D_list[0], entry_list[0])
    #         except Exception as e:
    #             logger.error(f"Failed to add to cache: {e}")

    #     return D_list, entry_list

    @PROFILER.profile_function("cache.lookup.profile")
    def lookup_in_cache(self, query_embedding: np.ndarray) -> Tuple[Optional[List[float]], Optional[List[Entry]]]:
        """Look up results in the semantic cache.

        Args:
            query_embedding: The embedding to search for in the cache.

        Returns:
            A tuple of (distances, entries) if found, or (None, None) if not found.
        """
        if not self.enable_cache or not self.semantic_cache:
            return None, None

        distances, entries = self.semantic_cache.search(query_embedding)
        if distances is not None and entries is not None:
            # logger.info("Query served from cache")
            return distances, entries

        return None, None

    @PROFILER.profile_function("cache.add.profile")
    def add_to_cache(self, query: str, query_embedding: np.ndarray, distances: List[float], entries: List[Entry]) -> None:
        """Add results to the semantic cache.

        Args:
            query: The original query string.
            query_embedding: The embedding of the query.
            distances: The distances of the search results.
            entries: The search result entries.
        """
        if not self.enable_cache or not self.semantic_cache:
            return

        try:
            self.semantic_cache.add(query, query_embedding, distances, entries)
        except Exception as e:
            logger.error(f"Failed to add to cache: {e}")

    @PROFILER.profile_function("database.MyScaleSearcherWithCache.search.profile")
    def search(
        self,
        query_list: list,
        num_selected: int,
        context: Optional[List[List[str]]] = None,
    ) -> Tuple[List[List[float]], List[List[Entry]]]:
        """Search for documents matching the queries, using cache when available."""
        assert len(query_list) == 1, "MyScale currently does not support batch mode."

        # Generate embedding for the query only once
        query = query_list[0]
        query_embedding = self.model.encode(query)

        # Try to get results from cache
        distances, entries = self.lookup_in_cache(query_embedding)
        if distances is not None and entries is not None:
            return [distances], [entries]

        # If cache miss or cache disabled, perform actual search
        # The following code is adapted from MyScaleSearcher.search
        if context is not None and context not in [[], [None]]:
            logger.warning("Ignoring context data in myscale search...")

        # Use the embedding we already computed instead of re-encoding
        emb_list = [query_embedding]

        query_sql = f"""SELECT d, title, text FROM {self.table_name}
            ORDER BY distance(emb,
                [{','.join(map(str, emb_list[0].tolist()))}]) AS d
            LIMIT {self.num_filtered if self.two_staged else num_selected}
            """

        if self.two_staged:
            self.ke_model.extract_keywords_from_text(query)
            from retrieval_qa_benchmark.transforms.searchers.myscale import is_sql_safe
            terms = [w for w in self.ke_model.get_ranked_phrases() if is_sql_safe(w)][
                : self.kw_topk
            ]
            terms_pattern = [f"(?i){x}" for x in terms]
            query_sql = (
                f"SELECT tempt.text AS text, tempt.title AS title, "
                f"distance1 + distance2 + tempt.d AS d "
                f"FROM ({query_sql}) tempt "
                f"ORDER BY "
                f"length(multiMatchAllIndices(arrayStringConcat("
                f"[tempt.title, tempt.text], ' '), {terms_pattern})) "
                f"AS distance1 DESC, "
                f"log(1 + countMatches(arrayStringConcat([tempt.title, "
                f"tempt.text], ' '), '(?i)({'|'.join(terms)})')) "
                f"AS distance2, d DESC LIMIT {num_selected}"
            )

        result = self.retrieve(query_sql)
        entry_list = [
            [
                Entry(rank=i, paragraph_id=i, title=r["title"], paragraph=r["text"])
                for i, r in enumerate(result)
            ]
        ]
        D_list = [[float(r["d"]) for r in result]]

        # Add results to cache
        self.add_to_cache(query, query_embedding, D_list[0], entry_list[0])

        return D_list, entry_list