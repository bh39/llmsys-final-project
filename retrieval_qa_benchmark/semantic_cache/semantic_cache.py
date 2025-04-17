"""Semantic cache for efficient retrieval."""

import faiss
import time
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from loguru import logger

from .eviction_policies import EvictionPolicy, FIFOPolicy, LRUPolicy, LFUPolicy
from retrieval_qa_benchmark.utils.profiler import PROFILER



def init_cache(index_type="L2", embedding_dim=768):
    """Initialize a Faiss index with ID mapping for efficient eviction."""
    if index_type == "L2":
        base_index = faiss.IndexFlatL2(embedding_dim)
    elif index_type == "COSINE":
        base_index = faiss.IndexFlatIP(embedding_dim)
    else:
        raise ValueError(f"Unsupported index type: {index_type}")

    index = faiss.IndexIDMap(base_index)
    return index
    # Create a standard Faiss index with Euclidean distance (L2)
    # base_index = faiss.IndexFlatL2(embedding_dim)
    # # Wrap it with an ID mapping layer to enable efficient removal
    # index = faiss.IndexIDMap(base_index)

    # return index


class SemanticCache:
    """A semantic cache with configurable eviction policies."""
    
    def normalize(self, embedding: np.ndarray) -> np.ndarray:
        """L2-normalize an embedding in-place and return it."""
        embedding = embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(embedding)
        return embedding

    def __init__(
        self,
        threshold=0.35,
        max_size=100,
        eviction_policy="FIFO",
        index_type="L2",
        embedding_dim=768
    ):
        """Initialize the semantic cache.

        Args:
            threshold (float): Threshold for semantic similarity.
            max_size (int): Maximum number of items in the cache.
            eviction_policy (str): The policy for evicting items ("FIFO", "LRU", or "LFU").
            embedding_dim (int): Dimension of the embeddings.
        """
        # Initialize Faiss index
        self.index = init_cache(index_type=index_type, embedding_dim=embedding_dim)

        self.threshold = threshold
        self.max_size = max_size
        self.index_type = index_type
        self.normalize_required = index_type == "COSINE"

        # Initialize cache storage using dictionaries for O(1) access
        self.cache = {
            "questions": {},
            "embeddings": {},
            "distances": {},
            "entries": {}
        }

        # Set up eviction policy
        self._setup_eviction_policy(eviction_policy)

        # ID tracking
        self.next_id = 0
        self.all_ids = set()  # Track all IDs currently in the cache


        # logger.info(f"Initialized semantic cache with {eviction_policy} policy (max size: {max_size})")

    def _setup_eviction_policy(self, policy_name: str) -> None:
        """Set up the appropriate eviction policy."""
        policy_map = {
            "FIFO": FIFOPolicy,
            "LRU": LRUPolicy,
            "LFU": LFUPolicy,
        }

        if policy_name not in policy_map:
            raise ValueError(f"Unsupported eviction policy: {policy_name}. "
                            f"Supported policies are: {', '.join(policy_map.keys())}")

        self.policy_name = policy_name
        self.eviction_policy = policy_map[policy_name]()

    def evict(self) -> None:
        """Evict items from the cache if necessary."""
        if len(self.all_ids) <= self.max_size:
            return

        # Calculate how many items to evict
        num_to_evict = len(self.all_ids) - self.max_size

        # Get items to evict according to policy
        items_to_evict = self.eviction_policy.get_items_to_evict(list(self.all_ids), num_to_evict)

        # Remove items from cache and index
        for item_id in items_to_evict:
            # Remove from Faiss index
            self.index.remove_ids(np.array([item_id]))

            # Remove from cache dictionaries
            self.cache["questions"].pop(item_id, None)
            self.cache["embeddings"].pop(item_id, None)
            self.cache["distances"].pop(item_id, None)
            self.cache["entries"].pop(item_id, None)

            # Remove from ID tracking
            self.all_ids.remove(item_id)

        # logger.info(f"Evicted {num_to_evict} items from cache")

    def search(self, query_embedding: np.ndarray) -> Tuple[Optional[List[float]], Optional[List[Any]]]:
        """Search for an answer in the cache using a pre-computed embedding.

        Args:
            query_embedding (np.ndarray): The pre-computed embedding for the query.

        Returns:
            tuple: (distances, entries) or (None, None) if no match.
        """

        try:
            # Search for the nearest neighbor in the index
            # D, I = self.index.search(np.array([query_embedding]), 1)
            query_embedding = np.array(query_embedding).astype(np.float32).reshape(1, -1)
            if self.normalize_required:
                query_embedding = self.normalize(query_embedding)
            D, I = self.index.search(query_embedding, 1)


            if "hits" not in PROFILER:
                PROFILER.counter["hits"] = 0
                PROFILER.accumulator["hits"] = 0

            PROFILER.counter["hits"] += 1
            # Check if we found a match within the threshold
            # logger.info(f"distance: {D[0][0]}")
            match = D[0][0] >= self.threshold if self.normalize_required else D[0][0] <= self.threshold
            if len(I) > 0 and len(I[0]) > 0 and match:
                # Get the Faiss ID of the match
                item_id = int(I[0][0])

                # Make sure it's actually in our cache
                if item_id in self.cache["entries"]:
                    # Record the access for the eviction policy
                    self.eviction_policy.access_item(item_id)

                    entries = self.cache["entries"][item_id]
                    distances = self.cache["distances"][item_id]
                    question = self.cache["questions"][item_id]

                    # logger.info(f"Cache hit for query embedding (question: {question[:50]}...)")
                    # logger.info(f"Distance: {D[0][0]:.3f} (threshold: {self.threshold})")
                    # logger.info(f"Cache lookup time: {elapsed_time:.3f} seconds")

                    PROFILER.accumulator["hits"] += 1

                    return distances, entries

            # If we reach here, we didn't find a match in the cache
            # logger.info(f"Cache miss for query embedding")
            return None, None

        except Exception as e:
            logger.error(f"Error during cache search: {e}")
            return None, None

    def add(self, query: str, query_embedding: np.ndarray, distances: List[float], entries: List[Any]) -> None:
        """Add a new entry to the cache.

        Args:
            query (str): The query that was searched.
            query_embedding (np.ndarray): The embedding for the query.
            distances (list): The distances returned by the search.
            entries (list): The entries returned by the search.
        """
        try:
            # Assign a new ID for this entry
            new_id = self.next_id
            self.next_id += 1

            # Add to cache dictionaries
            self.cache["questions"][new_id] = query
            self.cache["embeddings"][new_id] = query_embedding.tolist()
            self.cache["distances"][new_id] = distances
            self.cache["entries"][new_id] = entries

            # Add to tracking structures
            self.all_ids.add(new_id)

            # Record in eviction policy
            self.eviction_policy.add_item(new_id)

            # Add to Faiss index with the new ID
            # self.index.add_with_ids(np.array([query_embedding]), np.array([new_id]))
            query_embedding = np.array(query_embedding).astype(np.float32).reshape(1, -1)
            if self.normalize_required:
                query_embedding = self.normalize(query_embedding)
            self.index.add_with_ids(query_embedding, np.array([new_id]))
            self.cache["embeddings"][new_id] = query_embedding.tolist()


            # logger.info(f"Added query to cache: {query[:50]}...")

            # Apply eviction if needed
            self.evict()

        except Exception as e:
            logger.error(f"Error adding to cache: {e}")