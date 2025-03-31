import faiss
import time
import numpy as np
import json
import os
from abc import ABC, abstractmethod
from collections import OrderedDict, Counter
from typing import Dict, Any, List, Tuple, Optional

class EvictionPolicy(ABC):
    """Abstract base class for cache eviction policies."""
    
    @abstractmethod
    def add_item(self, item_id: int) -> None:
        """Record a new item being added to the cache."""
        pass
    
    @abstractmethod
    def access_item(self, item_id: int) -> None:
        """Record an item being accessed."""
        pass
    
    @abstractmethod
    def get_items_to_evict(self, current_ids: List[int], num_to_evict: int) -> List[int]:
        """Return a list of item IDs that should be evicted."""
        pass


class FIFOPolicy(EvictionPolicy):
    """First In First Out eviction policy."""
    
    def __init__(self):
        self.entry_order = []
    
    def add_item(self, item_id: int) -> None:
        self.entry_order.append(item_id)
    
    def access_item(self, item_id: int) -> None:
        # In FIFO, access doesn't change eviction order
        pass
    
    def get_items_to_evict(self, current_ids: List[int], num_to_evict: int) -> List[int]:
        # Filter out any IDs that are no longer in the cache
        self.entry_order = [id for id in self.entry_order if id in current_ids]
        # Return the oldest entries
        items_to_evict = self.entry_order[:num_to_evict]
        # Update the entry order list
        self.entry_order = self.entry_order[num_to_evict:]
        return items_to_evict


class LRUPolicy(EvictionPolicy):
    """Least Recently Used eviction policy."""
    
    def __init__(self):
        # OrderedDict to track access order
        self.access_order = OrderedDict()
    
    def add_item(self, item_id: int) -> None:
        # New items are considered recently used
        self.access_order[item_id] = None
    
    def access_item(self, item_id: int) -> None:
        # Move accessed item to the end (most recently used)
        if item_id in self.access_order:
            self.access_order.move_to_end(item_id)
    
    def get_items_to_evict(self, current_ids: List[int], num_to_evict: int) -> List[int]:
        # Remove any IDs that are no longer in the cache
        self.access_order = OrderedDict((k, v) for k, v in self.access_order.items() if k in current_ids)
        # Get the least recently used items
        items_to_evict = list(self.access_order.keys())[:num_to_evict]
        # Remove evicted items from our tracking
        for item_id in items_to_evict:
            self.access_order.pop(item_id, None)
        return items_to_evict


class LFUPolicy(EvictionPolicy):
    """Least Frequently Used eviction policy."""
    
    def __init__(self):
        # Count accesses for each item
        self.access_count = Counter()
        # Track insertion time for tiebreaking
        self.insertion_time = {}
        self.time = 0
    
    def add_item(self, item_id: int) -> None:
        # New items start with a count of 1
        self.access_count[item_id] = 1
        # Record insertion time for tiebreaking
        self.insertion_time[item_id] = self.time
        self.time += 1
    
    def access_item(self, item_id: int) -> None:
        # Increment access count
        if item_id in self.access_count:
            self.access_count[item_id] += 1
    
    def get_items_to_evict(self, current_ids: List[int], num_to_evict: int) -> List[int]:
        # Remove any IDs that are no longer in the cache
        self.access_count = Counter({k: v for k, v in self.access_count.items() if k in current_ids})
        self.insertion_time = {k: v for k, v in self.insertion_time.items() if k in current_ids}
        
        # Sort by frequency, then by insertion time (oldest first)
        items_by_priority = sorted(
            self.access_count.keys(),
            key=lambda x: (self.access_count[x], -self.insertion_time.get(x, 0))
        )
        
        # Get least frequently used items
        items_to_evict = items_by_priority[:num_to_evict]
        
        # Remove evicted items from our tracking
        for item_id in items_to_evict:
            self.access_count.pop(item_id, None)
            self.insertion_time.pop(item_id, None)
            
        return items_to_evict


def init_cache(embedding_dim=768):
    """Initialize a Faiss index with ID mapping for efficient eviction."""
    # Create a standard Faiss index with Euclidean distance (L2)
    base_index = faiss.IndexFlatL2(embedding_dim)
    # Wrap it with an ID mapping layer to enable efficient removal
    index = faiss.IndexIDMap(base_index)
    
    # You would also initialize your encoder here
    # For this example I'm assuming it's already defined elsewhere
    encoder = None  # Replace with your actual encoder initialization
    
    return index, encoder


class SemanticCache:
    """A semantic cache with configurable eviction policies."""
    
    def __init__(
        self, 
        json_file="cache_file.json", 
        threshold=0.35, 
        max_size=100, 
        eviction_policy="FIFO",
        embedding_dim=768
    ):
        """Initialize the semantic cache.
        
        Args:
            json_file (str): The file to store cache data.
            threshold (float): Threshold for semantic similarity.
            max_size (int): Maximum number of items in the cache.
            eviction_policy (str): The policy for evicting items ("FIFO", "LRU", or "LFU").
            embedding_dim (int): Dimension of the embeddings.
        """
        # Initialize Faiss index
        self.index, self.encoder = init_cache(embedding_dim)
        
        self.threshold = threshold
        self.json_file = json_file
        self.max_size = max_size
        
        # Initialize cache storage using dictionaries for O(1) access
        self.cache = {
            "questions": {},
            "embeddings": {},
            "answers": {},
            "response_text": {}
        }
        
        # Set up eviction policy
        self._setup_eviction_policy(eviction_policy)
        
        # ID tracking
        self.next_id = 0
        self.all_ids = set()  # Track all IDs currently in the cache
        
        # Load cache from disk if it exists
        self._load_cache()
    
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
    
    def _load_cache(self) -> None:
        """Load cache from disk or initialize if it doesn't exist."""
        if os.path.exists(self.json_file):
            with open(self.json_file, 'r') as f:
                data = json.load(f)
                
                # Convert list-based data to dictionary-based
                for i, (question, embedding, answer, response) in enumerate(zip(
                    data.get("questions", []),
                    data.get("embeddings", []),
                    data.get("answers", []),
                    data.get("response_text", [])
                )):
                    item_id = self.next_id
                    self.next_id += 1
                    
                    # Add to cache
                    self.cache["questions"][item_id] = question
                    self.cache["embeddings"][item_id] = embedding
                    self.cache["answers"][item_id] = answer
                    self.cache["response_text"][item_id] = response
                    
                    # Add to Faiss index
                    np_emb = np.array([embedding], dtype=np.float32)
                    self.index.add_with_ids(np_emb, np.array([item_id]))
                    
                    # Track IDs
                    self.all_ids.add(item_id)
                    
                    # Record in eviction policy
                    self.eviction_policy.add_item(item_id)
    
    def _save_cache(self) -> None:
        """Save cache to disk."""
        # Convert dictionary-based cache to list-based for JSON storage
        data = {
            "questions": [],
            "embeddings": [],
            "answers": [],
            "response_text": []
        }
        
        # Get sorted IDs based on policy (for FIFO order in file)
        if self.policy_name == "FIFO":
            # For FIFO, we can get the exact order from the policy
            ids_order = self.eviction_policy.entry_order
        else:
            # For other policies, just use any order
            ids_order = list(self.all_ids)
        
        # Add items in the appropriate order
        for item_id in ids_order:
            if item_id in self.cache["questions"]:
                data["questions"].append(self.cache["questions"][item_id])
                data["embeddings"].append(self.cache["embeddings"][item_id])
                data["answers"].append(self.cache["answers"][item_id])
                data["response_text"].append(self.cache["response_text"][item_id])
        
        # Save to disk
        with open(self.json_file, 'w') as f:
            json.dump(data, f)
    
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
            self.cache["answers"].pop(item_id, None)
            self.cache["response_text"].pop(item_id, None)
            
            # Remove from ID tracking
            self.all_ids.remove(item_id)
    
    def ask(self, question: str) -> str:
        """Retrieve an answer from the cache or generate a new one."""
        start_time = time.time()
        
        try:
            # Get embedding for the question
            embedding = self.encoder.encode([question])
            
            # Search for the nearest neighbor in the index
            self.index.nprobe = 8
            D, I = self.index.search(embedding, 1)
            
            # Check if we found a match within the threshold
            if len(I) > 0 and len(I[0]) > 0 and D[0][0] <= self.threshold:
                # Get the Faiss ID of the match
                item_id = int(I[0][0])
                
                # Make sure it's actually in our cache
                if item_id in self.cache["response_text"]:
                    # Record the access for the eviction policy
                    self.eviction_policy.access_item(item_id)
                    
                    response_text = self.cache["response_text"][item_id]
                    
                    print("Answer recovered from Cache.")
                    print(f"{D[0][0]:.3f} smaller than {self.threshold}")
                    print(f"Found cache with ID: {item_id} with score {D[0][0]:.3f}")
                    print(f"response_text: " + response_text)
                    
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Time taken: {elapsed_time:.3f} seconds")
                    
                    return response_text
            
            # If we reach here, we didn't find a match in the cache
            # Query the database for a new answer
            answer = query_database([question], 1)
            response_text = answer["documents"][0][0]
            
            # Assign a new ID for this entry
            new_id = self.next_id
            self.next_id += 1
            
            # Add to cache dictionaries
            self.cache["questions"][new_id] = question
            self.cache["embeddings"][new_id] = embedding[0].tolist()
            self.cache["answers"][new_id] = answer
            self.cache["response_text"][new_id] = response_text
            
            # Add to tracking structures
            self.all_ids.add(new_id)
            
            # Record in eviction policy
            self.eviction_policy.add_item(new_id)
            
            # Add to Faiss index with the new ID
            self.index.add_with_ids(embedding, np.array([new_id]))
            
            print("Answer recovered from Database.")
            print(f"response_text: {response_text}")
            
            # Apply eviction if needed
            self.evict()
            
            # Save cache to disk
            self._save_cache()
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.3f} seconds")
            
            return response_text
            
        except Exception as e:
            raise RuntimeError(f"Error during 'ask' method: {e}")


# Placeholder for the query_database function
def query_database(questions, k=1):
    """Query a database for answers to questions. This is just a placeholder."""
    # In a real implementation, this would query your vector database
    return {"documents": [["This is a placeholder answer."]]}


# Example usage:
if __name__ == "__main__":
    # Create cache with LRU policy
    cache = SemanticCache(
        json_file="cache_lru.json",
        threshold=0.35,
        max_size=100,
        eviction_policy="LRU"
    )
    
    # Ask some questions
    response = cache.ask("What is the capital of France?")
    print(f"Response: {response}")
    
    # Ask again to test LRU behavior
    response = cache.ask("What is the capital of France?")
    print(f"Response: {response}")