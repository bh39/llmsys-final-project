"""Eviction policies for semantic cache."""

from abc import ABC, abstractmethod
from collections import OrderedDict, Counter
from typing import List


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