"""Per-model, per-device LRU for immutable image embeddings, bounded in bytes."""

from collections import OrderedDict

import torch


class ImageEmbeddingCache:
    def __init__(self, capacity_bytes: int):
        if capacity_bytes < 0:
            raise ValueError("Image cache capacity must be non-negative")
        self.capacity_bytes = capacity_bytes
        self._entries: OrderedDict[str, torch.Tensor] = OrderedDict()
        self.clear()

    def get(self, key: str) -> torch.Tensor | None:
        value = self._entries.get(key)
        if value is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return value

    def put(self, key: str, value: torch.Tensor) -> None:
        size = value.numel() * value.element_size()
        # An oversized image must not flush useful smaller entries.
        if self.capacity_bytes == 0 or size > self.capacity_bytes:
            return
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.size_bytes -= previous.numel() * previous.element_size()
        while self.size_bytes + size > self.capacity_bytes:
            _, removed = self._entries.popitem(last=False)
            self.size_bytes -= removed.numel() * removed.element_size()
            self.evictions += 1
        # Own exactly this tensor's storage, not an alias of a larger batch.
        self._entries[key] = value.detach().clone(memory_format=torch.contiguous_format)
        self.size_bytes += size

    def clear(self) -> None:
        self._entries.clear()
        self.size_bytes = self.hits = self.misses = self.evictions = 0

    def stats(self) -> dict[str, int]:
        return dict(
            capacity_bytes=self.capacity_bytes,
            size_bytes=self.size_bytes,
            entries=len(self._entries),
            hits=self.hits,
            misses=self.misses,
            evictions=self.evictions,
        )
