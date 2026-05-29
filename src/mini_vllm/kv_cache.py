import math
from dataclasses import dataclass, field
from typing import Callable, Optional

from .struct import KVCacheBlock, Request


Hash = str


def default_hasher(block_ids: list[int]) -> Hash:
    # Stable, deterministic hash for a block.
    return str(tuple(block_ids))


@dataclass(frozen=True)
class AllocationResult:
    block_ids: list[int]
    slot_mapping: list[int]

    def __bool__(self) -> bool:
        return True

    def __iter__(self):
        yield self.block_ids
        yield self.slot_mapping


@dataclass
class LRUPrefixCache:
    _n_freed: int = 0
    _root_block: KVCacheBlock = field(init=False)
    _head: KVCacheBlock = field(init=False)
    _tail: KVCacheBlock = field(init=False)

    def __post_init__(self) -> None:
        self._root_block = KVCacheBlock(idx=-1, reference_cnt=0)
        self._head = KVCacheBlock(idx=-1)
        self._tail = KVCacheBlock(idx=-1)
        self._head.next = self._tail
        self._tail.prev = self._head

    def get_max_allocable(self, hashes: list[Hash]) -> int:
        block = self._root_block
        n_allocable = self._n_freed
        for block_hash in hashes:
            if block_hash in block.next_blocks:
                block = block.next_blocks[block_hash]
                if block.reference_cnt > 0:
                    n_allocable += 1
            else:
                break
        return n_allocable

    def allocate(self, hashes: list[Hash]) -> list[KVCacheBlock]:
        block = self._root_block
        allocated: list[KVCacheBlock] = []
        for block_hash in hashes:
            if block_hash in block.next_blocks:
                block = block.next_blocks[block_hash]
                assert block._hash == block_hash
                allocated.append(block)
                if block.reference_cnt == 0:
                    assert block.prev is not None
                    assert block.next is not None
                    block.prev.next = block.next
                    block.next.prev = block.prev
                    block.prev = block.next = None
                    self._n_freed -= 1
                block.reference_cnt += 1
            else:
                break
        return allocated

    def free(self, blocks: list[KVCacheBlock]) -> None:
        prev_block = self._root_block
        push_at_block = self._head
        for block in blocks:
            assert block._hash is not None
            prev_block.next_blocks[block._hash] = block
            block.parent = prev_block
            block.reference_cnt -= 1
            if block.reference_cnt == 0:
                assert push_at_block.next is not None
                push_at_block.next.prev = block
                block.next = push_at_block.next
                block.prev = push_at_block
                push_at_block.next = block
                push_at_block = block
                self._n_freed += 1
            prev_block = block

    def evict(self, n: int) -> Optional[list[KVCacheBlock]]:
        if self._n_freed < n:
            return None

        block = self._tail.prev
        blocks: list[KVCacheBlock] = []
        for _ in range(n):
            assert block is not self._head
            assert len(block.next_blocks) == 0
            assert block is not None
            assert block.next is not None
            assert block.prev is not None
            assert block.parent is not None
            assert block._hash is not None

            self._n_freed -= 1
            blocks.append(block)
            block.next.prev = block.prev
            block.prev.next = block.next
            next_block = block.prev
            block.next = block.prev = None
            block.parent.next_blocks.pop(block._hash)
            block.parent = None
            block._hash = None
            block = next_block
        return blocks


@dataclass
class PagedKVCacheManager:
    block_size: int
    num_blocks: int
    model_tag: str = "default"
    _free_blocks: list[KVCacheBlock] = field(default_factory=list)
    _prefix_cache: LRUPrefixCache = field(default_factory=LRUPrefixCache)
    _hasher: Callable[[list[int]], Hash] = default_hasher

    @staticmethod
    def from_vllm_config(vllm_config):
        from vllm.config import VllmConfig

        assert isinstance(vllm_config, VllmConfig)
        return PagedKVCacheManager(
            block_size=vllm_config.cache_config.block_size,
            num_blocks=vllm_config.cache_config.num_gpu_blocks,
            model_tag=vllm_config.model_config.model,
        )

    def __post_init__(self) -> None:
        for idx in range(self.num_blocks):
            self._free_blocks.append(KVCacheBlock(idx=idx))

    def _get_num_needed_blocks(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.block_size)

    def _hash(self, tokens: list[int]) -> list[Hash]:
        hashes: list[Hash] = []
        for i in range(self.block_size, len(tokens) + 1, self.block_size):
            hashes.append(self._hasher(tokens[i - self.block_size : i]))
        return hashes

    def _alloc_from_free_and_evict(self, n: int) -> list[KVCacheBlock]:
        assert n >= 0
        assert n <= len(self._free_blocks) + self._prefix_cache._n_freed

        n_from_free = min(len(self._free_blocks), n)
        n_from_evict = n - n_from_free
        free_blocks = self._free_blocks[:n_from_free]
        self._free_blocks = self._free_blocks[n_from_free:]
        evict_blocks = self._prefix_cache.evict(n_from_evict)
        assert evict_blocks is not None

        new_blocks = free_blocks + evict_blocks
        for block in new_blocks:
            assert block.reference_cnt == 0
            assert block._hash is None
            assert block.parent is None
            assert len(block.next_blocks) == 0
            assert block.prev is None and block.next is None
            block.reference_cnt += 1
        return new_blocks

    def allocate_prefill(self, request: Request) -> bool:
        assert request.n_computed_tokens == 0
        assert request.blocks is None
        prompt_hashes = self._hash(request.prompt_ids)
        
        n_blocks = self._get_num_needed_blocks(request.n_prompt_tokens)

        if (
            self._prefix_cache.get_max_allocable(prompt_hashes)
            + len(self._free_blocks)
            < n_blocks
        ):
            return False

        cached_blocks = self._prefix_cache.allocate(prompt_hashes)
        new_blocks = self._alloc_from_free_and_evict(n_blocks - len(cached_blocks))

        request.blocks = cached_blocks + new_blocks
        request.n_computed_tokens = len(cached_blocks) * self.block_size
        return True

    def allocate(
        self,
        request: Request,
        n_new_tokens: int,
    ) -> AllocationResult | None:
        assert request.blocks is not None
        start_token = request.n_computed_tokens
        end_token = start_token + n_new_tokens
        n_blocks_needed = self._get_num_needed_blocks(end_token) - request.n_blocks

        if n_blocks_needed > len(self._free_blocks) + self._prefix_cache._n_freed:
            return None

        new_blocks = self._alloc_from_free_and_evict(n_blocks_needed)
        request.blocks.extend(new_blocks)
        return AllocationResult(
            block_ids=request.block_ids,
            slot_mapping=self._get_slot_mapping(request, start_token, n_new_tokens),
        )

    def _get_slot_mapping(
        self,
        request: Request,
        start_token: int,
        n_tokens: int,
    ) -> list[int]:
        assert request.blocks is not None
        slots: list[int] = []
        for token_idx in range(start_token, start_token + n_tokens):
            block = request.blocks[token_idx // self.block_size]
            slots.append(block.idx * self.block_size + token_idx % self.block_size)
        return slots

    def free(self, request: Request) -> None:
        assert request.blocks is not None
        computed_hashes = self._hash(request.computed_tokens)
        for block, block_hash in zip(request.blocks, computed_hashes):
            assert block._hash is None or block._hash == block_hash
            block._hash = block_hash

        n_cached_blocks = len(computed_hashes)
        self._prefix_cache.free(request.blocks[:n_cached_blocks])
        for block in request.blocks[n_cached_blocks:]:
            assert block._hash is None
            self._free_blocks.append(block)
            block.reference_cnt -= 1
            assert block.reference_cnt == 0
