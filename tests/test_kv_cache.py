import random
import pytest

from mini_vllm.kv_cache import PagedKVCacheManager, Request


def _make_request(req_id: str, prompt_ids: list[int], block_size: int) -> Request:
    return Request(
        req_id=req_id,
        prompt_text="",
        block_size=block_size,
        prompt_ids=prompt_ids,
    )


def _collect_cache_tree_nodes(manager: PagedKVCacheManager) -> list:
    cache = manager._prefix_cache
    nodes = []
    seen = set()
    stack = list(cache._root_block.next_blocks.values())
    while stack:
        node = stack.pop()
        if node.idx in seen:
            continue
        seen.add(node.idx)
        nodes.append(node)
        stack.extend(node.next_blocks.values())
    return nodes


def _collect_lru_nodes(manager: PagedKVCacheManager) -> list:
    cache = manager._prefix_cache
    nodes = []
    cur = cache._head.next
    while cur is not cache._tail:
        nodes.append(cur)
        cur = cur.next
    return nodes


def _assert_prefix_cache_invariants(manager: PagedKVCacheManager):
    tree_nodes = _collect_cache_tree_nodes(manager)
    lru_nodes = _collect_lru_nodes(manager)

    assert manager._prefix_cache._n_freed == len(lru_nodes)
    assert all(node.reference_cnt == 0 for node in lru_nodes)

    zero_ref_nodes = {node.idx for node in tree_nodes if node.reference_cnt == 0}
    assert zero_ref_nodes == {node.idx for node in lru_nodes}


def _assert_no_orphans(manager: PagedKVCacheManager, live_blocks: set):
    tree_nodes = _collect_cache_tree_nodes(manager)
    all_seen = {block.idx for block in manager._free_blocks}
    all_seen |= {block.idx for block in tree_nodes}
    all_seen |= {block.idx for block in live_blocks}
    assert len(all_seen) == manager.num_blocks


def test_allocate_prefill_basic():
    manager = PagedKVCacheManager(model_tag="unit", block_size=4, num_blocks=8)
    request = _make_request("r1", list(range(8)), block_size=4)

    assert manager.allocate_prefill(request) is True
    assert request.n_computed_tokens == 0
    assert request.n_blocks == 2

    block_ids = request.block_ids
    assert block_ids == [0, 1]
    request.n_computed_tokens = 4
    manager.free(request)
    assert request.blocks[0]._hash is not None
    assert request.blocks[1]._hash is None


def test_free_populates_cache_and_reuse():
    manager = PagedKVCacheManager(model_tag="unit", block_size=4, num_blocks=8)
    request = _make_request("r1", list(range(8)), block_size=4)

    assert manager.allocate_prefill(request) is True
    # Mark prefill as complete so free() caches the blocks.
    request.n_computed_tokens = request.n_prompt_tokens
    manager.free(request)

    follow_up = _make_request("r2", list(range(8)), block_size=4)
    assert manager.allocate_prefill(follow_up) is True
    assert follow_up.n_computed_tokens == 8
    assert follow_up.block_ids == request.block_ids


def test_allocate_slot_mapping_grows_blocks():
    manager = PagedKVCacheManager(model_tag="unit", block_size=4, num_blocks=8)
    request = _make_request("r1", list(range(8)), block_size=4)

    assert manager.allocate_prefill(request) is True
    request.n_computed_tokens = 7
    block_ids, slot_mapping = manager.allocate(request, n_new_tokens=3)

    assert block_ids == [0, 1, 2]
    assert slot_mapping == [7, 8, 9]


def test_lru_eviction_reuses_oldest_blocks():
    manager = PagedKVCacheManager(model_tag="unit", block_size=4, num_blocks=3)

    r1 = _make_request("r1", list(range(8)), block_size=4)
    assert manager.allocate_prefill(r1) is True
    r1.n_computed_tokens = r1.n_prompt_tokens
    manager.free(r1)

    r2 = _make_request("r2", list(range(100, 104)), block_size=4)
    assert manager.allocate_prefill(r2) is True
    r2.n_computed_tokens = r2.n_prompt_tokens
    manager.free(r2)

    assert manager._prefix_cache._n_freed == 3

    r3 = _make_request("r3", list(range(200, 208)), block_size=4)
    assert manager.allocate_prefill(r3) is True
    assert r3.block_ids == [1, 0]
    assert manager._prefix_cache._n_freed == 1

    for block in r3.blocks:
        assert block._hash is None
        assert block.parent is None


def test_multi_turn_chat_prefix_reuse_with_generated_tokens():
    manager = PagedKVCacheManager(model_tag="unit", block_size=4, num_blocks=6)

    prompt = list(range(6))
    r1 = _make_request("r1", prompt, block_size=4)
    assert manager.allocate_prefill(r1) is True

    r1.generated_ids.extend([100, 101])
    r1.n_computed_tokens = r1.n_prompt_tokens + r1.n_generated_tokens
    manager.free(r1)

    r2_prompt = prompt + r1.generated_ids
    r2 = _make_request("r2", r2_prompt, block_size=4)
    assert manager.allocate_prefill(r2) is True
    assert r2.n_computed_tokens == 8
    assert r2.block_ids == r1.block_ids


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fuzz_allocations_and_frees(seed):
    rng = random.Random(seed)
    block_size = rng.randint(1, 4)
    num_blocks = rng.randint(4, 8)
    manager = PagedKVCacheManager(
        model_tag=f"fuzz-{seed}",
        block_size=block_size,
        num_blocks=num_blocks,
    )

    for step in range(20):
        prompt_len = rng.randint(1, block_size * num_blocks)
        prompt_ids = list(range(step * 1000, step * 1000 + prompt_len))
        request = _make_request(f"r{seed}-{step}", prompt_ids, block_size=block_size)

        ok = manager.allocate_prefill(request)
        if not ok:
            continue

        request.n_computed_tokens = request.n_prompt_tokens
        max_gen = block_size * num_blocks - request.n_computed_tokens
        gen_len = rng.randint(0, max_gen)

        if gen_len:
            block_ids, slot_mapping = manager.allocate(request, n_new_tokens=gen_len)
            assert block_ids == request.block_ids
            assert len(slot_mapping) == gen_len
            # assert slot_mapping == list(range(slot_mapping[0], slot_mapping[0] + gen_len))
            request.generated_ids.extend(
                range(step * 10000, step * 10000 + gen_len),
            )
            request.n_computed_tokens += gen_len

        manager.free(request)
        for block in request.blocks:
            assert block.reference_cnt == 0

        _assert_prefix_cache_invariants(manager)
        _assert_no_orphans(manager, live_blocks=set())


def test_many_allocations_leave_empty_lru_chain():
    manager = PagedKVCacheManager(model_tag="unit", block_size=4, num_blocks=4)
    request = _make_request("r1", list(range(16)), block_size=4)

    assert manager.allocate_prefill(request) is True
    assert manager._prefix_cache._n_freed == 0
    assert _collect_lru_nodes(manager) == []


def test_no_orphans_after_freeing_all_requests():
    manager = PagedKVCacheManager(model_tag="unit", block_size=4, num_blocks=6)

    r1 = _make_request("r1", list(range(8)), block_size=4)
    r2 = _make_request("r2", list(range(8, 16)), block_size=4)

    assert manager.allocate_prefill(r1) is True
    assert manager.allocate_prefill(r2) is True

    r1.n_computed_tokens = r1.n_prompt_tokens
    r2.n_computed_tokens = r2.n_prompt_tokens
    manager.free(r1)
    manager.free(r2)

    _assert_prefix_cache_invariants(manager)
    _assert_no_orphans(manager, live_blocks=set())


def test_multi_turn_always_hits_prefix_cache():
    manager = PagedKVCacheManager(model_tag="unit", block_size=4, num_blocks=8)

    prompt = list(range(8))
    request = _make_request("r1", prompt, block_size=4)
    assert manager.allocate_prefill(request) is True

    request.n_computed_tokens = request.n_prompt_tokens
    block_ids, _ = manager.allocate(request, n_new_tokens=4)
    assert block_ids == request.block_ids
    request.generated_ids.extend([100, 101, 102, 103])
    request.n_computed_tokens = request.n_prompt_tokens + request.n_generated_tokens
    manager.free(request)

    prompt = prompt + request.generated_ids
    follow_up = _make_request("r2", prompt, block_size=4)
    assert manager.allocate_prefill(follow_up) is True
    assert follow_up.n_computed_tokens == len(prompt)
    assert follow_up.block_ids == request.block_ids

    follow_up.n_computed_tokens = follow_up.n_prompt_tokens
    block_ids, _ = manager.allocate(follow_up, n_new_tokens=4)
    assert block_ids == follow_up.block_ids
    follow_up.generated_ids.extend([200, 201, 202, 203])
    follow_up.n_computed_tokens = follow_up.n_prompt_tokens + follow_up.n_generated_tokens
    manager.free(follow_up)

    prompt = prompt + follow_up.generated_ids
    r3 = _make_request("r3", prompt, block_size=4)
    assert manager.allocate_prefill(r3) is True
    assert r3.n_computed_tokens == len(prompt)
    assert r3.block_ids == follow_up.block_ids
