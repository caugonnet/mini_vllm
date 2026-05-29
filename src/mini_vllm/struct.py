from asyncio.queues import Queue
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Request:
    req_id: str
    prompt_text: str
    prompt_ids: list[int] = field(default_factory=list)
    block_size: int | None = None
    max_tokens: int | None = None
    ignore_eos: bool = True

    output_queue: Queue = field(default_factory=Queue)

    # Number of prompt/generated tokens already computed by the model.
    n_computed_tokens: int = 0
    blocks: Optional[list["KVCacheBlock"]] = None
    generated_ids: list[int] = field(default_factory=list, init=False)

    @property
    def finished_prefill(self) -> bool:
        return self.n_computed_tokens >= len(self.prompt_ids)

    @property
    def n_prompt_tokens(self) -> int:
        return len(self.prompt_ids)

    @property
    def n_generated_tokens(self) -> int:
        return len(self.generated_ids)

    @property
    def n_blocks(self) -> int:
        return len(self.blocks or [])

    @property
    def block_ids(self) -> list[int]:
        return [block.idx for block in self.blocks or []]

    @property
    def computed_tokens(self) -> list[int]:
        return (self.prompt_ids + self.generated_ids)[: self.n_computed_tokens]

    def validate_invariants(self) -> None:
        if self.n_computed_tokens < len(self.prompt_ids):
            assert len(self.generated_ids) == 0
        else:
            total_tokens = len(self.prompt_ids) + len(self.generated_ids)
            assert self.n_computed_tokens <= total_tokens

    @property
    def check_invariants(self) -> None:
        self.validate_invariants()


@dataclass
class Batch:
    req_ids: list[str] = field(default_factory=list)
    input_ids: list[int] = field(default_factory=list)
    context_lens: list[int] = field(default_factory=list)
    query_lens: list[int] = field(default_factory=list)
    block_idss: list[list[int]] = field(default_factory=list)

    @property
    def position_ids(self) -> list[int]:
        return [
            pos
            for context_len, query_len in zip(self.context_lens, self.query_lens)
            for pos in range(context_len, context_len + query_len)
        ]

    @property
    def seq_lens(self) -> list[int]:
        return [cl + ql for cl, ql in zip(self.context_lens, self.query_lens)]


BatchOutput = dict[str, list[int]]


@dataclass
class KVCacheBlock:
    idx: int
    reference_cnt: int = 0
    _hash: Optional[str] = None
    parent: Optional["KVCacheBlock"] = None
    next_blocks: dict[str, "KVCacheBlock"] = field(default_factory=dict)
    prev: Optional["KVCacheBlock"] = None
    next: Optional["KVCacheBlock"] = None


@dataclass
class Config:
    model_name: str
    max_memory_utilization: float
    block_size: int = 16
    max_num_batched_tokens: Optional[int] = None
