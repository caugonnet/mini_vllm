from dataclasses import dataclass, field
from collections import deque

from .kv_cache import PagedKVCacheManager
from .struct import Request, Batch, BatchOutput


@dataclass
class Scheduler:
    memory_manager: PagedKVCacheManager
    eos_token_id: int
    waiting_requests: deque[Request] = field(default_factory=deque)
    running_requests: dict[str, Request] = field(default_factory=dict)

    def add_request(self, request: Request) -> None:
        self.waiting_requests.append(request)

    def is_on(self) -> bool:
        return bool(self.waiting_requests or self.running_requests)

    def schedule(self) -> Batch:
        while self.waiting_requests:
            front = self.waiting_requests[0]
            if self.memory_manager.allocate_prefill(front):
                self.waiting_requests.popleft()
                self.running_requests[front.req_id] = front
            else:
                break

        batch = Batch()

        for req in self.running_requests.values():
            if req.n_computed_tokens < req.n_prompt_tokens:
                n_scheduled_tokens = req.n_prompt_tokens - req.n_computed_tokens
                input_ids = req.prompt_ids[
                    req.n_computed_tokens : req.n_computed_tokens + n_scheduled_tokens
                ]
            else:
                n_scheduled_tokens = 1
                input_ids = [req.computed_tokens[-1]]

            allocation = self.memory_manager.allocate(req, n_scheduled_tokens)
            if not allocation:
                continue

            batch.req_ids.append(req.req_id)
            batch.input_ids.extend(input_ids)
            batch.context_lens.append(req.n_computed_tokens)
            batch.block_idss.append(req.block_ids)
            batch.query_lens.append(n_scheduled_tokens)
            req.n_computed_tokens += n_scheduled_tokens
        return batch

    def process_output(
        self,
        batch_output: BatchOutput,
    ) -> tuple[list[tuple[Request, int]], list[Request]]:
        generated_tokens: list[tuple[Request, int]] = []
        finished_requests: list[Request] = []

        for req_id, tokens in batch_output.items():
            req = self.running_requests.get(req_id, None)
            assert req is not None
            assert len(tokens) > 0
            if req.n_computed_tokens >= req.n_prompt_tokens:
                new_token = tokens[-1]
                req.generated_ids.append(new_token)
                generated_tokens.append((req, new_token))
                hit_eos = (not req.ignore_eos) and (new_token == self.eos_token_id)
                hit_max_tokens = (
                    req.max_tokens is not None
                    and req.n_generated_tokens >= req.max_tokens
                )
                if hit_eos or hit_max_tokens:
                    self.memory_manager.free(req)
                    self.running_requests.pop(req.req_id)
                    finished_requests.append(req)
        return generated_tokens, finished_requests
