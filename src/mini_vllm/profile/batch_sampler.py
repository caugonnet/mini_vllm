from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from typing import Iterable, Optional

from mini_vllm.struct import Batch


@dataclass
class RequestSpec:
    context_len: int
    query_len: int
    req_id: Optional[str] = None
    block_ids: Optional[list[int]] = None


@dataclass
class BatchSpec:
    name: str
    batch_type: str  # "prefill", "decode", "mixed"
    requests: list[RequestSpec]


class BatchSampler:
    def __init__(self, block_size: int, num_blocks: int, seed: int = 0):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.seed = seed

    def load_specs(self, path: str) -> list[BatchSpec]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = self._load_yaml(path)
        return self._parse_specs(raw)

    def build_batches(self, specs: Iterable[BatchSpec]) -> list[Batch]:
        batches = []
        for spec in specs:
            batches.append(self._build_batch(spec))
        return batches

    def _build_batch(self, spec: BatchSpec) -> Batch:
        block_cursor = 0
        req_ids = []
        input_ids = []
        context_lens = []
        query_lens = []
        block_idss = []

        for i, req in enumerate(spec.requests):
            req_id = req.req_id or f"{spec.name}-{i}"
            req_ids.append(req_id)
            context_lens.append(req.context_len)
            query_lens.append(req.query_len)

            seq_len = req.context_len + req.query_len
            needed_blocks = int(math.ceil(seq_len / self.block_size))
            if req.block_ids is not None:
                if len(req.block_ids) < needed_blocks:
                    raise ValueError(
                        f"request {req_id} needs {needed_blocks} blocks, "
                        f"got {len(req.block_ids)}"
                    )
                block_ids = req.block_ids[:needed_blocks]
            else:
                if block_cursor + needed_blocks > self.num_blocks:
                    raise ValueError(
                        f"not enough blocks for batch {spec.name}: "
                        f"need {block_cursor + needed_blocks}, have {self.num_blocks}"
                    )
                block_ids = list(range(block_cursor, block_cursor + needed_blocks))
                block_cursor += needed_blocks
            block_idss.append(block_ids)

            # input_ids do not affect profiling; use zeros with correct length
            input_ids.extend([0] * req.query_len)

        return Batch(
            req_ids=req_ids,
            input_ids=input_ids,
            context_lens=context_lens,
            query_lens=query_lens,
            block_idss=block_idss,
        )

    def _parse_specs(self, raw: object) -> list[BatchSpec]:
        if not isinstance(raw, list):
            raise ValueError("batch config must be a list of batch specs")
        specs: list[BatchSpec] = []
        for item in raw:
            name = str(item.get("name", "batch"))
            batch_type = str(item.get("type", "mixed"))
            requests_raw = item.get("requests", [])
            requests: list[RequestSpec] = []
            for r in requests_raw:
                requests.append(
                    RequestSpec(
                        context_len=int(r["context_len"]),
                        query_len=int(r["query_len"]),
                        req_id=r.get("req_id"),
                        block_ids=r.get("block_ids"),
                    )
                )
            specs.append(BatchSpec(name=name, batch_type=batch_type, requests=requests))
        return specs

    def _load_yaml(self, path: str) -> object:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "YAML config requested but PyYAML is not installed. "
                "Install pyyaml or use JSON."
            ) from exc
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
