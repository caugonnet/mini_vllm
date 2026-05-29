import asyncio
import uuid
from asyncio import Queue
from dataclasses import dataclass
from typing import AsyncIterator, Callable

from transformers import PreTrainedTokenizer

from .struct import Request
from .scheduler import Scheduler
from .model_runner import ModelRunner


@dataclass
class Engine:
    tokenizer: PreTrainedTokenizer
    scheduler: Scheduler
    model_runner: ModelRunner

    async def add_request(self, request_json: dict) -> Callable[[], AsyncIterator[str]]:
        prompt = request_json.get('prompt')
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("request prompt must be a non-empty string")

        prompt_ids = self.tokenizer.encode(prompt)
        request = Request(
            req_id=str(uuid.uuid1()),
            prompt_text=prompt,
            prompt_ids=prompt_ids,
            block_size=self.scheduler.memory_manager.block_size,
            max_tokens=request_json.get('max_tokens', None),
            ignore_eos=request_json.get('ignore_eos', False),
            output_queue=Queue(),
        )

        self.scheduler.add_request(request)

        async def drain_output_queue() -> AsyncIterator[str]:
            while True:
                token = await request.output_queue.get()
                if token is None:
                    break
                yield token

        return drain_output_queue

    async def run(self) -> None:
        while True:
            await asyncio.sleep(0)
            if not self.scheduler.is_on():
                await asyncio.sleep(0.1)
                continue

            batch = self.scheduler.schedule()
            if not batch.req_ids:
                await asyncio.sleep(0)
                continue
            batch_output: dict[str, list[int]] = self.model_runner.execute_batch(batch)
            generated_tokens, finished_requests = self.scheduler.process_output(batch_output)
            for req, tok in generated_tokens:
                req.output_queue.put_nowait(self.tokenizer.decode(tok))
            for req in finished_requests:
                req.output_queue.put_nowait(None)
