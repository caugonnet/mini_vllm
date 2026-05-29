import asyncio
from typing import Optional

from .engine import Engine
from .struct import Config
from .vllm_utils import get_engine_from_vllm

class OfflineLLM:
    engine: Engine

    def __init__(self,
                 model_name: str,
                 max_memory_utilization: float = 0.8):
        config = Config(
            model_name=model_name,
            max_memory_utilization=max_memory_utilization,
        )
        self.engine = get_engine_from_vllm(config)
        self._engine_task: Optional[asyncio.Task] = None

    async def _ensure_running(self):
        if self._engine_task is None:
            self._engine_task = asyncio.create_task(self.engine.run())

    async def generate_streaming(self, prompt: str, **kwargs):
        await self._ensure_running()
        request_json = dict(kwargs)
        request_json["prompt"] = prompt
        token_gen = await self.engine.add_request(request_json)
        async for text in token_gen():
            yield text

    async def generate_async(self, prompt: str, **kwargs) -> str:
        tokens = []
        async for text in self.generate_streaming(prompt, **kwargs):
            tokens.append(text)
        return "".join(tokens)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.generate_async(prompt, **kwargs))
        raise RuntimeError("generate() cannot be called from an active event loop; use generate_async().")

    async def aclose(self):
        if self._engine_task is not None:
            self._engine_task.cancel()
            try:
                await self._engine_task
            except asyncio.CancelledError:
                pass
        
