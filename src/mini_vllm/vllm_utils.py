import sys

from transformers import AutoTokenizer

from vllm.config import (
    CacheConfig,
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)

from .struct import Config 
from .scheduler import Scheduler
from .engine import Engine 
from .model_runner import ModelRunner
from .kv_cache import PagedKVCacheManager


def _cudagraph_capture_sizes(max_num_batched_tokens: int) -> list[int]:
    sizes = [1, 2, 4, 8, 16, 32, 64, 128, 512, 1024, 2048]
    sizes = [size for size in sizes if size <= max_num_batched_tokens]
    if max_num_batched_tokens not in sizes:
        sizes.append(max_num_batched_tokens)
    return sorted(set(sizes))


def _patch_vllm_registry_subprocess_for_pyarrow() -> None:
    """Preload pyarrow in vLLM's model-inspection child process.

    In this environment, vLLM's registry subprocess can segfault in
    pyarrow/libarrow's allocator thread when the model interfaces are imported.
    Importing pyarrow before the registry avoids that native crash, so make the
    subprocess do the same.
    """
    try:
        import pyarrow  # noqa: F401
        import vllm.model_executor.models.registry as registry
    except Exception:
        return

    registry._SUBPROCESS_COMMAND = [
        sys.executable,
        "-c",
        (
            "import pyarrow; "
            "import runpy; "
            "runpy.run_module('vllm.model_executor.models.registry', "
            "run_name='__main__')"
        ),
    ]


def get_vllm_config(
    config: Config
):
    max_num_batched_tokens = (
        config.max_num_batched_tokens
        if config.max_num_batched_tokens is not None
        else 2048
    )
    _patch_vllm_registry_subprocess_for_pyarrow()

    model_config = ModelConfig(
        model=config.model_name,
        dtype="float16",
        seed=42,
    )
    scheduler_config = SchedulerConfig(
        max_num_seqs=10,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=2048,
        is_encoder_decoder=model_config.is_encoder_decoder,
    )
    cache_config = CacheConfig(
        block_size=config.block_size,
        gpu_memory_utilization=config.max_memory_utilization,
        cache_dtype="auto",
    )
    cudagraph_capture_sizes = _cudagraph_capture_sizes(max_num_batched_tokens)
    compilation_config = CompilationConfig(
        mode=CompilationMode.VLLM_COMPILE,
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        cudagraph_capture_sizes=cudagraph_capture_sizes,
        max_cudagraph_capture_size=cudagraph_capture_sizes[-1],
    )
    parallel_config = ParallelConfig()
    vllm_config = VllmConfig(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config
    )
    return vllm_config


def get_engine_from_vllm(
    config: Config 
):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    vllm_config = get_vllm_config(config)
    model_runner = ModelRunner(vllm_config)
    
    memory_manager = PagedKVCacheManager(
        config.block_size,
        num_blocks=model_runner.num_blocks
    )
    
    scheduler = Scheduler(memory_manager = memory_manager, 
                          eos_token_id = tokenizer.eos_token_id)

    engine = Engine(tokenizer, scheduler, model_runner)
    
    return engine 
    
# vllm_config = get_vllm_config()
