import os
import torch
import math 
import numpy as np
from typing import cast, Any


from itertools import accumulate
from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
    set_current_vllm_config,
)
from vllm.distributed.parallel_state import init_distributed_environment, ensure_model_parallel_initialized
from vllm.model_executor.model_loader import get_model_loader
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.worker.gpu.attn_utils import (
    build_attn_metadata,
    init_attn_backend,
    init_kv_cache,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig
)
from vllm.forward_context import set_forward_context, BatchDescriptor
from vllm.logger import init_logger


from .struct import Batch, BatchOutput

logger = init_logger(__name__)

class ModelRunner:
    def __init__(self,
                 vllm_config: VllmConfig):
        
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        self.vllm_config = vllm_config
        with set_current_vllm_config(vllm_config):
            init_distributed_environment()
            ensure_model_parallel_initialized(1, 1)
            model_loader = get_model_loader(vllm_config.load_config)
            self.model = model_loader.load_model(
                vllm_config=vllm_config, model_config=vllm_config.model_config
            )
        from vllm.platforms import current_platform
        self.device = current_platform.device_type
        logger.info(f"loaded {vllm_config.model_config.model}, takes {torch.cuda.device_memory_used(device = self.device) / 1e9:.3f} Gb")
        self.block_size = vllm_config.cache_config.block_size
        with set_current_vllm_config(vllm_config):
            self._init_kv_cache()
        self.batch_descriptors: set[BatchDescriptor] = set()
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs

        self.input_ids_buf = torch.empty(
            self.max_num_tokens, dtype=torch.int64, device=self.device
        )
        self.position_ids_buf = torch.empty(
            self.max_num_tokens, dtype=torch.int64, device=self.device
        )
        compilation_config = vllm_config.compilation_config
        logger.debug(
            "CUDA graph config: mode=%s, capture_sizes=%s, max_capture_size=%s",
            compilation_config.cudagraph_mode,
            compilation_config.cudagraph_capture_sizes,
            compilation_config.max_cudagraph_capture_size,
        )
        
    def _compute_available_memory(self, utilization: float):
        props = torch.cuda.get_device_properties(device = self.device)
        total_memory = props.total_memory
        used_memory = torch.cuda.device_memory_used(device = self.device)
        assert total_memory * utilization > used_memory, f"no memory available, {total_memory}, {used_memory}"
        return int(total_memory * utilization - used_memory) 
        
    def _init_kv_cache(self):

        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec
        available_memory = self._compute_available_memory(self.vllm_config.cache_config.gpu_memory_utilization)
        kv_cache_configs = get_kv_cache_configs(self.vllm_config, [kv_cache_spec], [available_memory])
        self.kv_cache_config: KVCacheConfig = kv_cache_configs[0]
        logger.info(f'Allocate {available_memory / 1e9:.3f}Gb to KV Cache, #Blocks: {self.kv_cache_config.num_blocks}, #tokens {self.kv_cache_config.num_blocks * self.block_size}')
        self.attn_backends, self.attn_groups, _ = init_attn_backend(
            self.kv_cache_config, 
            self.vllm_config, 
            self.device
        )
        self.kv_caches = []
        init_kv_cache(
            self.kv_caches,
            forward_context = self.vllm_config.compilation_config.static_forward_context, 
            kv_cache_config = self.kv_cache_config, 
            attn_backends = self.attn_backends,
            device = self.device,
            cache_dtype = self.vllm_config.cache_config.cache_dtype,
        )
        
    def _build_attention_meta(
        self,
        batch: Batch
    ):
        num_reqs = len(batch.query_lens)
        assert num_reqs <= self.max_num_reqs, (num_reqs, self.max_num_reqs)

        query_start_loc_lst = [0] + list(accumulate(batch.query_lens))
        query_start_loc_cpu = torch.tensor(query_start_loc_lst, dtype = torch.int32, device = 'cpu')
        query_start_loc_gpu = query_start_loc_cpu.to(self.device)
        _seq_lens_cpu = torch.tensor(batch.seq_lens, dtype = torch.int32, device = 'cpu')
        seq_lens = _seq_lens_cpu.to(self.device)
        max_seq_len = max(batch.seq_lens)
        max_query_len = max(batch.query_lens)
        num_computed_tokens_cpu = torch.tensor(batch.context_lens, dtype = torch.int32, device = 'cpu')
        num_tokens = sum(batch.query_lens)
        max_num_blocks = int(math.ceil(max_seq_len/self.block_size))
        block_tables = np.full(shape = (num_reqs, max_num_blocks), fill_value = -1, dtype = np.int32)
        for i in range(len(batch.block_idss)):
            block_ids = batch.block_idss[i]
            block_tables[i, :len(block_ids)] = block_ids 
        block_tables = torch.from_numpy(block_tables).to(dtype = torch.int32).to(self.device)
        slot_mappings = []
        assert len(batch.context_lens) == len(batch.query_lens) == len(batch.block_idss)
        for seq_len, ctx_len, block_ids in zip(batch.seq_lens, batch.context_lens, batch.block_idss):
            slot_mapping = [
                block_ids[idx // self.block_size] * self.block_size + idx % self.block_size for idx in range(ctx_len, seq_len)
            ]
            slot_mappings.extend(slot_mapping)
        slot_mappings = torch.tensor(slot_mappings, dtype = torch.int64, device = self.device)
        kv_cache_groups = self.kv_cache_config.kv_cache_groups
        # validate block coverage before building slot mapping
        for seq_len, block_ids in zip(batch.seq_lens, batch.block_idss):
            need_blocks = (seq_len + self.block_size - 1) // self.block_size
            assert len(block_ids) >= need_blocks, (len(block_ids), need_blocks, seq_len)

        block_tables_by_group = tuple(block_tables for _ in kv_cache_groups)
        slot_mappings_by_group = slot_mappings.unsqueeze(0).expand(
            len(kv_cache_groups), -1
        )
        return build_attn_metadata(
            attn_groups=self.attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=query_start_loc_gpu,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables_by_group,
            slot_mappings=slot_mappings_by_group,
            kv_cache_config=self.kv_cache_config,
            seq_lens_cpu_upper_bound=_seq_lens_cpu,
        )

    def _can_use_piecewise_cudagraph(self, num_tokens: int) -> bool:
        compilation_config = self.vllm_config.compilation_config
        cudagraph_mode = compilation_config.cudagraph_mode
        max_capture_size = compilation_config.max_cudagraph_capture_size
        return (
            cudagraph_mode is not None
            and cudagraph_mode.has_piecewise_cudagraphs()
            and max_capture_size is not None
            and num_tokens <= max_capture_size
        )

    def _get_padded_num_tokens(self, num_tokens: int) -> int:
        if (
            not self._can_use_piecewise_cudagraph(num_tokens)
        ):
            return num_tokens
        capture_sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes or []
        for size in capture_sizes:
            if num_tokens <= size:
                return size
        return num_tokens

    def _prepare_inputs(self, batch: Batch, num_tokens_padded: int):
        num_tokens = len(batch.input_ids)
        assert num_tokens == sum(batch.query_lens), (num_tokens, batch.query_lens)
        assert num_tokens <= num_tokens_padded <= self.max_num_tokens, (
            num_tokens,
            num_tokens_padded,
            self.max_num_tokens,
        )

        input_ids = torch.as_tensor(
            batch.input_ids, dtype=torch.int64, device=self.device
        )
        positions = torch.as_tensor(
            batch.position_ids, dtype=torch.int64, device=self.device
        )
        self.input_ids_buf[:num_tokens].copy_(input_ids)
        self.position_ids_buf[:num_tokens].copy_(positions)

        if num_tokens_padded > num_tokens:
            self.input_ids_buf[num_tokens:num_tokens_padded].fill_(0)
            self.position_ids_buf[num_tokens:num_tokens_padded].fill_(0)

        return (
            self.input_ids_buf[:num_tokens_padded],
            self.position_ids_buf[:num_tokens_padded],
        )

    def _get_cudagraph_context(
        self,
        num_tokens: int,
        requested_runtime_mode: CUDAGraphMode | None,
    ) -> tuple[CUDAGraphMode, BatchDescriptor | None, int]:
        can_use_piecewise_cudagraph = self._can_use_piecewise_cudagraph(num_tokens)
        num_tokens_padded = self._get_padded_num_tokens(num_tokens)
        batch_descriptor = BatchDescriptor(num_tokens=num_tokens_padded)

        if requested_runtime_mode is not None:
            if requested_runtime_mode == CUDAGraphMode.NONE:
                descriptor = BatchDescriptor(num_tokens=num_tokens)
                logger.debug(
                    "CUDA graph dispatch: requested=%s, tokens=%d, padded=%d "
                    "-> NONE",
                    requested_runtime_mode,
                    num_tokens,
                    num_tokens_padded,
                )
                return CUDAGraphMode.NONE, descriptor, num_tokens
            if (
                requested_runtime_mode == CUDAGraphMode.PIECEWISE
                and not can_use_piecewise_cudagraph
            ):
                logger.debug(
                    "CUDA graph dispatch: requested=%s, eligible=%s, "
                    "tokens=%d, padded=%d -> NONE",
                    requested_runtime_mode,
                    can_use_piecewise_cudagraph,
                    num_tokens,
                    num_tokens_padded,
                )
                return CUDAGraphMode.NONE, None, num_tokens
            logger.debug(
                "CUDA graph dispatch: requested=%s, tokens=%d, padded=%d, "
                "descriptor=%s",
                requested_runtime_mode,
                num_tokens,
                num_tokens_padded,
                batch_descriptor,
            )
            return requested_runtime_mode, batch_descriptor, num_tokens_padded

        if not can_use_piecewise_cudagraph:
            logger.debug(
                "CUDA graph dispatch: ineligible, tokens=%d, max_capture_size=%s, "
                "mode=%s -> NONE",
                num_tokens,
                self.vllm_config.compilation_config.max_cudagraph_capture_size,
                self.vllm_config.compilation_config.cudagraph_mode,
            )
            return CUDAGraphMode.NONE, None, num_tokens

        if batch_descriptor not in self.batch_descriptors:
            self.batch_descriptors.add(batch_descriptor)
            logger.debug(
                "CUDA graph dispatch: first run for %s, tokens=%d, padded=%d "
                "-> warmup NONE",
                batch_descriptor,
                num_tokens,
                num_tokens_padded,
            )
            return CUDAGraphMode.NONE, batch_descriptor, num_tokens_padded

        logger.debug(
            "CUDA graph dispatch: tokens=%d, padded=%d, descriptor=%s -> PIECEWISE",
            num_tokens,
            num_tokens_padded,
            batch_descriptor,
        )
        return CUDAGraphMode.PIECEWISE, batch_descriptor, num_tokens_padded

    @property
    def num_blocks(self):
        return self.kv_cache_config.num_blocks

    def execute_batch(
        self,
        batch: Batch,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
    ) -> BatchOutput:
        num_tokens = sum(batch.query_lens)
        (
            cudagraph_runtime_mode,
            batch_descriptor,
            num_tokens_padded,
        ) = self._get_cudagraph_context(num_tokens, cudagraph_runtime_mode)

        attn_meta = self._build_attention_meta(batch)
        input_ids, positions = self._prepare_inputs(batch, num_tokens_padded)
        generated_tokens = {}
        with torch.inference_mode(), set_current_vllm_config(self.vllm_config):
            with set_forward_context(attn_metadata=attn_meta, 
                                     vllm_config = self.vllm_config,
                                     num_tokens = num_tokens_padded,
                                     cudagraph_runtime_mode= cudagraph_runtime_mode,
                                     batch_descriptor = batch_descriptor):
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions
                )
                logits = self.model.compute_logits(hidden_states)
                tokens = torch.argmax(logits, dim = -1).cpu().tolist()
        
        prev = 0
        for idx, req_id in zip(accumulate(batch.query_lens), batch.req_ids):
            generated_tokens[req_id] = tokens[prev:idx]
            prev = idx
        return generated_tokens
