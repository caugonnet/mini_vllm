# vLLM-Omni patches

These patches modify the vendored `3rdparty/vllm-omni` submodule. They are kept
here (rather than as a submodule pointer) because the submodule's `origin` is the
upstream repository, which we cannot push to.

## `omni-profiling-nvtx-and-12gb-fixes.patch`

Base submodule commit: `68a1d23e4d48ddd3cf6f1ef2bce641d48f6a0bdf` (`v0.21.0rc2-62-g68a1d23e`).

Contents:

- **NVTX stage annotations** (`vllm_omni/engine/stage_engine_core_proc.py`,
  `vllm_omni/engine/async_omni_engine.py`): when `VLLM_OMNI_NVTX_STAGES=1` is set,
  each stage worker wraps every engine-core step in an NVTX range tagged with its
  logical stage id (e.g. `omni_stage0.step`, `omni_stage1.step`). Because each
  stage runs in its own process/stream, this makes cross-stage overlap directly
  readable in an Nsight Systems timeline and lets
  `mini_vllm.profile.omni_nsys_cli` compute a cross-stage concurrency ratio. The
  stage id is threaded through `spawn_stage_core(..., omni_stage_id=...)`.

- **Single-12GB-GPU Qwen3-TTS config** (`vllm_omni/deploy/qwen3_tts.yaml`): the
  default per-stage `gpu_memory_utilization: 0.3` is tuned for an 80 GB card and
  leaves no KV headroom on 12 GB; bumped talker to `0.5` and code2wav to `0.35`.
  The code2wav stage is switched to `enforce_eager: true` to avoid a nested
  CUDA-graph capture conflict (the vocoder owns its own CUDA graph, so the
  engine-level capture on top fails with `operation not permitted when stream is
  capturing`).

## Apply

```bash
git -C 3rdparty/vllm-omni apply ../../patches/vllm-omni/omni-profiling-nvtx-and-12gb-fixes.patch
```

## Regenerate

```bash
git -C 3rdparty/vllm-omni diff > patches/vllm-omni/omni-profiling-nvtx-and-12gb-fixes.patch
```
