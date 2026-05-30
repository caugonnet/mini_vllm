# mini-vLLM

A small experimental inference engine built on top of selected vLLM internals.
It includes:

- async request scheduling
- paged KV-cache block management with prefix reuse
- a minimal offline generation wrapper
- a small FastAPI completion endpoint
- profiling utilities for batch latency, power, and CUDA graph experiments

This repository keeps the CUDA/vLLM runtime separate from the base package.
Install vLLM deliberately for your CUDA stack instead of letting the mini-vLLM
package resolver choose a torch wheel for you.

## Install

```bash
git submodule update --init --recursive
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip "setuptools<82" wheel

CUDA_HOME=/usr/local/cuda-13.0 \
PATH=/usr/local/cuda-13.0/bin:$PATH \
CC=/usr/bin/gcc-13 \
CXX=/usr/bin/g++-13 \
CUDAHOSTCXX=/usr/bin/g++-13 \
CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13" \
MAX_JOBS=8 \
python -m pip install -e 3rdparty/vllm # ~30 minutes
```

The pinned vLLM submodule currently tracks the latest release tag, `v0.21.0`.
That release pins `torch==2.11.0` in its CUDA requirements. Installing vLLM
normally may downgrade or replace an existing torch install to match that pin.
CUDA 13 builds also need a C++17-capable host compiler. On this workstation,
the default `g++` is too old for vLLM/PyTorch CUDA extensions, so the install
command above pins CMake/NVCC to `/usr/bin/g++-13`.

If `pip install -e 3rdparty/vllm` tries to resolve an incompatible torch wheel,
install the torch/vLLM stack for your machine first, then install this package
without the `runtime` extra:

```bash
pip install -e ".[server,profile,test]"
```

Use `.[runtime]` only when you want pip to resolve `torch`, `transformers`, and
`vllm` from your configured package indexes.

### Optional: vLLM-Omni (multi-stage omni experiments)

`vllm-omni` is a second submodule that reuses the **same** `vllm` you built
above. The `3rdparty/vllm` and `3rdparty/vllm-omni` submodules both track the
`v0.21.0` line, and `vllm-omni` neither pins nor pulls `vllm`/`torch` in its
requirements (it only emits a `RuntimeWarning` on a major/minor mismatch). So
install it on top of the existing editable vLLM:

```bash
# Reuse the deliberate vllm/torch build; do NOT let pip re-resolve them.
python -m pip install -e 3rdparty/vllm-omni --no-deps

# Then add vllm-omni's own deps (audits the hard pins it brings, e.g.
# diffusers==0.38.0, accelerate==1.12.0, fa3-fwd==0.0.3, onnxruntime).
python -m pip install -r 3rdparty/vllm-omni/requirements/cuda.txt

# Extra helpers used by the omni profiling experiments.
pip install -e ".[omni]"
```

Note: `vllm-omni` monkeypatches a few `vllm` platform modules at import; this is
expected on the matching `0.21` build.

The omni profiling experiments rely on small local edits to the `vllm-omni`
submodule (NVTX stage annotations + a single-12GB-GPU Qwen3-TTS config). They are
kept as a patch because the submodule's `origin` is upstream:

```bash
# run from the repo root
git -C 3rdparty/vllm-omni apply ../../patches/vllm-omni/omni-profiling-nvtx-and-12gb-fixes.patch
```

See `patches/vllm-omni/README.md` for details.

## Run profiling
```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_USE_STANDALONE_COMPILE=0 \
PYTHONPATH="$PWD/3rdparty/vllm:$PWD/src" \
python \
  -m mini_vllm.profile.cuda_graph_cli \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --batch_config mini_vllm/profile/batch_config_batch_size_sweep.json \
  --device_index 0 \
  --output cuda_graph_batch_size_sweep.debug.phases.jsonl
```

This will dump the profiling result into `cuda_graph_batch_size_sweep.debug.phases.jsonl`.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 nsys profile \
  --trace=cuda,nvtx \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=repeat-shutdown:2 \
  --gpu-metrics-devices=0 \
  --output nsys_cuda_graph_profile \
  --force-overwrite=true \
python -m mini_vllm.profile.cuda_graph_cli \
  --model_name Qwen/Qwen2.5-7B-Instruct \
  --batch_config mini_vllm/profile/batch_config_test.json \
  --output cuda_graph_profile.jsonl \
  --no_progress
```
This will generate `nsys_cuda_graph_profile.{1,2}.nsys-rep`, `1` for the non-cuda graph capture, `2` is for the cuda graph capture. 

**Down below haven't been checked to work**

## Run Offline Inference

```python
from mini_vllm.offline_inference import OfflineLLM

llm = OfflineLLM("facebook/opt-125m")
print(llm.generate("Carnegie Mellon University is known for ", max_tokens=64))
```

## Run The API Server

```bash
MINI_VLLM_MODEL_NAME=facebook/opt-125m \
python3 -m mini_vllm.api_server
```

Then:

```bash
curl http://localhost:8000/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Hello","max_tokens":64,"ignore_eos":false}'
```

## Tests

The KV-cache tests are CPU-safe:

```bash
pytest tests/test_kv_cache.py
```

The offline and API tests require CUDA, vLLM, and model downloads.
