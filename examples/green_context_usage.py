from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mini_vllm.green_context import (
    create_green_context_stream,
    current_green_context,
    green_context,
)


def timed_matmul(
    label: str,
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    ctx,
    warmup_iters: int = 5,
    timed_iters: int = 50,
) -> torch.Tensor:
    out = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)

    for _ in range(warmup_iters):
        torch.mm(x, w, out=out)
    ctx.stream.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(ctx.stream)
    for _ in range(timed_iters):
        torch.mm(x, w, out=out)
    end.record(ctx.stream)
    end.synchronize()

    avg_ms = start.elapsed_time(end) / timed_iters
    flops_per_matmul = 2 * x.shape[0] * x.shape[1] * w.shape[1]
    tflops = flops_per_matmul / (avg_ms / 1_000.0) / 1e12
    print(
        f"{label} timing: {ctx.num_sms}/{ctx.total_sms} SMs, "
        f"avg={avg_ms:.4f} ms, {tflops:.2f} TFLOP/s"
    )
    return out


@green_context(sm_percent=25.0, device=0)
def decorated_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ctx = current_green_context()
    assert ctx is not None
    print(
        f"decorator: cuda:{ctx.device_id}, "
        f"{ctx.num_sms}/{ctx.total_sms} SMs, stream={ctx.stream.cuda_stream}"
    )
    return timed_matmul("decorator", x, w, ctx=ctx)


def context_manager_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    with green_context(sm_percent=50.0, device=x.device) as ctx:
        print(
            f"context manager: cuda:{ctx.device_id}, "
            f"{ctx.num_sms}/{ctx.total_sms} SMs, stream={ctx.stream.cuda_stream}"
        )
        return timed_matmul("context manager", x, w, ctx=ctx)


def explicit_stream_matmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ctx = create_green_context_stream(sm_percent=10.0, device=x.device)
    print(
        f"explicit stream: cuda:{ctx.device_id}, "
        f"{ctx.num_sms}/{ctx.total_sms} SMs, stream={ctx.stream.cuda_stream}"
    )
    with ctx.activate():
        return timed_matmul("explicit stream", x, w, ctx=ctx)


def benchmark_matmul_by_sm_count(
    x: torch.Tensor,
    w: torch.Tensor,
    *,
    sm_counts: Sequence[int] | None = None,
    warmup_iters: int = 5,
    timed_iters: int = 50,
) -> None:
    if x.ndim != 2 or w.ndim != 2:
        raise ValueError("benchmark_matmul_by_sm_count expects 2D tensors")
    if x.shape[1] != w.shape[0]:
        raise ValueError(
            f"incompatible matmul shapes: {tuple(x.shape)} x {tuple(w.shape)}"
        )

    device_id = x.device.index
    if device_id is None:
        device_id = torch.cuda.current_device()
    total_sms = torch.cuda.get_device_properties(device_id).multi_processor_count

    if sm_counts is None:
        sm_counts = (
            max(1, total_sms // 8),
            max(1, total_sms // 4),
            max(1, total_sms // 2),
            total_sms,
        )

    requested_sms = sorted(
        {max(1, min(total_sms, int(num_sms))) for num_sms in sm_counts}
    )
    out = torch.empty((x.shape[0], w.shape[1]), device=x.device, dtype=x.dtype)

    print(
        "\nmatmul timing "
        f"shape={tuple(x.shape)}x{tuple(w.shape)}, "
        f"dtype={x.dtype}, timed_iters={timed_iters}"
    )
    print(f"{'requested_sms':>13} {'actual_sms':>10} {'avg_ms':>10} {'tflops':>10}")

    flops_per_matmul = 2 * x.shape[0] * x.shape[1] * w.shape[1]
    for num_sms in requested_sms:
        sm_percent = 100.0 * num_sms / total_sms
        ctx = create_green_context_stream(sm_percent=sm_percent, device=x.device)

        with ctx.activate():
            for _ in range(warmup_iters):
                torch.mm(x, w, out=out)
            ctx.stream.synchronize()

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(ctx.stream)
            for _ in range(timed_iters):
                torch.mm(x, w, out=out)
            end.record(ctx.stream)
            end.synchronize()

        avg_ms = start.elapsed_time(end) / timed_iters
        tflops = flops_per_matmul / (avg_ms / 1_000.0) / 1e12
        print(f"{num_sms:13d} {ctx.num_sms:10d} {avg_ms:10.4f} {tflops:10.2f}")


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for green context examples")

    device = torch.device("cuda:0")
    x = torch.randn((1024, 1024), device=device)
    w = torch.randn((1024, 1024), device=device)

    y0 = decorated_matmul(x, w)
    y1 = context_manager_matmul(x, w)
    y2 = explicit_stream_matmul(x, w)

    torch.cuda.synchronize(device)
    print(y0.shape, y1.shape, y2.shape)
    benchmark_matmul_by_sm_count(x, w)


if __name__ == "__main__":
    main()
