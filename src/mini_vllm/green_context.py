from __future__ import annotations

import functools
import inspect
import math
import threading
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, ParamSpec, TypeVar


P = ParamSpec("P")
R = TypeVar("R")

_CURRENT_GREEN_CONTEXT: ContextVar["GreenContextHandle | None"] = ContextVar(
    "mini_vllm_current_green_context",
    default=None,
)


class GreenContextUnavailable(RuntimeError):
    """Raised when PyTorch CUDA Green Context support is unavailable."""


@dataclass(frozen=True)
class GreenContextHandle:
    """Runtime handle for a PyTorch CUDA Green Context and its stream."""

    green_context: Any
    stream: Any
    device_id: int
    num_sms: int
    total_sms: int
    sm_percent: float
    enabled: bool = True
    reason: str | None = None
    _torch: Any = field(default=None, repr=False, compare=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @property
    def device(self) -> str:
        return f"cuda:{self.device_id}" if self.device_id >= 0 else "cuda"

    def activate(self) -> "_GreenContextActivation":
        return _GreenContextActivation(self)


class _GreenContextActivation:
    def __init__(self, handle: GreenContextHandle):
        self.handle = handle
        self._stream_context: Any = None
        self._token: Any = None
        self._active = False

    def __enter__(self) -> GreenContextHandle:
        self.handle._lock.acquire()
        self._token = _CURRENT_GREEN_CONTEXT.set(self.handle)

        if not self.handle.enabled:
            self._active = True
            return self.handle

        try:
            self.handle.green_context.set_context()
            self._stream_context = self.handle._torch.cuda.stream(self.handle.stream)
            self._stream_context.__enter__()
            self._active = True
            return self.handle
        except BaseException:
            _CURRENT_GREEN_CONTEXT.reset(self._token)
            self.handle._lock.release()
            raise

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        try:
            if self.handle.enabled and self._active:
                if self._stream_context is not None:
                    self._stream_context.__exit__(exc_type, exc, tb)
                self.handle.green_context.pop_context()
        finally:
            if self._token is not None:
                _CURRENT_GREEN_CONTEXT.reset(self._token)
            self.handle._lock.release()
        return False


class TorchGreenContext:
    """Decorator/context manager backed by ``torch.cuda.GreenContext``.

    ``sm_percent`` is a percentage, so pass ``50`` for half of the device SMs.
    The percentage is converted to a concrete SM count with ``ceil`` and then
    passed to PyTorch's ``GreenContext.create``.
    """

    def __init__(
        self,
        *,
        sm_percent: float = 100.0,
        device: int | str | Any | None = None,
        strict: bool = True,
        synchronize: bool = False,
    ):
        self.sm_percent = sm_percent
        self.device = device
        self.strict = strict
        self.synchronize = synchronize
        self._handle: GreenContextHandle | None = None
        self._lock = threading.RLock()

    def __call__(self, func: Callable[P, R]) -> Callable[P, R]:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                with self:
                    return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            with self:
                return func(*args, **kwargs)

        return wrapper

    def __enter__(self) -> GreenContextHandle:
        activation = self.create_stream().activate()
        self._activation = activation
        return activation.__enter__()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        handle = self.create_stream()
        activation = getattr(self, "_activation", None)
        if activation is None:
            return False
        try:
            if self.synchronize and handle.enabled:
                handle.stream.synchronize()
            return activation.__exit__(exc_type, exc, tb)
        finally:
            self._activation = None

    def create_stream(self) -> GreenContextHandle:
        if self._handle is None:
            self._handle = _create_green_context_stream(
                sm_percent=self.sm_percent,
                device=self.device,
                strict=self.strict,
                lock=self._lock,
            )
        return self._handle


def green_context(
    func: Callable[P, R] | None = None,
    *,
    sm_percent: float = 100.0,
    device: int | str | Any | None = None,
    strict: bool = True,
    synchronize: bool = False,
) -> TorchGreenContext | Callable[P, R]:
    """Decorate a function or create a context manager using a green-context stream."""

    manager = TorchGreenContext(
        sm_percent=sm_percent,
        device=device,
        strict=strict,
        synchronize=synchronize,
    )
    if func is None:
        return manager
    return manager(func)


def create_green_context_stream(
    *,
    sm_percent: float = 100.0,
    device: int | str | Any | None = None,
    strict: bool = True,
) -> GreenContextHandle:
    """Create a PyTorch green context and return the stream generated from it."""

    return _create_green_context_stream(
        sm_percent=sm_percent,
        device=device,
        strict=strict,
        lock=threading.RLock(),
    )


def current_green_context() -> GreenContextHandle | None:
    return _CURRENT_GREEN_CONTEXT.get()


def _create_green_context_stream(
    *,
    sm_percent: float,
    device: int | str | Any | None,
    strict: bool,
    lock: threading.RLock,
) -> GreenContextHandle:
    try:
        import torch
    except ImportError as exc:
        if strict:
            raise GreenContextUnavailable("PyTorch is required for CUDA Green Contexts") from exc
        return _disabled_handle(str(exc), lock)

    try:
        device_id = _resolve_device_id(torch, device)
        num_sms, total_sms = _resolve_num_sms(torch, device_id, sm_percent)
        _ensure_primary_context(torch, device_id)

        green_ctx_type = getattr(torch.cuda, "GreenContext", None)
        if green_ctx_type is None:
            raise GreenContextUnavailable("torch.cuda.GreenContext is not available")

        # Keep construction on PyTorch's green-context API, then derive the
        # stream from that context so downstream PyTorch ops can use it.
        green_ctx = green_ctx_type.create(num_sms, device_id)
        green_ctx.set_context()
        try:
            stream = green_ctx.Stream()
        finally:
            green_ctx.pop_context()

        return GreenContextHandle(
            green_context=green_ctx,
            stream=stream,
            device_id=device_id,
            num_sms=num_sms,
            total_sms=total_sms,
            sm_percent=sm_percent,
            _torch=torch,
            _lock=lock,
        )
    except Exception as exc:
        if strict:
            raise
        reason = f"{type(exc).__name__}: {exc}"
        return _disabled_handle(reason, lock, torch=torch)


def _disabled_handle(
    reason: str,
    lock: threading.RLock,
    torch: Any | None = None,
) -> GreenContextHandle:
    return GreenContextHandle(
        green_context=None,
        stream=None,
        device_id=-1,
        num_sms=0,
        total_sms=0,
        sm_percent=0.0,
        enabled=False,
        reason=reason,
        _torch=torch,
        _lock=lock,
    )


def _resolve_device_id(torch: Any, device: int | str | Any | None) -> int:
    if not torch.cuda.is_available():
        raise GreenContextUnavailable("CUDA is not available")

    if device is None:
        return int(torch.cuda.current_device())

    if isinstance(device, int):
        return device

    torch_device = torch.device(device)
    if torch_device.type != "cuda":
        raise ValueError(f"green contexts require a CUDA device, got {torch_device}")
    if torch_device.index is None:
        return int(torch.cuda.current_device())
    return int(torch_device.index)


def _resolve_num_sms(torch: Any, device_id: int, sm_percent: float) -> tuple[int, int]:
    if not math.isfinite(sm_percent) or sm_percent <= 0.0 or sm_percent > 100.0:
        raise ValueError("sm_percent must be in the range (0, 100]")

    total_sms = int(torch.cuda.get_device_properties(device_id).multi_processor_count)
    if total_sms <= 0:
        raise GreenContextUnavailable(f"device cuda:{device_id} reports no SMs")

    num_sms = int(math.ceil(total_sms * (sm_percent / 100.0)))
    return max(1, min(total_sms, num_sms)), total_sms


def _ensure_primary_context(torch: Any, device_id: int) -> None:
    torch.cuda.set_device(device_id)
    torch.cuda.init()
    torch.cuda.current_stream(device_id)


__all__ = [
    "GreenContextHandle",
    "GreenContextUnavailable",
    "TorchGreenContext",
    "create_green_context_stream",
    "current_green_context",
    "green_context",
]
