from __future__ import annotations

from dataclasses import dataclass
import subprocess
import threading
import time
from typing import Optional
import shutil


@dataclass
class PowerSample:
    t: float
    power_w: float
    graphics_clock_mhz: Optional[int] = None
    mem_clock_mhz: Optional[int] = None


class EnergyMeter:
    def __init__(self, device_index: int = 0, sample_interval_s: float = 0.01):
        self.device_index = device_index
        self.sample_interval_s = sample_interval_s
        self._samples: list[PowerSample] = []
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._nvml_available = self._try_init_nvml()
        self._nvidia_smi = shutil.which("nvidia-smi")

    def start(self) -> None:
        self._samples = []
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[PowerSample]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        return list(self._samples)

    def measure(self, duration_s: float) -> list[PowerSample]:
        self.start()
        time.sleep(duration_s)
        return self.stop()

    def summarize(self, samples: Optional[list[PowerSample]] = None) -> dict[str, float]:
        samples = samples if samples is not None else self._samples
        if len(samples) < 2:
            return {"avg_power_w": 0.0, "energy_j": 0.0, "duration_s": 0.0}
        energy_j = 0.0
        for s0, s1 in zip(samples[:-1], samples[1:]):
            dt = s1.t - s0.t
            energy_j += 0.5 * (s0.power_w + s1.power_w) * dt
        duration_s = samples[-1].t - samples[0].t
        avg_power_w = energy_j / duration_s if duration_s > 0 else 0.0
        return {
            "avg_power_w": avg_power_w,
            "energy_j": energy_j,
            "duration_s": duration_s,
        }

    def _sample_loop(self) -> None:
        while not self._stop.is_set():
            sample = self._read_power()
            if sample is not None:
                self._samples.append(sample)
            time.sleep(self.sample_interval_s)

    def _read_power(self) -> Optional[PowerSample]:
        t = time.monotonic()
        if self._nvml_available:
            return self._read_power_nvml(t)
        if self._nvidia_smi:
            return self._read_power_smi(t)
        return None

    def _try_init_nvml(self) -> bool:
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            return True
        except Exception:
            return False

    def _read_power_nvml(self, t: float) -> Optional[PowerSample]:
        try:
            import pynvml  # type: ignore

            handle = pynvml.nvmlDeviceGetHandleByIndex(self.device_index)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
            power_w = power_mw / 1000.0
            graphics_clock = None
            mem_clock = None
            try:
                graphics_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS
                )
                mem_clock = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )
            except Exception:
                pass
            return PowerSample(
                t=t,
                power_w=power_w,
                graphics_clock_mhz=graphics_clock,
                mem_clock_mhz=mem_clock,
            )
        except Exception:
            return None

    def _read_power_smi(self, t: float) -> Optional[PowerSample]:
        try:
            out = subprocess.check_output(
                [
                    self._nvidia_smi,
                    "--query-gpu=power.draw,clocks.gr,clocks.mem",
                    "--format=csv,noheader,nounits",
                    "-i",
                    str(self.device_index),
                ],
                text=True,
            ).strip()
            if not out:
                return None
            parts = [p.strip() for p in out.split(",")]
            power_w = float(parts[0])
            graphics_clock = int(float(parts[1])) if len(parts) > 1 else None
            mem_clock = int(float(parts[2])) if len(parts) > 2 else None
            return PowerSample(
                t=t,
                power_w=power_w,
                graphics_clock_mhz=graphics_clock,
                mem_clock_mhz=mem_clock,
            )
        except Exception:
            return None


class GpuFrequencyController:
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._nvidia_smi = shutil.which("nvidia-smi")
        if self._nvidia_smi is None:
            raise RuntimeError("nvidia-smi not found; cannot control GPU clocks")

    def set_graphics_clock(self, min_mhz: int, max_mhz: int) -> None:
        subprocess.check_call(
            [
                self._nvidia_smi,
                "-i",
                str(self.device_index),
                "-lgc",
                f"{min_mhz},{max_mhz}",
            ]
        )

    def reset_graphics_clock(self) -> None:
        subprocess.check_call([self._nvidia_smi, "-i", str(self.device_index), "-rgc"])

    def set_power_limit(self, watts: int) -> None:
        subprocess.check_call(
            [self._nvidia_smi, "-i", str(self.device_index), "-pl", str(watts)]
        )
