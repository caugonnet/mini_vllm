from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


DEFAULT_SYSTEM = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech."
)

# Default Qwen2.5/Qwen3-Omni chat template. Other omni models may use a
# different template; override it with "prompt_template" in the workload JSON.
DEFAULT_TEXT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{media}{question}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Media placeholder snippets for the canonical Qwen-Omni template.
_MEDIA_PLACEHOLDERS = {
    "audio": "<|audio_bos|><|AUDIO|><|audio_eos|>",
    "image": "<|vision_bos|><|IMAGE|><|vision_eos|>",
    "video": "<|vision_bos|><|VIDEO|><|vision_eos|>",
}

QUERY_TYPES = (
    "text",
    "use_image",
    "use_audio",
    "use_video",
    "use_mixed_modalities",
    "use_audio_in_video",
)


@dataclass
class OmniWorkloadSpec:
    name: str = "omni_workload"
    query_type: str = "text"
    num_prompts: int = 1
    question: Optional[str] = None
    prompts: Optional[list[str]] = None
    system: str = DEFAULT_SYSTEM
    prompt_template: str = DEFAULT_TEXT_TEMPLATE
    modalities: Optional[list[str]] = None
    max_tokens: int = 256
    temperature: float = 0.0
    num_frames: int = 16
    sampling_rate: int = 16000
    media: dict = field(default_factory=dict)


def load_spec(path: str) -> OmniWorkloadSpec:
    resolved = _resolve_path(path)
    if not os.path.exists(resolved):
        raise FileNotFoundError(resolved)
    if resolved.endswith(".json"):
        with open(resolved, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        import yaml  # type: ignore

        with open(resolved, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError("omni workload config must be a JSON/YAML object")

    query_type = str(raw.get("query_type", "text"))
    if query_type not in QUERY_TYPES:
        raise ValueError(f"unknown query_type {query_type!r}; expected one of {QUERY_TYPES}")

    return OmniWorkloadSpec(
        name=str(raw.get("name", "omni_workload")),
        query_type=query_type,
        num_prompts=int(raw.get("num_prompts", 1)),
        question=raw.get("question"),
        prompts=raw.get("prompts"),
        system=str(raw.get("system", DEFAULT_SYSTEM)),
        prompt_template=str(raw.get("prompt_template", DEFAULT_TEXT_TEMPLATE)),
        modalities=raw.get("modalities"),
        max_tokens=int(raw.get("max_tokens", 256)),
        temperature=float(raw.get("temperature", 0.0)),
        num_frames=int(raw.get("num_frames", 16)),
        sampling_rate=int(raw.get("sampling_rate", 16000)),
        media=raw.get("media", {}) or {},
    )


def _resolve_path(path: str) -> str:
    p = Path(path)
    if p.exists():
        return str(p)
    here = Path(__file__).resolve().parent
    candidate = here / p.name
    if candidate.exists():
        return str(candidate)
    source_root = Path(__file__).resolve().parents[2]
    candidate = source_root / path
    if candidate.exists():
        return str(candidate)
    return path


def _format_text_prompt(spec: OmniWorkloadSpec, question: str, media: str = "") -> str:
    return spec.prompt_template.format(
        system=spec.system,
        media=media,
        question=question,
    )


def _media_snippet(modalities: list[str]) -> str:
    return "".join(_MEDIA_PLACEHOLDERS[m] for m in modalities if m in _MEDIA_PLACEHOLDERS)


def _build_one_prompt(spec: OmniWorkloadSpec, question: str) -> dict:
    """Build a single Omni prompt dict for the configured query type.

    Media query types lazily import vLLM asset helpers so a text-only workload
    has no audio/video dependencies.
    """
    if spec.query_type == "text":
        return {"prompt": _format_text_prompt(spec, question)}

    multi_modal_data: dict = {}
    mm_processor_kwargs: dict = {}
    modality_order: list[str] = []

    if spec.query_type in ("use_image", "use_mixed_modalities"):
        modality_order.append("image")
        multi_modal_data["image"] = _load_image(spec.media.get("image_path"))
    if spec.query_type in ("use_audio", "use_mixed_modalities"):
        modality_order.insert(0, "audio")
        multi_modal_data["audio"] = _load_audio(
            spec.media.get("audio_path"), spec.sampling_rate
        )
    if spec.query_type in ("use_video", "use_mixed_modalities"):
        modality_order.append("video")
        multi_modal_data["video"] = _load_video(
            spec.media.get("video_path"), spec.num_frames
        )
    if spec.query_type == "use_audio_in_video":
        video, audio = _load_video_with_audio(
            spec.media.get("video_path"), spec.num_frames, spec.sampling_rate
        )
        multi_modal_data["video"] = video
        multi_modal_data["audio"] = audio
        modality_order = ["video", "audio"]
        mm_processor_kwargs["use_audio_in_video"] = True

    prompt = _format_text_prompt(spec, question, media=_media_snippet(modality_order))
    inputs: dict = {"prompt": prompt, "multi_modal_data": multi_modal_data}
    if mm_processor_kwargs:
        inputs["mm_processor_kwargs"] = mm_processor_kwargs
    return inputs


def build_prompts(spec: OmniWorkloadSpec) -> list[dict]:
    if spec.prompts:
        if spec.query_type != "text":
            raise ValueError("explicit 'prompts' is only supported for query_type='text'")
        prompts = [{"prompt": _format_text_prompt(spec, q)} for q in spec.prompts]
    else:
        question = spec.question or _default_question(spec.query_type)
        prompts = [_build_one_prompt(spec, question) for _ in range(spec.num_prompts)]

    if spec.modalities is not None:
        for prompt in prompts:
            prompt["modalities"] = list(spec.modalities)
    return prompts


def _default_question(query_type: str) -> str:
    return {
        "text": "Explain how a multi-stage omni inference pipeline works in 15 words.",
        "use_image": "What is the content of this image?",
        "use_audio": "What is the content of this audio?",
        "use_video": "Why is this video interesting?",
        "use_mixed_modalities": (
            "What is recited in the audio? What is in this image? "
            "Why is this video interesting?"
        ),
        "use_audio_in_video": "Describe the content of the video, then transcribe the speech.",
    }[query_type]


def _load_image(image_path: Optional[str]):
    from vllm.assets.image import ImageAsset
    from vllm.multimodal.image import convert_image_mode

    if image_path:
        from PIL import Image

        return convert_image_mode(Image.open(image_path), "RGB")
    return convert_image_mode(ImageAsset("cherry_blossom").pil_image, "RGB")


def _load_audio(audio_path: Optional[str], sampling_rate: int):
    if audio_path:
        import numpy as np
        from vllm.multimodal.media.audio import load_audio

        signal, sr = load_audio(audio_path, sr=sampling_rate)
        return (signal.astype(np.float32), sr)
    from vllm.assets.audio import AudioAsset

    return AudioAsset("mary_had_lamb").audio_and_sample_rate


def _load_video(video_path: Optional[str], num_frames: int):
    from vllm.assets.video import VideoAsset, video_to_ndarrays

    if video_path:
        return video_to_ndarrays(video_path, num_frames=num_frames)
    return VideoAsset(name="baby_reading", num_frames=num_frames).np_ndarrays


def _load_video_with_audio(video_path: Optional[str], num_frames: int, sampling_rate: int):
    from vllm.assets.video import VideoAsset, video_to_ndarrays

    if video_path:
        import numpy as np
        from vllm.multimodal.media.audio import load_audio

        frames = video_to_ndarrays(video_path, num_frames=num_frames)
        signal, sr = load_audio(video_path, sr=sampling_rate)
        return frames, (signal.astype(np.float32), sr)
    asset = VideoAsset(name="baby_reading", num_frames=num_frames)
    return asset.np_ndarrays, asset.get_audio(sampling_rate=sampling_rate)
