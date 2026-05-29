import os

import pytest


@pytest.mark.integration
def test_offline_generation() -> None:
    if os.environ.get("MINI_VLLM_RUN_INTEGRATION") != "1":
        pytest.skip("requires CUDA, vLLM, and model download")

    from mini_vllm.offline_inference import OfflineLLM

    llm = OfflineLLM(model_name="facebook/opt-125m")
    response = llm.generate(
        "Carnegie Mellon University is known for ",
        max_tokens=100,
        ignore_eos=True,
    )
    assert response
