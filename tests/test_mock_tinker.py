"""MOCK of the Tinker service. This is NOT a real optimizer step.

No model is sampled and no weights change: ``FakeServiceClient`` stands in for
``tinker.ServiceClient``, and the fake sampling client replays scripted tool calls.
Everything else is real: this repository's training entrypoint, the Cookbook's
``cli_main`` / ``rl.train.main`` loop, its Harbor env and reward code, a Docker
sandbox, and the hidden grader computing rewards from warehouse state. What it
proves is the wiring from rollout through execution reward and group advantage
to the optimizer call.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("chz")
tinker = pytest.importorskip("tinker")
pytest.importorskip("tinker_cookbook.rl.train")
torch = pytest.importorskip("torch")

from tinker import TensorData  # noqa: E402
from tinker_cookbook.recipes.harbor_rl import harbor_env  # noqa: E402
from tinker_cookbook.renderers.base import (  # noqa: E402
    ParseTermination,
    RenderedMessage,
    Renderer,
    ToolCall,
)
from tinker_cookbook.rl import train as cookbook_train  # noqa: E402

from eltbench_tinker.train import _run, _validate, parser  # noqa: E402

from .test_docker_sandbox import GOOD, HOST, docker_works  # noqa: E402

pytestmark = pytest.mark.skipif(not docker_works(), reason="Docker daemon unavailable")

STOP = "<|end|>"
MARKER_ORACLE = "# rollout-with-pipeline"
ACTIONS = {
    "oracle": [
        MARKER_ORACLE
        + "\ncat > /workspace/submission/pipeline.sql <<'EOF'\n"
        + GOOD
        + "\nEOF\ncd /workspace && elt-sql -f submission/pipeline.sql",
    ],
    "nop": [],
}


class FakeTokenizer:
    """One token per byte, so no tokenizer download is needed."""

    name_or_path = "mock/byte-tokenizer"
    eos_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(text.encode())

    def decode(self, tokens: list[int], **_: Any) -> str:
        return bytes(tokens).decode(errors="replace")


class FakeRenderer(Renderer):
    def __init__(self, tokenizer: FakeTokenizer):
        super().__init__(tokenizer)

    def get_stop_sequences(self) -> list[str]:
        return [STOP]

    def render_message(self, message, ctx) -> RenderedMessage:
        calls = message.get("tool_calls") or []
        text = str(message.get("content") or "")
        for call in calls:
            text += call.function.arguments
        header = tinker.EncodedTextChunk(tokens=self.tokenizer.encode(f"<{message['role']}>"))
        if ctx.is_last and message["role"] == "assistant" and not text:
            return RenderedMessage(header=header, output=[])
        output = [tinker.EncodedTextChunk(tokens=self.tokenizer.encode(text + STOP))]
        return RenderedMessage(header=header, output=output)

    def parse_response(self, response):
        text = self.tokenizer.decode(response).removesuffix(STOP)
        if text.startswith("{"):
            call = ToolCall(function=ToolCall.FunctionBody(name="bash", arguments=text))
            return {"role": "assistant", "content": "", "tool_calls": [call]}, ParseTermination.STOP_SEQUENCE
        return {"role": "assistant", "content": text}, ParseTermination.STOP_SEQUENCE

    def create_conversation_prefix_with_tools(self, tools, system_prompt: str = ""):
        return [{"role": "system", "content": system_prompt}]


class Done:
    """A resolved Tinker API future."""

    def __init__(self, value: Any):
        self.value = value

    async def result_async(self) -> Any:
        return self.value


class FakeSamplingClient:
    """Replays a script per rollout. First rollout to ask gets the pipeline script."""

    def __init__(self, ledger: "Ledger"):
        self.ledger = ledger

    async def sample_async(self, prompt, num_samples, sampling_params) -> Any:
        tokens = [t for chunk in prompt.chunks for t in chunk.tokens]
        text = bytes(tokens).decode(errors="replace")
        done_turns = text.count("<assistant>") - 1
        if MARKER_ORACLE in text:
            script = "oracle"
        elif done_turns == 0 and "oracle" not in self.ledger.handed_out:
            script = "oracle"
            self.ledger.handed_out.add("oracle")
        else:
            script = "nop"
        actions = ACTIONS[script]
        if done_turns < len(actions):
            reply = json.dumps({"command": actions[done_turns]})
        else:
            reply = "done"
        generated = list((reply + STOP).encode())
        sequence = SimpleNamespace(
            tokens=generated, logprobs=[-0.5] * len(generated), stop_reason="stop"
        )
        return SimpleNamespace(sequences=[sequence])


class Ledger:
    def __init__(self) -> None:
        self.handed_out: set[str] = set()
        self.forward_backward: list[list[Any]] = []
        self.optim_steps: list[Any] = []
        self.loss_fns: list[str] = []
        self.order: list[str] = []


class FakeTrainingClient:
    def __init__(self, ledger: Ledger):
        self.ledger = ledger

    async def forward_backward_async(self, data, loss_fn, loss_fn_config=None) -> Done:
        self.ledger.order.append("forward_backward")
        self.ledger.forward_backward.append(list(data))
        self.ledger.loss_fns.append(loss_fn)
        outputs = [
            {"logprobs": TensorData.from_torch(torch.zeros(len(d.loss_fn_inputs["target_tokens"].data)))}
            for d in data
        ]
        return Done(SimpleNamespace(loss_fn_outputs=outputs, metrics={}))

    async def optim_step_async(self, adam_params) -> Done:
        self.ledger.order.append("optim_step")
        self.ledger.optim_steps.append(adam_params)
        return Done(SimpleNamespace(metrics={}))

    async def save_weights_and_get_sampling_client_async(self, *_: Any, **__: Any) -> FakeSamplingClient:
        return FakeSamplingClient(self.ledger)

    def create_sampling_client(self, *_: Any, **__: Any) -> FakeSamplingClient:
        return FakeSamplingClient(self.ledger)

    async def save_state_async(self, name: str, **_: Any) -> Done:
        return Done(SimpleNamespace(path=f"mock://state/{name}"))

    async def save_weights_for_sampler_async(self, name: str, **_: Any) -> Done:
        return Done(SimpleNamespace(path=f"mock://sampler/{name}"))


class FakeServiceClient:
    ledger: Ledger

    def __init__(self, **_: Any):
        pass

    def __enter__(self) -> "FakeServiceClient":
        return self

    def __exit__(self, *_: Any) -> None:
        return None

    async def create_lora_training_client_async(self, *_: Any, **__: Any) -> FakeTrainingClient:
        return FakeTrainingClient(self.ledger)


def test_mock_tinker_step_uses_execution_derived_advantages(tmp_path: Path, monkeypatch) -> None:
    """MOCK: rollout -> hidden-grader reward -> group advantage -> optimizer call."""
    from eltbench_tinker.harbor_task import materialize_local_harbor_task
    from eltbench_tinker.task import ELTTask, ExpectedTable

    fixture = Path(__file__).parent / "fixtures" / "customer_revenue"
    task_dir = materialize_local_harbor_task(
        ELTTask(
            task_id="customer-revenue",
            instruction="Create customer_revenue(customer_id, name, revenue) from the two CSV files.",
            source_dir=fixture,
            expected_tables=(
                ExpectedTable(
                    "customer_revenue",
                    (("customer_id", "BIGINT"), ("name", "VARCHAR"), ("revenue", "DECIMAL(38,2)")),
                    ((1, "Ada", 20.00), (2, "Linus", 9.00), (3, "Grace", 0.00)),
                    ("customer_id",),
                ),
            ),
        ),
        tmp_path / "tasks",
    )

    ledger = Ledger()
    FakeServiceClient.ledger = ledger
    tokenizer = FakeTokenizer()
    monkeypatch.setenv("TINKER_API_KEY", "mock-key-not-used")
    monkeypatch.setattr(cookbook_train.tinker, "ServiceClient", FakeServiceClient)
    monkeypatch.setattr(cookbook_train, "get_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(harbor_env.tokenizer_utils, "get_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(harbor_env, "get_renderer", lambda *_a, **_k: FakeRenderer(tokenizer))

    rewards: list[float] = []
    real_grader = harbor_env.HarborReward.__call__

    async def recording(self, history):
        reward, metrics = await real_grader(self, history)
        rewards.append(reward)
        return reward, metrics

    monkeypatch.setattr(harbor_env.HarborReward, "__call__", recording)

    args = parser().parse_args(
        [
            str(task_dir),
            "--model", "mock/model",
            "--renderer", "mock",
            "--steps", "1",
            "--group-size", "2",
            "--groups-per-batch", "1",
            "--max-turns", "4",
            "--log-path", str(tmp_path / "run"),
            "--sandbox", "docker",
            "--grader-timeout", "120",
            *(["--docker-host", HOST] if HOST else []),
        ]  # fmt: skip
    )
    _validate(args)
    asyncio.run(_run(args))

    # Real grader rewards, from warehouse state, one per rollout.
    assert sorted(rewards) == [0.0, 1.0]

    # The Cookbook's loop made exactly one mocked optimizer call, after the gradient.
    assert ledger.order == ["forward_backward", "optim_step"]
    assert ledger.loss_fns == ["importance_sampling"]
    data = ledger.forward_backward[0]
    assert len(data) == 2

    # Advantages are group-centred execution rewards: +0.5 for pipeline, -0.5 for nop.
    per_datum = sorted(
        {round(float(value), 6) for value in d.loss_fn_inputs["advantages"].data if value != 0.0}.pop()
        for d in data
    )
    assert per_datum == [-0.5, 0.5]
