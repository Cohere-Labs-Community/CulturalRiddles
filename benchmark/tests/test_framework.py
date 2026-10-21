from __future__ import annotations

import json
import sys
import types
from collections import namedtuple
from pathlib import Path

import pytest
import riddles_framework.models.transformers_local_adapter as local_adapter_module
from riddles_framework.cache_clients import make_transformers_client
from riddles_framework.community_display import (
    format_community_display,
    format_country,
    format_language_name,
    format_script_iso,
    format_script_name,
)
from riddles_framework.constants import GenerationStatus, ResponseFormat, RiddleLanguagePresented
from riddles_framework.dataset_loader import load_benchmark_instances
from riddles_framework.exceptions import (
    PromptTemplateNotFoundError,
    UnsupportedConfigurationError,
)
from riddles_framework.factory import create_model
from riddles_framework.models.dummy import DummyModel
from riddles_framework.prompts import (
    build_condition_id,
    build_prompt,
    build_substitutions,
    list_available_templates,
    load_prompt_template,
)
from riddles_framework.registry import register_model_adapter
from riddles_framework.runner import BenchmarkRunner
from riddles_framework.schemas import (
    BenchmarkInstance,
    GenerationResult,
    ModelCapabilities,
    ModelSpec,
    PromptConfig,
    PromptTemplate,
    RunConfig,
)
from riddles_framework.utils import looks_like_refusal
from riddles_framework.yaml_config import load_run_config_from_yaml


def test_cli_rejects_unregistered_historical_adapter_before_run(monkeypatch, capsys):
    from riddles_framework import cli

    spec = types.SimpleNamespace(provider="deepseek", access_pattern="direct_api")
    monkeypatch.setattr(cli, "register_builtin_adapters", lambda: None)
    monkeypatch.setattr(cli, "list_registered_adapters", lambda: [])
    monkeypatch.setattr(
        cli,
        "load_run_config_from_yaml",
        lambda path: types.SimpleNamespace(model_spec=spec),
    )
    monkeypatch.setattr(sys, "argv", ["riddles-framework", "--config", "historical.yaml"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "Unsupported adapter 'deepseek:direct_api'" in capsys.readouterr().err


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_dataset_row() -> dict:
    return {
        "riddle_id": "tamil_india_history_001",
        "source_row_number": 1,
        "language": "tam",
        "community": "tamil_india",
        "topic": "History",
        "is_wildcard": False,
        "riddle_original": "\u0b92\u0bb0\u0bc1 \u0bb5\u0bb0\u0bb2\u0bbe\u0bb1\u0bcd\u0bb1\u0bc1 \u0baa\u0bc1\u0ba4\u0bbf\u0bb0\u0bcd",
        "riddle_english": "A historical riddle",
        "reference_answer_original": "\u0bb0\u0bbe\u0b9c\u0bb0\u0bbe\u0b9c \u0b9a\u0bc7\u0bbe\u0bb4\u0ba9\u0bcd",
        "reference_answer_english": "Rajaraja Chola",
        "riddle_hint": "\u0bae\u0ba9\u0bcd\u0ba9\u0ba9\u0bcd",
        "riddle_note": None,
    }


def test_dataset_loader_reads_valid_jsonl(tmp_path: Path):
    dataset_path = tmp_path / "dataset.jsonl"
    write_jsonl(dataset_path, [sample_dataset_row()])

    instances = load_benchmark_instances(dataset_path)
    assert len(instances) == 1
    assert instances[0].riddle_id == "tamil_india_history_001"
    assert instances[0].language == "tam"


def test_benchmark_instance_validation():
    row = sample_dataset_row()
    inst = BenchmarkInstance.from_dict(row)
    inst.validate()
    assert inst.community == "tamil_india"


def test_prompt_build_source_no_hint():
    inst = BenchmarkInstance.from_dict(sample_dataset_row())
    template = PromptTemplate(
        template_id="riddle_basic",
        template_version="v1",
        instruction_text="Solve this riddle.",
    )
    cfg = PromptConfig(
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        response_format=ResponseFormat.FREE_FORM,
    )
    prompt = build_prompt(inst, template, cfg)
    assert "RIDDLE:" in prompt
    assert "HINT:" not in prompt
    assert (
        "\u0b92\u0bb0\u0bc1 \u0bb5\u0bb0\u0bb2\u0bbe\u0bb1\u0bcd\u0bb1\u0bc1 \u0baa\u0bc1\u0ba4\u0bbf\u0bb0\u0bcd"
        in prompt
    )


def test_prompt_build_source_with_hint():
    inst = BenchmarkInstance.from_dict(sample_dataset_row())
    template = PromptTemplate(
        template_id="riddle_basic",
        template_version="v1",
        instruction_text="Solve this riddle.",
    )
    cfg = PromptConfig(
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=True,
        response_format=ResponseFormat.FREE_FORM,
    )
    prompt = build_prompt(inst, template, cfg)
    assert "HINT:" in prompt
    assert "\u0bae\u0ba9\u0bcd\u0ba9\u0ba9\u0bcd" in prompt


def test_condition_id():
    cid = build_condition_id(RiddleLanguagePresented.SOURCE, False)
    assert cid == "source_no_hint"


def test_load_prompt_template_master():
    tpl = load_prompt_template("riddle_master", "v1")
    assert tpl.template_id == "riddle_master"
    assert tpl.template_version == "v1"
    assert tpl.recommended_response_format == ResponseFormat.JSON


def test_load_prompt_template_missing_raises():
    with pytest.raises(PromptTemplateNotFoundError):
        load_prompt_template("nonexistent", "v999")


def test_list_available_templates_includes_shipped_ones():
    available = list_available_templates()
    assert ("riddle_master", "v1") in available


def _dummy_spec() -> ModelSpec:
    return ModelSpec(
        model_name="Dummy Echo",
        model_id="dummy/echo-v1",
        provider="dummy",
        access_pattern="local",
        provider_model_name="dummy-echo-v1",
        capabilities=ModelCapabilities(
            supports_reasoning=False,
            supports_search=False,
            supports_structured_output=True,
            supports_batch=False,
            supports_seed=False,
        ),
        closed_source=False,
    )


def test_dummy_runner_creates_outputs(tmp_path: Path):
    dataset_path = tmp_path / "dataset.jsonl"
    first_row = sample_dataset_row()
    second_row = sample_dataset_row()
    second_row["riddle_id"] = "tamil_india_history_002"
    second_row["source_row_number"] = 2
    write_jsonl(dataset_path, [first_row, second_row])

    spec = _dummy_spec()

    run_config = RunConfig(
        model_spec=spec,
        runner_name="tester",
        dataset_path=str(dataset_path),
        dataset_version="test_v1",
        prompt_template_id="riddle_basic",
        prompt_template_version="v1",
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        response_format=ResponseFormat.JSON,
        temperature=None,
        max_output_tokens=32,
    )

    model = DummyModel(spec)
    template = PromptTemplate(
        template_id="riddle_basic",
        template_version="v1",
        instruction_text="Solve this riddle.",
    )

    runner = BenchmarkRunner(
        run_config=run_config,
        prompt_template=template,
        model=model,
        output_root=tmp_path / "runs",
        max_instances=1,
    )
    run_id = runner.run()

    run_dir = tmp_path / "runs" / run_id
    assert run_dir.exists()
    assert (run_dir / "run_config.json").exists()
    assert (run_dir / "run_metadata.json").exists()
    assert (run_dir / "results.jsonl").exists()
    assert (run_dir / "manifest.json").exists()

    results = (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(results) == 1
    row = json.loads(results[0])
    assert row["generation_status"] == "success"
    assert row["raw_response"] is not None
    # Temperature should be recorded as None when not set.
    assert row["temperature"] is None
    # model_params field should be present (even if empty) for analysis joins.
    assert "model_params" in row


def test_runner_rejects_english_with_hint(tmp_path: Path):
    dataset_path = tmp_path / "dataset.jsonl"
    write_jsonl(dataset_path, [sample_dataset_row()])

    spec = _dummy_spec()
    run_config = RunConfig(
        model_spec=spec,
        runner_name="tester",
        dataset_path=str(dataset_path),
        dataset_version="test_v1",
        prompt_template_id="riddle_basic",
        prompt_template_version="v1",
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.ENGLISH,
        is_hint_provided=True,
    )

    model = DummyModel(spec)
    template = PromptTemplate(
        template_id="riddle_basic",
        template_version="v1",
        instruction_text="Solve this riddle.",
    )

    runner = BenchmarkRunner(
        run_config=run_config,
        prompt_template=template,
        model=model,
        output_root=tmp_path / "runs",
    )

    with pytest.raises(UnsupportedConfigurationError):
        runner.run()


def test_model_params_are_threaded_through(tmp_path: Path):
    """The runner should forward model_params verbatim to the adapter."""
    dataset_path = tmp_path / "dataset.jsonl"
    write_jsonl(dataset_path, [sample_dataset_row()])

    spec = _dummy_spec()
    run_config = RunConfig(
        model_spec=spec,
        runner_name="tester",
        dataset_path=str(dataset_path),
        dataset_version="test_v1",
        prompt_template_id="riddle_basic",
        prompt_template_version="v1",
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        model_params={"top_p": 0.95, "frequency_penalty": 0.1},
    )

    model = DummyModel(spec)
    template = PromptTemplate(
        template_id="riddle_basic",
        template_version="v1",
        instruction_text="Solve this riddle.",
    )

    runner = BenchmarkRunner(
        run_config=run_config,
        prompt_template=template,
        model=model,
        output_root=tmp_path / "runs",
    )
    run_id = runner.run()

    row = json.loads(
        (tmp_path / "runs" / run_id / "results.jsonl").read_text(encoding="utf-8").strip()
    )
    assert row["model_params"] == {"top_p": 0.95, "frequency_penalty": 0.1}
    # The Dummy adapter echoes model_params into provider_details for verification.
    assert row["provider_details"]["model_params"] == {
        "top_p": 0.95,
        "frequency_penalty": 0.1,
    }


def test_runner_uses_generate_batch_when_batch_mode_enabled(tmp_path: Path):
    rows = []
    for index in range(3):
        row = sample_dataset_row()
        row["riddle_id"] = f"riddle_{index}"
        rows.append(row)

    dataset_path = tmp_path / "dataset.jsonl"
    write_jsonl(dataset_path, rows)

    spec = ModelSpec(
        model_name="Batch Local",
        model_id="local/batch-test",
        provider="local",
        access_pattern="local",
        provider_model_name="batch-test",
        capabilities=ModelCapabilities(supports_batch=True),
    )

    class BatchModel(DummyModel):
        def __init__(self, model_spec):
            super().__init__(model_spec)
            self.batches: list[list[str]] = []

        def generate(self, *args, **kwargs):
            raise AssertionError("runner should call generate_batch in batch mode")

        def generate_batch(self, prompts, **kwargs):
            self.batches.append(list(prompts))
            return [
                GenerationResult(
                    raw_response=f"batch answer {index}",
                    status=GenerationStatus.SUCCESS,
                    provider_details={"batch_size": len(prompts)},
                )
                for index, _ in enumerate(prompts)
            ]

    run_config = RunConfig(
        model_spec=spec,
        runner_name="tester",
        dataset_path=str(dataset_path),
        dataset_version="test_v1",
        prompt_template_id="riddle_basic",
        prompt_template_version="v1",
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        response_format=ResponseFormat.FREE_FORM,
        batch_mode=True,
        batch_size=2,
    )

    model = BatchModel(spec)
    template = PromptTemplate(
        template_id="riddle_basic",
        template_version="v1",
        instruction_text="Solve this riddle.",
    )

    runner = BenchmarkRunner(
        run_config=run_config,
        prompt_template=template,
        model=model,
        output_root=tmp_path / "runs",
    )
    run_id = runner.run()

    assert [len(batch) for batch in model.batches] == [2, 1]

    run_dir = tmp_path / "runs" / run_id
    results = [
        json.loads(line)
        for line in (run_dir / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert len(results) == 3
    assert [row["generation_status"] for row in results] == ["success"] * 3
    assert [row["batch_size"] for row in results] == [2, 2, 2]

    metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["batch_mode_requested"] is True
    assert metadata["batch_mode_used"] is True
    assert metadata["batch_size"] == 2

    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["batch_mode_used"] is True
    assert manifest["batch_size"] == 2


def test_registry_and_factory_dummy():
    key = "dummytest:local"

    try:
        register_model_adapter(key, lambda spec, cfg: DummyModel(spec))
    except Exception:
        pass

    spec = ModelSpec(
        model_name="Dummy Test",
        model_id="dummytest/echo-v1",
        provider="dummytest",
        access_pattern="local",
        provider_model_name="dummy-echo-v1",
    )
    model = create_model(spec, {})
    assert model.model_id == "dummytest/echo-v1"


def _local_spec() -> ModelSpec:
    return ModelSpec(
        model_name="Tiny Local",
        model_id="local/tiny",
        provider="local",
        access_pattern="local",
        provider_model_name="tiny-local-model",
        capabilities=ModelCapabilities(supports_batch=True, supports_seed=True),
        closed_source=False,
    )


class _FakeTokenizer:
    pad_token_id = None
    eos_token_id = 0
    pad_token = None
    eos_token = "<eos>"

    @classmethod
    def from_pretrained(cls, model_name: str, **kwargs):
        assert model_name == "tiny-local-model"
        return cls()

    def __call__(self, text: str | list[str], **kwargs):
        if kwargs.get("return_tensors") == "pt":
            if isinstance(text, list):
                return {
                    "input_ids": _FakeTensor([[1, 2, 3], [0, 1, 2]]),
                    "attention_mask": _FakeTensor([[1, 1, 1], [0, 1, 1]]),
                }
            return {"input_ids": _FakeTensor([[1, 2, 3]])}
        if text == "local answer":
            return {"input_ids": [10, 11]}
        if text == "cached answer":
            return {"input_ids": [12, 13]}
        if text == "batch answer 1":
            return {"input_ids": [14, 15, 16]}
        if text == "batch answer 2":
            return {"input_ids": [17, 18]}
        return {"input_ids": [1, 2, 3]}

    def decode(self, token_ids, skip_special_tokens: bool = True):
        assert skip_special_tokens
        return "cached answer"


class _FakeTensor:
    def __init__(self, data):
        self.data = data
        self.shape = (
            (len(data), len(data[0])) if data and isinstance(data[0], list) else (len(data),)
        )
        self.ndim = len(self.shape)

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.data[item]
        value = self.data[item]
        return _FakeTensor(value) if isinstance(value, list) else value

    def to(self, device):
        return self

    def sum(self, dim=None):
        if dim == 1 and self.ndim == 2:
            return _FakeTensor([sum(row) for row in self.data])
        if dim is None:
            if self.ndim == 2:
                return sum(sum(row) for row in self.data)
            return sum(self.data)
        raise NotImplementedError

    def tolist(self):
        return self.data


def test_local_transformers_adapter_non_cached(monkeypatch):
    calls: dict[str, object] = {}

    def fake_pipeline(task, *, model, tokenizer, model_kwargs):
        assert task == "text-generation"
        assert model == "tiny-local-model"
        assert isinstance(tokenizer, _FakeTokenizer)
        calls["model_kwargs"] = model_kwargs

        def run(prompt, **kwargs):
            calls["prompt"] = prompt
            calls["kwargs"] = kwargs
            return [{"generated_text": "local answer"}]

        return run

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        pipeline=fake_pipeline,
        set_seed=lambda seed: calls.setdefault("seed", seed),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    adapter = local_adapter_module.LocalTransformersAdapter(
        _local_spec(),
        {
            "cache_enabled": False,
            "model_params": {
                "model_load": {"device_map": "auto", "torch_dtype": "float16"},
                "tokenizer_load": {"trust_remote_code": True},
            },
        },
    )

    result = adapter.generate(
        "prompt",
        temperature=0.7,
        max_output_tokens=5,
        seed=123,
        model_params={"top_p": 0.9},
    )

    assert result.status == GenerationStatus.SUCCESS
    assert result.raw_response == "local answer"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert calls["model_kwargs"]["device_map"] == "auto"
    assert calls["model_kwargs"]["dtype"] == "float16"
    assert result.provider_details == {
        "model": "tiny-local-model",
        "cache_enabled": False,
        "batch_size": 1,
        "local_cache_client_mode": None,
        "cache_client_interface": None,
        "model_load_params": {
            "device_map": "auto",
            "dtype": "float16",
        },
        "cache_metadata": None,
    }
    assert calls["seed"] == 123
    assert calls["kwargs"]["max_new_tokens"] == 5
    assert calls["kwargs"]["temperature"] == 0.7
    assert calls["kwargs"]["do_sample"] is True
    assert calls["kwargs"]["top_p"] == 0.9
    assert "model_load" not in calls["kwargs"]
    assert "tokenizer_load" not in calls["kwargs"]


def test_local_transformers_adapter_non_cached_generate_batch(monkeypatch):
    calls: dict[str, object] = {}

    def fake_pipeline(task, *, model, tokenizer, model_kwargs):
        assert task == "text-generation"
        calls["model_kwargs"] = model_kwargs

        def run(prompts, **kwargs):
            calls["prompts"] = prompts
            calls["kwargs"] = kwargs
            return [
                [{"generated_text": "batch answer 1"}],
                [{"generated_text": "batch answer 2"}],
            ]

        return run

    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        pipeline=fake_pipeline,
        set_seed=lambda seed: calls.setdefault("seed", seed),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    adapter = local_adapter_module.LocalTransformersAdapter(
        _local_spec(),
        {
            "cache_enabled": False,
            "batch_size": 2,
            "model_params": {
                "model_load": {"device_map": "auto", "torch_dtype": "float16"},
            },
        },
    )

    results = adapter.generate_batch(
        ["prompt 1", "prompt 2"],
        max_output_tokens=5,
        seed=123,
        model_params={
            "top_p": 0.9,
            "model_load": {"device_map": "auto", "torch_dtype": "float16"},
        },
    )

    assert [result.status for result in results] == [
        GenerationStatus.SUCCESS,
        GenerationStatus.SUCCESS,
    ]
    assert [result.raw_response for result in results] == [
        "batch answer 1",
        "batch answer 2",
    ]
    assert [result.output_tokens for result in results] == [3, 2]
    assert calls["prompts"] == ["prompt 1", "prompt 2"]
    assert calls["kwargs"]["batch_size"] == 2
    assert calls["kwargs"]["max_new_tokens"] == 5
    assert calls["kwargs"]["do_sample"] is False
    assert calls["kwargs"]["top_p"] == 0.9
    assert "model_load" not in calls["kwargs"]
    assert calls["model_kwargs"]["dtype"] == "float16"
    assert results[0].provider_details["batch_size"] == 2
    assert results[0].provider_details["batch_count"] == 2
    assert results[0].provider_details["cache_enabled"] is False


def test_local_transformers_adapter_cached_uses_cachesaver(monkeypatch):
    calls: dict[str, object] = {}
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        set_seed=lambda seed: calls.setdefault("seed", seed),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    Metadata = namedtuple("Metadata", ["cached", "duplicated"])

    class FakeCompletions:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return "cached answer", Metadata(
                cached=[True],
                duplicated=[False],
            )

    class FakeClient:
        chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=FakeCompletions().create)
        )

    def fake_make_transformers_client(use_cache, **kwargs):
        calls["client_kwargs"] = kwargs
        assert use_cache is True
        return FakeClient()

    monkeypatch.setattr(
        local_adapter_module,
        "make_transformers_client",
        fake_make_transformers_client,
    )

    adapter = local_adapter_module.LocalTransformersAdapter(
        _local_spec(),
        {
            "cache_enabled": True,
            "cache_namespace": "test/local",
            "cache_dir": ".cache/test",
            "batch_size": 4,
            "local_cache_client_mode": "sync",
            "model_params": {
                "model_load": {"device_map": "auto", "torch_dtype": "float16"},
            },
        },
    )

    result = adapter.generate(
        "prompt",
        max_output_tokens=7,
        seed=456,
        model_params={
            "top_k": 20,
            "model_load": {"device_map": "auto", "torch_dtype": "float16"},
        },
    )

    assert result.status == GenerationStatus.SUCCESS
    assert result.raw_response == "cached answer"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.provider_details["cache_enabled"] is True
    assert result.provider_details["batch_size"] == 4
    assert result.provider_details["local_cache_client_mode"] == "sync"
    assert result.provider_details["cache_client_interface"] == "chat"
    assert result.provider_details["cache_metadata"] == {
        "cached": [True],
        "duplicated": [False],
    }
    assert calls["seed"] == 456
    assert calls["client_kwargs"] == {
        "model_name": "tiny-local-model",
        "namespace": "test/local",
        "cache_dir": ".cache/test",
        "batch_size": 4,
        "client_mode": "sync",
        "device_map": "auto",
        "dtype": "float16",
    }
    assert calls["kwargs"]["max_new_tokens"] == 7
    assert calls["kwargs"]["do_sample"] is False
    assert calls["kwargs"]["top_k"] == 20
    assert "model_load" not in calls["kwargs"]
    assert calls["kwargs"]["metadata"] is True


def test_local_transformers_adapter_cached_uses_documented_chat_interface(monkeypatch):
    calls: dict[str, object] = {}
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        set_seed=lambda seed: calls.setdefault("seed", seed),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    Metadata = namedtuple("Metadata", ["cached", "duplicated"])

    class FakeCompletions:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return "chat answer", Metadata(cached=[False], duplicated=[False])

    class FakeClient:
        _riddles_transformers_interface = "chat"
        chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=FakeCompletions().create)
        )

    monkeypatch.setattr(
        local_adapter_module,
        "make_transformers_client",
        lambda use_cache, **kwargs: FakeClient(),
    )

    adapter = local_adapter_module.LocalTransformersAdapter(
        _local_spec(),
        {"cache_enabled": True},
    )

    result = adapter.generate("prompt", max_output_tokens=9)

    assert result.status == GenerationStatus.SUCCESS
    assert result.raw_response == "chat answer"
    assert result.provider_details["cache_client_interface"] == "chat"
    assert calls["kwargs"]["messages"] == [{"role": "user", "content": "prompt"}]
    assert calls["kwargs"]["max_new_tokens"] == 9
    assert calls["kwargs"]["do_sample"] is False
    assert calls["kwargs"]["metadata"] is True


def test_local_transformers_adapter_cached_uses_installed_generate_interface(monkeypatch):
    calls: dict[str, object] = {}
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        set_seed=lambda seed: calls.setdefault("seed", seed),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    Metadata = namedtuple("Metadata", ["cached", "duplicated"])

    class FakeCompletions:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return _FakeTensor([[1, 2, 3, 4, 5]]), Metadata(
                cached=[True],
                duplicated=[False],
            )

    class FakeClient:
        _riddles_transformers_interface = "generate"
        model = types.SimpleNamespace(model=types.SimpleNamespace(device="cpu"))
        chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=FakeCompletions().create)
        )

    monkeypatch.setattr(
        local_adapter_module,
        "make_transformers_client",
        lambda use_cache, **kwargs: FakeClient(),
    )

    adapter = local_adapter_module.LocalTransformersAdapter(
        _local_spec(),
        {
            "cache_enabled": True,
            "batch_size": 4,
            "local_cache_client_mode": "sync",
        },
    )

    result = adapter.generate(
        "prompt",
        max_output_tokens=7,
        model_params={"top_k": 20},
    )

    assert result.status == GenerationStatus.SUCCESS
    assert result.raw_response == "cached answer"
    assert result.input_tokens == 3
    assert result.output_tokens == 2
    assert result.provider_details["cache_client_interface"] == "generate"
    assert calls["kwargs"]["max_new_tokens"] == 7
    assert calls["kwargs"]["do_sample"] is False
    assert calls["kwargs"]["pad_token_id"] == 0
    assert calls["kwargs"]["top_k"] == 20
    assert calls["kwargs"]["metadata"] is True


def test_local_transformers_adapter_cached_generate_interface_batches(monkeypatch):
    calls: dict[str, object] = {}
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        set_seed=lambda seed: calls.setdefault("seed", seed),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    Metadata = namedtuple("Metadata", ["cached", "duplicated"])

    class FakeCompletions:
        def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return _FakeTensor([[1, 2, 3, 4, 5], [0, 1, 2, 6, 7]]), Metadata(
                cached=[False],
                duplicated=[False],
            )

    class FakeClient:
        _riddles_transformers_interface = "generate"
        model = types.SimpleNamespace(model=types.SimpleNamespace(device="cuda:0"))
        chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=FakeCompletions().create)
        )

    monkeypatch.setattr(
        local_adapter_module,
        "make_transformers_client",
        lambda use_cache, **kwargs: FakeClient(),
    )

    adapter = local_adapter_module.LocalTransformersAdapter(
        _local_spec(),
        {
            "cache_enabled": True,
            "batch_size": 2,
            "local_cache_client_mode": "sync",
        },
    )

    results = adapter.generate_batch(
        ["prompt 1", "prompt 2"],
        max_output_tokens=7,
        model_params={"top_k": 20},
    )

    assert [result.status for result in results] == [
        GenerationStatus.SUCCESS,
        GenerationStatus.SUCCESS,
    ]
    assert [result.raw_response for result in results] == [
        "cached answer",
        "cached answer",
    ]
    assert [result.input_tokens for result in results] == [3, 2]
    assert calls["kwargs"]["max_new_tokens"] == 7
    assert calls["kwargs"]["do_sample"] is False
    assert calls["kwargs"]["pad_token_id"] == 0
    assert calls["kwargs"]["top_k"] == 20
    assert calls["kwargs"]["metadata"] is True
    assert results[0].provider_details["batch_size"] == 2
    assert results[0].provider_details["batch_count"] == 2
    assert results[0].provider_details["cache_client_interface"] == "generate"


def test_local_transformers_adapter_cached_uses_async_client(monkeypatch):
    calls: dict[str, object] = {}
    fake_transformers = types.SimpleNamespace(
        AutoTokenizer=_FakeTokenizer,
        set_seed=lambda seed: calls.setdefault("seed", seed),
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    Metadata = namedtuple("Metadata", ["cached", "duplicated"])

    class FakeCompletions:
        async def create(self, **kwargs):
            calls["kwargs"] = kwargs
            return "async answer", Metadata(cached=[True], duplicated=[False])

    class FakeClient:
        chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=FakeCompletions().create)
        )

    def fake_make_transformers_client(use_cache, **kwargs):
        calls["client_kwargs"] = kwargs
        return FakeClient()

    monkeypatch.setattr(
        local_adapter_module,
        "make_transformers_client",
        fake_make_transformers_client,
    )

    adapter = local_adapter_module.LocalTransformersAdapter(
        _local_spec(),
        {
            "cache_enabled": True,
            "batch_size": 2,
            "local_cache_client_mode": "async",
        },
    )

    result = adapter.generate("prompt", max_output_tokens=11)

    assert result.status == GenerationStatus.SUCCESS
    assert result.raw_response == "async answer"
    assert calls["client_kwargs"]["client_mode"] == "async"
    assert result.provider_details["local_cache_client_mode"] == "async"
    assert result.provider_details["cache_client_interface"] == "chat"
    assert calls["kwargs"]["max_new_tokens"] == 11
    assert calls["kwargs"]["metadata"] is True


def test_make_transformers_client_cached_uses_sync_hf_wrapper(monkeypatch):
    calls: dict[str, object] = {}

    class FakeHFTransformers:
        def __init__(self, *, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs

    fake_module = types.SimpleNamespace(HFTransformers=FakeHFTransformers)
    monkeypatch.setitem(sys.modules, "cachesaver.models.transformers", fake_module)

    client = make_transformers_client(
        True,
        model_name="tiny-local-model",
        namespace="test/local",
        cache_dir=".cache/test",
        batch_size=None,
        client_mode="sync",
    )

    assert isinstance(client, FakeHFTransformers)
    assert calls["model_name"] == "tiny-local-model"
    assert calls["kwargs"] == {
        "namespace": "test/local",
        "cachedir": ".cache/test",
    }
    assert client._riddles_transformers_client_mode == "sync"
    assert client._riddles_transformers_interface == "chat"


def test_make_transformers_client_cached_uses_async_hf_wrapper(monkeypatch):
    calls: dict[str, object] = {}

    class FakeAsyncHFTransformers:
        def __init__(self, *, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs

    fake_module = types.SimpleNamespace(AsyncHFTransformers=FakeAsyncHFTransformers)
    monkeypatch.setitem(sys.modules, "cachesaver.models.transformers", fake_module)

    client = make_transformers_client(
        True,
        model_name="tiny-local-model",
        namespace="test/local",
        cache_dir=".cache/test",
        batch_size=4,
        client_mode="async",
    )

    assert isinstance(client, FakeAsyncHFTransformers)
    assert calls["model_name"] == "tiny-local-model"
    assert calls["kwargs"] == {
        "namespace": "test/local",
        "cachedir": ".cache/test",
        "batch_size": 4,
    }
    assert client._riddles_transformers_client_mode == "async"
    assert client._riddles_transformers_interface == "chat"


def test_make_transformers_client_cached_sync_falls_back_to_installed_wrapper(monkeypatch):
    calls: dict[str, object] = {}

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_name, **kwargs):
            calls["model_name"] = model_name
            calls["kwargs"] = kwargs
            return types.SimpleNamespace()

    fake_module = types.SimpleNamespace(AutoModelForCausalLM=FakeAutoModelForCausalLM)
    monkeypatch.setitem(sys.modules, "cachesaver.models.transformers", fake_module)

    client = make_transformers_client(
        True,
        model_name="tiny-local-model",
        namespace="test/local",
        cache_dir=".cache/test",
        batch_size=4,
        client_mode="sync",
    )

    assert calls["model_name"] == "tiny-local-model"
    assert calls["kwargs"] == {
        "namespace": "test/local",
        "cachedir": ".cache/test",
        "batch_size": 4,
    }
    assert client._riddles_transformers_client_mode == "sync"
    assert client._riddles_transformers_interface == "generate"


def test_yaml_config_loading(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(
        """
runner_name: tester
dataset_path: data/canonical/riddles.jsonl
dataset_version: canonical_v1
prompt_template_id: riddle_basic
prompt_template_version: v1
prompt_language: eng
riddle_language_presented: source
is_hint_provided: false
reasoning_mode: null
search_enabled: false
response_format: json
batch_mode: false
batch_size: 4
temperature: null
max_output_tokens: 64
seed: null
model_params:
  top_p: 0.9
cache_enabled: false
cache_backend: null
cache_namespace: null
cache_dir: null
local_cache_client_mode: async
api_key_env_var: null
base_url: null
runner_version: v1.0.0
notes: test config
strict_capability_check: true

model_spec:
  model_name: Dummy Echo
  model_id: dummy/echo-v1
  provider: dummy
  access_pattern: local
  provider_model_name: dummy-echo-v1
  closed_source: false
  notes: Dummy model
  capabilities:
    supports_reasoning: false
    supports_search: false
    supports_structured_output: true
    supports_batch: false
    supports_seed: false
""",
        encoding="utf-8",
    )

    cfg = load_run_config_from_yaml(str(yaml_path))
    assert cfg.runner_name == "tester"
    assert cfg.model_spec.model_id == "dummy/echo-v1"
    assert cfg.riddle_language_presented.value == "source"
    # Optional temperature defaults to None when YAML says null.
    assert cfg.temperature is None
    # model_params loads through.
    assert cfg.model_params == {"top_p": 0.9}
    # response_format is parsed when explicitly set (default is free_form).
    assert cfg.response_format == ResponseFormat.JSON
    assert cfg.batch_size == 4
    assert cfg.local_cache_client_mode == "async"


def test_refusal_heuristic():
    # Policy refusals: should be tagged.
    assert looks_like_refusal("I'm sorry, but I can't help with that.")
    assert looks_like_refusal("I cannot assist with that.")

    # Honest uncertainty: must NOT be tagged as refusal.
    assert not looks_like_refusal("I cannot determine the answer from the given information.")
    assert not looks_like_refusal("I do not have enough information.")
    assert not looks_like_refusal("I don't know.")

    # Normal answers: not refusals.
    assert not looks_like_refusal("Paris")
    assert not looks_like_refusal("")
    assert not looks_like_refusal(None)


# ---------------------------------------------------------------------------
# Cultural-context substitution tests
# ---------------------------------------------------------------------------


def test_community_display_canonical_metadata_lookup():
    assert format_community_display("tamil_india") == "Tamil, India"
    assert format_community_display("arabic_morocco") == "Moroccan Arabic, Morocco"
    assert format_language_name("tamil_india") == "Tamil"
    assert format_country("tamil_india") == "India"


def test_master_prompt_uses_metadata_for_multi_underscore_community(tmp_path, monkeypatch):
    import riddles_framework.community_display as community_display

    metadata = tmp_path / "corpus_metadata.csv"
    metadata.write_text(
        "community,language,country,script_family\nafaan_oromo_oromo,Afaan Oromo,Ethiopia,Latin\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(community_display, "CORPUS_METADATA_PATH", metadata)
    row = sample_dataset_row()
    row["community"] = "afaan_oromo_oromo"
    row["language"] = "gaz"
    instance = BenchmarkInstance.from_dict(row)
    substitutions = build_substitutions(instance)
    assert substitutions["language_name"] == "Afaan Oromo"
    assert substitutions["country"] == "Ethiopia"
    assert substitutions["community_display"] == "Afaan Oromo, Ethiopia"
    assert substitutions["script_iso"] == "Latn"
    cfg = PromptConfig(
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        response_format=ResponseFormat.JSON,
    )
    prompt = build_prompt(instance, load_prompt_template("riddle_master", "v1"), cfg)
    assert "Afaan Oromo, Ethiopia (Latin script)" in prompt
    assert "(unspecified)" not in prompt


def test_community_display_fallback_for_unknown_community():
    # Unknown community keys should still render reasonably.
    assert format_community_display("klingon_qonos") == "Klingon, Qonos"
    # Bare key with no underscore.
    assert format_community_display("esperanto") == "Esperanto"
    # Empty string -> placeholder.
    assert format_community_display("") == "(unspecified)"


def test_build_substitutions_from_instance():
    inst = BenchmarkInstance.from_dict(sample_dataset_row())
    subs = build_substitutions(inst)
    assert subs["language"] == "tam"
    assert subs["language_name"] == "Tamil"
    assert subs["community"] == "tamil_india"
    assert subs["community_display"] == "Tamil, India"
    assert subs["country"] == "India"
    assert subs["script_iso"] == "Taml"
    assert subs["script_name"] == "Tamil script"
    assert subs["topic"] == "History"


def test_build_substitutions_missing_topic_uses_placeholder():
    row = sample_dataset_row()
    row["topic"] = None
    inst = BenchmarkInstance.from_dict(row)
    subs = build_substitutions(inst)
    assert subs["topic"] == "(unspecified)"


def test_template_with_placeholders_renders_with_cultural_context():
    inst = BenchmarkInstance.from_dict(sample_dataset_row())
    template = PromptTemplate(
        template_id="test",
        template_version="v1",
        instruction_text=(
            "You are solving a cultural riddle from {community_display} "
            "in {language_name}. Topic: {topic}."
        ),
    )
    cfg = PromptConfig(
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        response_format=ResponseFormat.FREE_FORM,
    )
    prompt = build_prompt(inst, template, cfg)
    assert "Tamil, India" in prompt
    assert "Tamil" in prompt
    assert "History" in prompt
    # Placeholder syntax should be gone after substitution.
    assert "{community_display}" not in prompt
    assert "{language_name}" not in prompt
    assert "{topic}" not in prompt


def test_template_without_placeholders_is_unchanged():
    """Backward compatibility: existing templates with no placeholders."""
    inst = BenchmarkInstance.from_dict(sample_dataset_row())
    template = PromptTemplate(
        template_id="test",
        template_version="v1",
        instruction_text="Solve this riddle.",
    )
    cfg = PromptConfig(
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        response_format=ResponseFormat.FREE_FORM,
    )
    prompt = build_prompt(inst, template, cfg)
    # Legacy RIDDLE: block should be present (no cultural context auto-injected).
    assert "RIDDLE:" in prompt
    assert "Solve this riddle." in prompt
    # No cultural metadata should leak into the prompt when not requested.
    assert "Tamil, India" not in prompt
    assert "History" not in prompt


def test_template_unknown_placeholder_left_intact():
    """A template can reference a placeholder we don't support without crashing."""
    inst = BenchmarkInstance.from_dict(sample_dataset_row())
    template = PromptTemplate(
        template_id="test",
        template_version="v1",
        instruction_text="Solve this {community_display} riddle. {made_up_placeholder}",
    )
    cfg = PromptConfig(
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        response_format=ResponseFormat.FREE_FORM,
    )
    prompt = build_prompt(inst, template, cfg)
    # Known placeholder substituted.
    assert "Tamil, India" in prompt
    # Unknown placeholder left as-is so the template author can spot it.
    assert "{made_up_placeholder}" in prompt


def test_template_using_riddle_text_placeholder_takes_full_control():
    """When the template uses {riddle_text}, the framework should not prepend RIDDLE:."""
    inst = BenchmarkInstance.from_dict(sample_dataset_row())
    template = PromptTemplate(
        template_id="test",
        template_version="v1",
        instruction_text='For {community_display}: "{riddle_text}"',
    )
    cfg = PromptConfig(
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        response_format=ResponseFormat.FREE_FORM,
    )
    prompt = build_prompt(inst, template, cfg)
    # Should NOT contain the legacy RIDDLE: header when template uses {riddle_text}.
    assert "RIDDLE:" not in prompt
    assert "Tamil, India" in prompt


# ---------------------------------------------------------------------------
# Script (ISO 15924) tests
# ---------------------------------------------------------------------------


def test_script_iso_known_communities():
    assert format_script_iso("tamil_india") == "Taml"
    assert format_script_iso("arabic_morocco") == "Arab"
    assert format_script_iso("hungarian_hungary") == "Latn"
    # Metadata identifies these as Han, without asserting a script variant.
    assert format_script_iso("chinese_china") == "Hani"
    assert format_script_iso("cantonese_china") == "Hani"
    assert format_script_iso("russian_russia") == "Cyrl"
    assert format_script_iso("thai_thailand") == "Thai"
    assert format_script_iso("sinhala_sri_lanka") == "Sinh"


def test_script_iso_unknown_community_returns_empty():
    """Unknown communities should return empty string so the template
    author can see the gap rather than getting a fabricated code."""
    assert format_script_iso("klingon_qonos") == ""
    assert format_script_iso("") == ""


def test_script_name_human_readable():
    assert format_script_name("arabic_morocco") == "Arabic script"
    assert format_script_name("tamil_india") == "Tamil script"
    assert format_script_name("english_usa") == "Latin script"
    assert format_script_name("russian_russia") == "Cyrillic script"
    # Unknown -> placeholder, not empty (so prompt prose still reads okay).
    assert format_script_name("klingon_qonos") == "(unspecified)"


def test_template_with_script_placeholder_renders_correctly():
    """Templates can use {script_name} to disambiguate scripts.

    Specifically Abdellah's case: 'arabic_morocco' should render as
    'Arabic script' so the model knows not to romanize.
    """
    row = sample_dataset_row()
    row["community"] = "arabic_morocco"
    row["language"] = "ary"
    inst = BenchmarkInstance.from_dict(row)
    template = PromptTemplate(
        template_id="test",
        template_version="v1",
        instruction_text="Riddle from {community_display}, written in {script_name}.",
    )
    cfg = PromptConfig(
        prompt_language="eng",
        riddle_language_presented=RiddleLanguagePresented.SOURCE,
        is_hint_provided=False,
        response_format=ResponseFormat.FREE_FORM,
    )
    prompt = build_prompt(inst, template, cfg)
    assert "Moroccan Arabic, Morocco" in prompt
    assert "Arabic script" in prompt
