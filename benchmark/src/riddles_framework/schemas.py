from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .constants import GenerationStatus, ResponseFormat, RiddleLanguagePresented


def _safe_format(text: str, subs: dict[str, str]) -> str:
    """Apply {placeholder} substitutions, leaving unknown ones alone.

    Equivalent to str.format(**subs) but never raises KeyError for
    placeholders that aren't in `subs`. Used by PromptTemplate.render
    so templates can declare placeholders the caller doesn't always
    provide (e.g. a topic placeholder when topic is None).
    """
    import string

    formatter = string.Formatter()
    out: list[str] = []
    for literal, field_name, format_spec, conversion in formatter.parse(text):
        if literal:
            out.append(literal)
        if field_name is None:
            continue
        if field_name in subs:
            value = subs[field_name]
            if conversion:
                value = formatter.convert_field(value, conversion)
            if format_spec:
                value = format(value, format_spec)
            out.append(value)
        else:
            # Leave unknown placeholder intact, including any format spec.
            spec_part = f":{format_spec}" if format_spec else ""
            conv_part = f"!{conversion}" if conversion else ""
            out.append("{" + field_name + conv_part + spec_part + "}")
    return "".join(out)


@dataclass
class BenchmarkInstance:
    riddle_id: str
    source_row_number: int | None
    language: str
    community: str
    topic: str | None
    is_wildcard: bool

    riddle_original: str
    riddle_english: str | None
    riddle_hint: str | None
    riddle_note: str | None

    reference_answer_original: str
    reference_answer_english: str | None

    run_batch: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkInstance":
        return cls(
            riddle_id=data["riddle_id"],
            source_row_number=data.get("source_row_number"),
            language=data["language"],
            community=data["community"],
            topic=data.get("topic"),
            is_wildcard=bool(data.get("is_wildcard", False)),
            riddle_original=str(data["riddle_original"]),
            riddle_english=str(data["riddle_english"])
            if data.get("riddle_english") is not None
            else None,
            riddle_hint=str(data["riddle_hint"]) if data.get("riddle_hint") is not None else None,
            riddle_note=str(data["riddle_note"]) if data.get("riddle_note") is not None else None,
            reference_answer_original=str(data["reference_answer_original"]),
            reference_answer_english=str(data["reference_answer_english"])
            if data.get("reference_answer_english") is not None
            else None,
            run_batch=data.get("run_batch"),
        )

    def validate(self) -> None:
        if not self.riddle_id:
            raise ValueError("missing riddle_id")
        if not self.language:
            raise ValueError(f"{self.riddle_id}: missing language")
        if not self.community:
            raise ValueError(f"{self.riddle_id}: missing community")
        if not self.riddle_original or not self.riddle_original.strip():
            raise ValueError(f"{self.riddle_id}: missing riddle_original")
        if not self.reference_answer_original or not self.reference_answer_original.strip():
            raise ValueError(f"{self.riddle_id}: missing reference_answer_original")


@dataclass
class ModelCapabilities:
    supports_reasoning: bool = False
    supports_search: bool = False
    supports_structured_output: bool = False
    supports_batch: bool = False
    supports_seed: bool = False


@dataclass
class ModelSpec:
    model_name: str
    model_id: str
    provider: str
    access_pattern: str
    provider_model_name: str
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    size_label: str | None = None
    closed_source: bool | None = None
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["capabilities"] = asdict(self.capabilities)
        return d


@dataclass
class PromptTemplate:
    template_id: str
    template_version: str
    instruction_text: str
    # Templates can declare a recommended response format. Informational only;
    # the actual response_format used is whatever RunConfig says.
    recommended_response_format: ResponseFormat | None = None

    def render(
        self,
        *,
        riddle_text: str,
        hint_text: str | None = None,
        substitutions: dict[str, str] | None = None,
    ) -> str:
        """Render the final prompt string.

        Templates can opt in to cultural-context injection by including
        Python str.format-style placeholders in instruction_text, e.g.:

            instruction_text: |
              You are solving a cultural riddle from {community_display}
              (written in {language_name}). The topic is {topic}.

        Supported placeholders (all optional; templates can use any subset):
            {language}            ISO 639-3 code, e.g. "tam"
            {language_name}       human-readable language, e.g. "Tamil"
            {community}           community key, e.g. "tamil_india"
            {community_display}   curated readable form, e.g. "Tamil, India"
            {country}             country portion, e.g. "India"
            {script_iso}          ISO 15924 code, e.g. "Taml"
            {script_name}         readable script, e.g. "Tamil script"
            {topic}               riddle topic, or "(unspecified)" if missing
            {riddle_text}         the riddle text (source or English)
            {hint_text}           the hint text, or empty string if absent

        Backward compatibility: if a template does NOT use {riddle_text}
        and {hint_text} placeholders, the framework prepends the legacy
        RIDDLE:/HINT: blocks the way it always has. This means existing
        templates without placeholders keep working unchanged.
        """
        subs = dict(substitutions or {})
        subs.setdefault("riddle_text", riddle_text)
        subs.setdefault("hint_text", hint_text or "")

        # Apply placeholders if the template uses them. Use a safe
        # substituter that leaves unknown placeholders alone rather than
        # raising KeyError, so a template using {language} doesn't crash
        # if a future caller forgets to pass it.
        instruction = _safe_format(self.instruction_text, subs)

        # If the template author chose to handle {riddle_text} themselves,
        # skip the legacy RIDDLE:/HINT: prepending entirely — they're in
        # full control of layout.
        if "{riddle_text}" in self.instruction_text:
            return instruction

        if hint_text:
            return f"{instruction}\n\nRIDDLE:\n{riddle_text}\n\nHINT:\n{hint_text}\n"
        return f"{instruction}\n\nRIDDLE:\n{riddle_text}\n"


@dataclass
class PromptConfig:
    prompt_language: str
    riddle_language_presented: RiddleLanguagePresented
    is_hint_provided: bool
    response_format: ResponseFormat = ResponseFormat.JSON
    reasoning_mode: str | None = None


@dataclass
class RunConfig:
    model_spec: ModelSpec
    runner_name: str

    dataset_path: str
    dataset_version: str

    prompt_template_id: str
    prompt_template_version: str
    prompt_language: str
    riddle_language_presented: RiddleLanguagePresented
    is_hint_provided: bool

    reasoning_mode: str | None = None
    search_enabled: bool = False
    # Default to structured output per project guidance.
    response_format: ResponseFormat = ResponseFormat.JSON
    batch_mode: bool = False
    batch_size: int | None = None
    # Optional run-batch filter. None runs all batches; e.g. ["v1"] runs only
    # the v1 communities. Overridden by the CLI --run-batch flag.
    run_batches: list[str] | None = None

    # Temperature is intentionally Optional and defaults to None.
    # Reasoning models (OpenAI o-series, Anthropic thinking, DeepSeek R1) can
    # fall into degenerate loops at low temperature, so the framework relies
    # on the model's default unless temperature is explicitly set.
    temperature: float | None = None
    max_output_tokens: int = 256
    seed: int | None = None

    # Arbitrary provider-specific parameters passed through to the adapter.
    # Merged LAST in the request kwargs so it overrides framework-set values.
    # Use this for: top_p, frequency_penalty, presence_penalty, top_k, etc.
    model_params: dict[str, Any] = field(default_factory=dict)

    cache_enabled: bool = False
    cache_backend: str | None = None
    cache_namespace: str | None = None
    cache_dir: str | None = None
    local_cache_client_mode: str = "sync"

    api_key_env_var: str | None = None
    base_url: str | None = None

    runner_version: str = "v1.0.0"
    notes: str | None = None
    strict_capability_check: bool = True
    # Abort a batch run on any non-success/empty generation instead of
    # recording it. Off by default: several models legitimately produce
    # empties/truncations, and analysis records those as categories.
    strict_uploadable_rows: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["model_spec"] = self.model_spec.to_dict()
        d["riddle_language_presented"] = self.riddle_language_presented.value
        d["response_format"] = self.response_format.value
        return d


@dataclass
class RunMetadata:
    run_id: str
    run_started_at: str
    run_completed_at: str | None

    runner_name: str
    runner_version: str

    model_name: str
    model_id: str
    model_provider: str
    provider_model_name: str
    access_pattern: str

    dataset_version: str
    total_instances: int
    completed_instances: int = 0
    skipped_instances: int = 0
    error_instances: int = 0

    batch_mode_requested: bool = False
    batch_mode_used: bool = False
    batch_size: int | None = None
    run_batches: list[str] | None = None

    cache_enabled: bool = False
    cache_backend: str | None = None
    cache_namespace: str | None = None
    cache_dir: str | None = None
    local_cache_client_mode: str = "sync"

    prompt_template_id: str | None = None
    prompt_template_version: str | None = None
    prompt_language: str | None = None
    riddle_language_presented: str | None = None
    is_hint_provided: bool | None = None
    reasoning_mode: str | None = None
    search_enabled: bool = False
    response_format: str = ResponseFormat.JSON.value
    temperature: float | None = None
    max_output_tokens: int = 256
    seed: int | None = None
    model_params: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GenerationResult:
    raw_response: str | None
    status: GenerationStatus
    error_message: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: int | None = None
    reasoning_artifact: str | None = None
    reasoning_artifact_type: str | None = None
    provider_details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class InstanceResult:
    instance_id: str
    run_id: str
    riddle_id: str
    condition_id: str

    model_input: str
    raw_response: str | None
    generation_status: str
    error_message: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None

    parsed_json: dict[str, Any] | None = None
    parse_status: str | None = None
    parse_error: str | None = None
    provider_details: dict[str, Any] | None = None

    model_name: str | None = None
    model_id: str | None = None
    model_provider: str | None = None
    provider_model_name: str | None = None
    access_pattern: str | None = None

    prompt_template_id: str | None = None
    prompt_template_version: str | None = None
    prompt_language: str | None = None
    riddle_language_presented: str | None = None
    is_hint_provided: bool | None = None
    reasoning_mode: str | None = None
    search_enabled: bool = False
    response_format: str = ResponseFormat.JSON.value
    batch_size: int | None = None
    temperature: float | None = None
    max_output_tokens: int = 256
    seed: int | None = None
    model_params: dict[str, Any] = field(default_factory=dict)

    cache_enabled: bool = False
    cache_backend: str | None = None
    cache_namespace: str | None = None
    local_cache_client_mode: str = "sync"

    instance_language: str | None = None
    instance_community: str | None = None
    instance_run_batch: str | None = None
    instance_topic: str | None = None
    reference_answer_original: str | None = None
    reference_answer_english: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
