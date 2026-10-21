from __future__ import annotations

from pathlib import Path
from typing import Any

from .constants import GenerationStatus
from .dataset_loader import load_benchmark_instances
from .exceptions import PromptBuildError, UnsupportedConfigurationError
from .prompts import build_condition_id, build_prompt
from .schemas import (
    GenerationResult,
    InstanceResult,
    PromptConfig,
    PromptTemplate,
    RunConfig,
    RunMetadata,
    utc_now_iso,
)
from .utils import make_uuid
from .writers import RunWriter


def validate_run_config(run_config: RunConfig) -> None:
    """Fail fast on invalid framework-level configurations.

    Currently rejects:
    - english + hints (open question; hints are source-language only in v1)
    """
    if run_config.riddle_language_presented.value == "english" and run_config.is_hint_provided:
        raise UnsupportedConfigurationError(
            "english + hints is not a supported benchmark condition"
        )


def validate_model_capabilities(model, run_config: RunConfig) -> None:
    caps = model.capabilities

    if run_config.reasoning_mode and not caps.supports_reasoning:
        raise UnsupportedConfigurationError(f"{model.model_id} does not support reasoning")
    if run_config.search_enabled and not caps.supports_search:
        raise UnsupportedConfigurationError(f"{model.model_id} does not support search")
    if (
        run_config.response_format.value in {"json", "json_schema"}
        and not caps.supports_structured_output
    ):
        raise UnsupportedConfigurationError(f"{model.model_id} does not support structured output")
    if run_config.seed is not None and not caps.supports_seed:
        raise UnsupportedConfigurationError(f"{model.model_id} does not support seed")
    if run_config.batch_mode and not caps.supports_batch:
        raise UnsupportedConfigurationError(f"{model.model_id} does not support batch mode")


class BenchmarkRunner:
    def __init__(
        self,
        *,
        run_config: RunConfig,
        prompt_template: PromptTemplate,
        model,
        output_root: str | Path = "data/runs",
        max_instances: int | None = None,
    ) -> None:
        self.run_config = run_config
        self.prompt_template = prompt_template
        self.model = model
        self.output_root = Path(output_root)
        self.max_instances = max_instances

    def _update_metadata_counts(
        self,
        metadata: RunMetadata,
        gen: GenerationResult,
    ) -> None:
        if gen.status == GenerationStatus.SUCCESS:
            metadata.completed_instances += 1
        elif gen.status in {GenerationStatus.SKIPPED, GenerationStatus.UNSUPPORTED_CONFIG}:
            metadata.skipped_instances += 1
        else:
            metadata.error_instances += 1

    def _generate_one(self, prompt: str) -> GenerationResult:
        return self.model.generate(
            prompt,
            temperature=self.run_config.temperature,
            max_output_tokens=self.run_config.max_output_tokens,
            seed=self.run_config.seed,
            reasoning_mode=self.run_config.reasoning_mode,
            search_enabled=self.run_config.search_enabled,
            response_format=self.run_config.response_format.value,
            model_params=dict(self.run_config.model_params),
        )

    def _generate_batch(self, prompts: list[str]) -> list[GenerationResult]:
        generations = self.model.generate_batch(
            prompts,
            temperature=self.run_config.temperature,
            max_output_tokens=self.run_config.max_output_tokens,
            seed=self.run_config.seed,
            reasoning_mode=self.run_config.reasoning_mode,
            search_enabled=self.run_config.search_enabled,
            response_format=self.run_config.response_format.value,
            model_params=dict(self.run_config.model_params),
        )
        if len(generations) != len(prompts):
            raise ValueError(
                "model.generate_batch returned "
                f"{len(generations)} results for {len(prompts)} prompts"
            )
        return generations

    def _append_generated_result(
        self,
        *,
        writer: RunWriter,
        metadata: RunMetadata,
        instance,
        run_id: str,
        condition_id: str,
        prompt: str,
        gen: GenerationResult,
    ) -> None:
        result = self._build_instance_result(
            instance=instance,
            run_id=run_id,
            condition_id=condition_id,
            model_input=prompt,
            gen=gen,
        )
        writer.append_result(result.to_dict())
        self._update_metadata_counts(metadata, gen)

    def _append_skipped_result(
        self,
        *,
        writer: RunWriter,
        metadata: RunMetadata,
        instance,
        run_id: str,
        condition_id: str,
        skipped_reason: str,
    ) -> None:
        result = self._build_instance_result(
            instance=instance,
            run_id=run_id,
            condition_id=condition_id,
            model_input="",
            skipped_reason=skipped_reason,
        )
        writer.append_result(result.to_dict())
        metadata.skipped_instances += 1

    def _process_batch(
        self,
        *,
        writer: RunWriter,
        metadata: RunMetadata,
        batch: list[tuple[Any, str]],
        run_id: str,
        condition_id: str,
    ) -> None:
        prompts = [prompt for _, prompt in batch]
        try:
            generations = self._generate_batch(prompts)
        except Exception as exc:
            generations = [
                GenerationResult(
                    raw_response=None,
                    status=GenerationStatus.ERROR,
                    error_message=str(exc),
                )
                for _ in prompts
            ]

        for (instance, prompt), gen in zip(batch, generations):
            if self.run_config.strict_uploadable_rows and (
                gen.status != GenerationStatus.SUCCESS or not str(gen.raw_response or "").strip()
            ):
                raise RuntimeError(
                    "batch generation produced a non-uploadable row: "
                    f"riddle_id={instance.riddle_id} status={gen.status.value}"
                )
            self._append_generated_result(
                writer=writer,
                metadata=metadata,
                instance=instance,
                run_id=run_id,
                condition_id=condition_id,
                prompt=prompt,
                gen=gen,
            )

    def _build_instance_result(
        self,
        *,
        instance,
        run_id: str,
        condition_id: str,
        model_input: str,
        gen: GenerationResult | None = None,
        skipped_reason: str | None = None,
    ) -> InstanceResult:
        if gen is None:
            return InstanceResult(
                instance_id=make_uuid(),
                run_id=run_id,
                riddle_id=instance.riddle_id,
                condition_id=condition_id,
                model_input=model_input,
                raw_response=None,
                generation_status=GenerationStatus.SKIPPED.value,
                error_message=skipped_reason,
                input_tokens=None,
                output_tokens=None,
                latency_ms=None,
                model_name=self.model.model_name,
                model_id=self.model.model_id,
                model_provider=self.model.model_provider,
                provider_model_name=self.model.provider_model_name,
                access_pattern=self.model.access_pattern,
                prompt_template_id=self.prompt_template.template_id,
                prompt_template_version=self.prompt_template.template_version,
                prompt_language=self.run_config.prompt_language,
                riddle_language_presented=self.run_config.riddle_language_presented.value,
                is_hint_provided=self.run_config.is_hint_provided,
                reasoning_mode=self.run_config.reasoning_mode,
                search_enabled=self.run_config.search_enabled,
                response_format=self.run_config.response_format.value,
                batch_size=self.run_config.batch_size,
                temperature=self.run_config.temperature,
                max_output_tokens=self.run_config.max_output_tokens,
                seed=self.run_config.seed,
                model_params=dict(self.run_config.model_params),
                cache_enabled=self.run_config.cache_enabled,
                cache_backend=self.run_config.cache_backend,
                cache_namespace=self.run_config.cache_namespace,
                instance_language=instance.language,
                instance_community=instance.community,
                instance_run_batch=instance.run_batch,
                instance_topic=instance.topic,
                reference_answer_original=instance.reference_answer_original,
                reference_answer_english=instance.reference_answer_english,
            )

        return InstanceResult(
            instance_id=make_uuid(),
            run_id=run_id,
            riddle_id=instance.riddle_id,
            condition_id=condition_id,
            model_input=model_input,
            raw_response=gen.raw_response,
            generation_status=gen.status.value,
            error_message=gen.error_message,
            input_tokens=gen.input_tokens,
            output_tokens=gen.output_tokens,
            latency_ms=gen.latency_ms,
            provider_details=gen.provider_details,
            model_name=self.model.model_name,
            model_id=self.model.model_id,
            model_provider=self.model.model_provider,
            provider_model_name=self.model.provider_model_name,
            access_pattern=self.model.access_pattern,
            prompt_template_id=self.prompt_template.template_id,
            prompt_template_version=self.prompt_template.template_version,
            prompt_language=self.run_config.prompt_language,
            riddle_language_presented=self.run_config.riddle_language_presented.value,
            is_hint_provided=self.run_config.is_hint_provided,
            reasoning_mode=self.run_config.reasoning_mode,
            search_enabled=self.run_config.search_enabled,
            response_format=self.run_config.response_format.value,
            batch_size=self.run_config.batch_size,
            temperature=self.run_config.temperature,
            max_output_tokens=self.run_config.max_output_tokens,
            seed=self.run_config.seed,
            model_params=dict(self.run_config.model_params),
            cache_enabled=self.run_config.cache_enabled,
            cache_backend=self.run_config.cache_backend,
            cache_namespace=self.run_config.cache_namespace,
            instance_language=instance.language,
            instance_community=instance.community,
            instance_run_batch=instance.run_batch,
            instance_topic=instance.topic,
            reference_answer_original=instance.reference_answer_original,
            reference_answer_english=instance.reference_answer_english,
        )

    def run(self) -> str:
        # Validate config BEFORE touching the filesystem.
        validate_run_config(self.run_config)

        run_id = make_uuid()
        run_dir = self.output_root / run_id
        writer = RunWriter(run_dir)

        instances = load_benchmark_instances(
            self.run_config.dataset_path,
            run_batches=self.run_config.run_batches,
        )
        if self.max_instances is not None:
            instances = instances[: self.max_instances]

        metadata = RunMetadata(
            run_id=run_id,
            run_started_at=utc_now_iso(),
            run_completed_at=None,
            runner_name=self.run_config.runner_name,
            runner_version=self.run_config.runner_version,
            model_name=self.model.model_name,
            model_id=self.model.model_id,
            model_provider=self.model.model_provider,
            provider_model_name=self.model.provider_model_name,
            access_pattern=self.model.access_pattern,
            dataset_version=self.run_config.dataset_version,
            total_instances=len(instances),
            run_batches=self.run_config.run_batches,
            batch_mode_requested=self.run_config.batch_mode,
            batch_mode_used=False,
            cache_enabled=self.run_config.cache_enabled,
            cache_backend=self.run_config.cache_backend,
            cache_namespace=self.run_config.cache_namespace,
            cache_dir=self.run_config.cache_dir,
            batch_size=self.run_config.batch_size,
            prompt_template_id=self.run_config.prompt_template_id,
            prompt_template_version=self.run_config.prompt_template_version,
            prompt_language=self.run_config.prompt_language,
            riddle_language_presented=self.run_config.riddle_language_presented.value,
            is_hint_provided=self.run_config.is_hint_provided,
            reasoning_mode=self.run_config.reasoning_mode,
            search_enabled=self.run_config.search_enabled,
            response_format=self.run_config.response_format.value,
            temperature=self.run_config.temperature,
            max_output_tokens=self.run_config.max_output_tokens,
            seed=self.run_config.seed,
            model_params=dict(self.run_config.model_params),
            notes=self.run_config.notes,
        )

        writer.write_run_config(self.run_config.to_dict())

        try:
            validate_model_capabilities(self.model, self.run_config)
        except UnsupportedConfigurationError as exc:
            metadata.run_completed_at = utc_now_iso()
            metadata.error_instances = len(instances)
            writer.write_run_metadata(metadata.to_dict())
            writer.write_manifest(
                {
                    "run_id": run_id,
                    "status": "failed_before_start",
                    "reason": str(exc),
                    "total_instances": len(instances),
                }
            )
            return run_id

        prompt_config = PromptConfig(
            prompt_language=self.run_config.prompt_language,
            riddle_language_presented=self.run_config.riddle_language_presented,
            is_hint_provided=self.run_config.is_hint_provided,
            response_format=self.run_config.response_format,
            reasoning_mode=self.run_config.reasoning_mode,
        )

        condition_id = build_condition_id(
            self.run_config.riddle_language_presented,
            self.run_config.is_hint_provided,
        )

        if self.run_config.batch_mode:
            metadata.batch_mode_used = True
            batch_size = self.run_config.batch_size or 1
            batch: list[tuple[Any, str]] = []

            for instance in instances:
                try:
                    prompt = build_prompt(instance, self.prompt_template, prompt_config)
                except PromptBuildError as exc:
                    self._append_skipped_result(
                        writer=writer,
                        metadata=metadata,
                        instance=instance,
                        run_id=run_id,
                        condition_id=condition_id,
                        skipped_reason=str(exc),
                    )
                    continue

                batch.append((instance, prompt))
                if len(batch) >= batch_size:
                    self._process_batch(
                        writer=writer,
                        metadata=metadata,
                        batch=batch,
                        run_id=run_id,
                        condition_id=condition_id,
                    )
                    batch = []

            if batch:
                self._process_batch(
                    writer=writer,
                    metadata=metadata,
                    batch=batch,
                    run_id=run_id,
                    condition_id=condition_id,
                )
        else:
            for instance in instances:
                try:
                    prompt = build_prompt(instance, self.prompt_template, prompt_config)
                except PromptBuildError as exc:
                    self._append_skipped_result(
                        writer=writer,
                        metadata=metadata,
                        instance=instance,
                        run_id=run_id,
                        condition_id=condition_id,
                        skipped_reason=str(exc),
                    )
                    continue

                try:
                    gen = self._generate_one(prompt)
                except Exception as exc:
                    gen = GenerationResult(
                        raw_response=None,
                        status=GenerationStatus.ERROR,
                        error_message=str(exc),
                    )

                self._append_generated_result(
                    writer=writer,
                    metadata=metadata,
                    instance=instance,
                    run_id=run_id,
                    condition_id=condition_id,
                    prompt=prompt,
                    gen=gen,
                )

        metadata.run_completed_at = utc_now_iso()
        writer.write_run_metadata(metadata.to_dict())
        writer.write_manifest(
            {
                "run_id": run_id,
                "status": "completed",
                "model_name": self.model.model_name,
                "model_id": self.model.model_id,
                "run_batches": metadata.run_batches,
                "total_instances": metadata.total_instances,
                "completed_instances": metadata.completed_instances,
                "skipped_instances": metadata.skipped_instances,
                "error_instances": metadata.error_instances,
                "batch_mode_used": metadata.batch_mode_used,
                "batch_size": metadata.batch_size,
                "cache_enabled": metadata.cache_enabled,
                "cache_namespace": metadata.cache_namespace,
            }
        )
        return run_id
