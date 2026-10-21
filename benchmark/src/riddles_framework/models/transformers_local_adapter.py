from __future__ import annotations

import asyncio
import inspect
import time
from typing import Any

from ..cache_clients import make_transformers_client
from ..constants import GenerationStatus
from ..schemas import GenerationResult, ModelSpec
from ..utils import looks_like_refusal
from .base import Model


class LocalTransformersAdapter(Model):
    def __init__(self, spec: ModelSpec, cfg: dict) -> None:
        super().__init__(spec)
        self.use_cache = cfg.get("cache_enabled", False)
        self.batch_size = cfg.get("batch_size") or 1
        self.cache_client_mode = cfg.get("local_cache_client_mode", "sync")
        if self.cache_client_mode not in {"sync", "async"}:
            raise ValueError("local_cache_client_mode must be 'sync' or 'async'")

        from transformers import AutoTokenizer

        configured_model_params = cfg.get("model_params") or {}
        self.model_load_params = self._normalize_model_load_params(
            configured_model_params.get("model_load", {})
        )
        self.tokenizer_load_params = dict(configured_model_params.get("tokenizer_load", {}))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.provider_model_name,
            **self.tokenizer_load_params,
        )

        if self.use_cache:
            client_kwargs: dict[str, Any] = {
                "model_name": self.provider_model_name,
                "client_mode": self.cache_client_mode,
                "batch_size": self.batch_size,
                **self.model_load_params,
            }
            if cfg.get("cache_namespace") is not None:
                client_kwargs["namespace"] = cfg["cache_namespace"]
            if cfg.get("cache_dir") is not None:
                client_kwargs["cache_dir"] = cfg["cache_dir"]

            self.client = make_transformers_client(True, **client_kwargs)
            self.cache_client_interface = getattr(
                self.client,
                "_riddles_transformers_interface",
                "chat",
            )
            self.pipeline = None
        else:
            from transformers import pipeline

            self.pipeline = pipeline(
                "text-generation",
                model=self.provider_model_name,
                tokenizer=self.tokenizer,
                model_kwargs=self.model_load_params,
            )
            self.client = None
            self.cache_client_interface = None

    def _normalize_model_load_params(self, params: dict[str, Any]) -> dict[str, Any]:
        load_params = dict(params or {})

        # Transformers 5.x renamed `torch_dtype` to `dtype`; accept the old key as
        # a deprecated alias so existing configs keep working.
        if "torch_dtype" in load_params and "dtype" not in load_params:
            load_params["dtype"] = load_params.pop("torch_dtype")
        else:
            load_params.pop("torch_dtype", None)

        dtype = load_params.get("dtype")
        if isinstance(dtype, str):
            # Pass the dtype as a string; Transformers resolves it to a torch dtype
            # internally, so the adapter never needs to import torch.
            aliases = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}
            canonical = aliases.get(dtype, dtype)
            if canonical not in {"auto", "float16", "bfloat16", "float32"}:
                raise ValueError(
                    "model_params.model_load.dtype must be one of "
                    "auto, float16, bfloat16, or float32"
                )
            load_params["dtype"] = canonical
        return load_params

    def _generation_model_params(
        self,
        model_params: dict[str, Any] | None,
    ) -> dict[str, Any]:
        generation_params = dict(model_params or {})
        generation_params.pop("model_load", None)
        generation_params.pop("tokenizer_load", None)
        return generation_params

    def _token_count(self, text: str | None) -> int | None:
        if text is None:
            return None
        try:
            tokenized = self.tokenizer(text, add_special_tokens=False)
            return len(tokenized.get("input_ids", []))
        except Exception:
            return None

    def _encode_prompt(self, prompt: str) -> dict[str, Any]:
        tokenized = self.tokenizer(prompt, return_tensors="pt")
        model = getattr(getattr(self.client, "model", None), "model", None)
        device = getattr(model, "device", None)
        if device is None and hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except Exception:
                device = None
        if device is not None:
            tokenized = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in tokenized.items()
            }
        return dict(tokenized)

    def _encode_prompts(self, prompts: list[str]) -> dict[str, Any]:
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

        tokenized = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        )
        model = getattr(getattr(self.client, "model", None), "model", None)
        device = getattr(model, "device", None)
        if device is None and hasattr(model, "parameters"):
            try:
                device = next(model.parameters()).device
            except Exception:
                device = None
        if device is not None:
            tokenized = {
                key: value.to(device) if hasattr(value, "to") else value
                for key, value in tokenized.items()
            }
        return dict(tokenized)

    def _input_token_count(self, encoded: dict[str, Any], prompt: str) -> int | None:
        input_ids = encoded.get("input_ids")
        try:
            return int(input_ids.shape[-1])
        except Exception:
            return self._token_count(prompt)

    def _batch_input_token_counts(
        self,
        encoded: dict[str, Any],
        prompts: list[str],
    ) -> list[int | None]:
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            try:
                return [int(value) for value in attention_mask.sum(dim=1).tolist()]
            except Exception:
                pass
        return [self._token_count(prompt) for prompt in prompts]

    def _decode_generated_text(self, generated_ids: Any, input_tokens: int | None) -> str | None:
        if (
            isinstance(generated_ids, (list, tuple))
            and generated_ids
            and isinstance(generated_ids[0], (list, tuple))
        ):
            sequence = generated_ids[0]
        else:
            try:
                sequence = (
                    generated_ids[0] if getattr(generated_ids, "ndim", 1) > 1 else generated_ids
                )
            except Exception:
                sequence = generated_ids

        try:
            if input_tokens is not None:
                sequence = sequence[input_tokens:]
        except Exception:
            pass

        try:
            return self.tokenizer.decode(sequence, skip_special_tokens=True)
        except Exception:
            return str(generated_ids)

    def _decode_generated_texts(
        self,
        generated_ids: Any,
        encoded: dict[str, Any],
        prompt_count: int,
    ) -> list[str | None]:
        input_ids = encoded.get("input_ids")
        try:
            input_width = int(input_ids.shape[-1])
        except Exception:
            input_width = None

        texts: list[str | None] = []
        for index in range(prompt_count):
            try:
                sequence = generated_ids[index]
            except Exception:
                texts.append(str(generated_ids))
                continue

            try:
                if input_width is not None:
                    sequence = sequence[input_width:]
            except Exception:
                pass

            try:
                texts.append(self.tokenizer.decode(sequence, skip_special_tokens=True))
            except Exception:
                texts.append(str(sequence))
        return texts

    def _extract_text_from_chat_response(self, response: Any) -> str | None:
        if response is None:
            return None
        if isinstance(response, str):
            return response
        if isinstance(response, dict):
            if isinstance(response.get("generated_text"), str):
                return response["generated_text"]
            choices = response.get("choices")
            if choices:
                message = choices[0].get("message", {}) if isinstance(choices[0], dict) else None
                if isinstance(message, dict):
                    return message.get("content")
        if isinstance(response, list) and response:
            return self._extract_text_from_chat_response(response[0])

        choices = getattr(response, "choices", None)
        if choices:
            message = getattr(choices[0], "message", None)
            text = getattr(message, "content", None)
            if text is not None:
                return text

        text = getattr(response, "generated_text", None)
        if text is not None:
            return text

        return str(response)

    def _result_from_text(
        self,
        text: str | None,
        *,
        start: float,
        input_tokens: int | None,
        provider_details: dict[str, Any] | None,
        error_message: str | None = None,
    ) -> GenerationResult:
        latency_ms = int((time.time() - start) * 1000)
        if error_message is not None:
            return GenerationResult(
                raw_response=None,
                status=GenerationStatus.ERROR,
                error_message=error_message,
                input_tokens=input_tokens,
                latency_ms=latency_ms,
                provider_details=provider_details,
            )

        status = GenerationStatus.SUCCESS
        if text is None or str(text).strip() == "":
            status = GenerationStatus.EMPTY
        elif looks_like_refusal(text):
            status = GenerationStatus.REFUSAL

        return GenerationResult(
            raw_response=text,
            status=status,
            input_tokens=input_tokens,
            output_tokens=self._token_count(text),
            latency_ms=latency_ms,
            provider_details=provider_details,
        )

    def _provider_details(
        self,
        *,
        cache_metadata: Any = None,
        batch_count: int | None = None,
    ) -> dict[str, Any]:
        details = {
            "model": self.provider_model_name,
            "cache_enabled": self.use_cache,
            "batch_size": self.batch_size,
            "local_cache_client_mode": self.cache_client_mode if self.client is not None else None,
            "cache_client_interface": self.cache_client_interface
            if self.client is not None
            else None,
            "model_load_params": {key: str(value) for key, value in self.model_load_params.items()},
            "batch_count": batch_count,
        }
        if self.client is not None:
            details["cache_metadata"] = {
                "cached": getattr(cache_metadata, "cached", None),
                "duplicated": getattr(cache_metadata, "duplicated", None),
            }
        else:
            details["cache_metadata"] = None
        return details

    def _call_cached_chat_client(
        self,
        prompt: str,
        *,
        temperature: float | None,
        max_output_tokens: int,
        model_params: dict[str, Any] | None,
    ) -> tuple[str | None, Any]:
        request_kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": prompt}],
            "max_new_tokens": max_output_tokens,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature
            request_kwargs["do_sample"] = temperature > 0.0
        else:
            request_kwargs["do_sample"] = False

        if model_params:
            request_kwargs.update(model_params)

        result = self.client.chat.completions.create(
            **request_kwargs,
            metadata=True,
        )
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        response, cache_metadata = result
        return self._extract_text_from_chat_response(response), cache_metadata

    def _call_cached_generate_client(
        self,
        prompt: str,
        *,
        temperature: float | None,
        max_output_tokens: int,
        model_params: dict[str, Any] | None,
    ) -> tuple[str | None, Any, int | None]:
        encoded = self._encode_prompt(prompt)
        input_tokens = self._input_token_count(encoded, prompt)
        request_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max_output_tokens,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature
            request_kwargs["do_sample"] = temperature > 0.0
        else:
            request_kwargs["do_sample"] = False

        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            request_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        elif getattr(self.tokenizer, "eos_token_id", None) is not None:
            request_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        if model_params:
            request_kwargs.update(model_params)

        generated_ids, cache_metadata = self.client.chat.completions.create(
            **request_kwargs,
            metadata=True,
        )
        text = self._decode_generated_text(generated_ids, input_tokens)
        return text, cache_metadata, input_tokens

    def _call_cached_generate_client_batch(
        self,
        prompts: list[str],
        *,
        temperature: float | None,
        max_output_tokens: int,
        model_params: dict[str, Any] | None,
    ) -> tuple[list[str | None], Any, list[int | None]]:
        encoded = self._encode_prompts(prompts)
        input_tokens = self._batch_input_token_counts(encoded, prompts)
        request_kwargs: dict[str, Any] = {
            **encoded,
            "max_new_tokens": max_output_tokens,
        }

        if temperature is not None:
            request_kwargs["temperature"] = temperature
            request_kwargs["do_sample"] = temperature > 0.0
        else:
            request_kwargs["do_sample"] = False

        if getattr(self.tokenizer, "pad_token_id", None) is not None:
            request_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        elif getattr(self.tokenizer, "eos_token_id", None) is not None:
            request_kwargs.setdefault("pad_token_id", self.tokenizer.eos_token_id)

        if model_params:
            request_kwargs.update(model_params)

        generated_ids, cache_metadata = self.client.chat.completions.create(
            **request_kwargs,
            metadata=True,
        )
        texts = self._decode_generated_texts(generated_ids, encoded, len(prompts))
        return texts, cache_metadata, input_tokens

    def _call_pipeline_batch(
        self,
        prompts: list[str],
        *,
        temperature: float | None,
        max_output_tokens: int,
        model_params: dict[str, Any] | None,
    ) -> list[str | None]:
        pipe_kwargs: dict[str, Any] = {
            "max_new_tokens": max_output_tokens,
            "return_full_text": False,
            "batch_size": self.batch_size,
        }
        if temperature is not None:
            pipe_kwargs["temperature"] = temperature
            pipe_kwargs["do_sample"] = temperature > 0.0
        else:
            pipe_kwargs["do_sample"] = False

        if model_params:
            pipe_kwargs.update(model_params)

        outputs = self.pipeline(prompts, **pipe_kwargs)
        texts: list[str | None] = []
        for output in outputs:
            item = output[0] if isinstance(output, list) and output else output
            if isinstance(item, dict):
                texts.append(item.get("generated_text"))
            else:
                texts.append(str(item) if item is not None else None)
        return texts

    def generate(
        self,
        prompt: str,
        *,
        temperature: float | None = None,
        max_output_tokens: int = 256,
        seed: int | None = None,
        reasoning_mode: str | None = None,
        search_enabled: bool = False,
        response_format: str = "json",
        model_params: dict[str, Any] | None = None,
    ) -> GenerationResult:
        start = time.time()
        try:
            generation_model_params = self._generation_model_params(model_params)

            if seed is not None:
                from transformers import set_seed

                set_seed(seed)

            if self.client is not None:
                if self.cache_client_interface == "generate":
                    text, cache_metadata, input_tokens = self._call_cached_generate_client(
                        prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        model_params=generation_model_params,
                    )
                else:
                    text, cache_metadata = self._call_cached_chat_client(
                        prompt,
                        temperature=temperature,
                        max_output_tokens=max_output_tokens,
                        model_params=generation_model_params,
                    )
                    input_tokens = self._token_count(prompt)
                output_tokens = self._token_count(text)
            else:
                pipe_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_output_tokens,
                    "return_full_text": False,
                }
                if temperature is not None:
                    pipe_kwargs["temperature"] = temperature
                    pipe_kwargs["do_sample"] = temperature > 0.0
                else:
                    # Greedy decoding by default; no sampling.
                    pipe_kwargs["do_sample"] = False

                if generation_model_params:
                    pipe_kwargs.update(generation_model_params)

                outputs = self.pipeline(prompt, **pipe_kwargs)
                text = outputs[0]["generated_text"] if outputs else None
                input_tokens = self._token_count(prompt)
                output_tokens = self._token_count(text)

            latency_ms = int((time.time() - start) * 1000)

            status = GenerationStatus.SUCCESS
            if text is None or str(text).strip() == "":
                status = GenerationStatus.EMPTY
            elif looks_like_refusal(text):
                status = GenerationStatus.REFUSAL

            return GenerationResult(
                raw_response=text,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                provider_details={
                    "model": self.provider_model_name,
                    "cache_enabled": self.use_cache,
                    "batch_size": self.batch_size,
                    "local_cache_client_mode": self.cache_client_mode
                    if self.client is not None
                    else None,
                    "cache_client_interface": self.cache_client_interface
                    if self.client is not None
                    else None,
                    "model_load_params": {
                        key: str(value) for key, value in self.model_load_params.items()
                    },
                    "cache_metadata": {
                        "cached": getattr(cache_metadata, "cached", None),
                        "duplicated": getattr(cache_metadata, "duplicated", None),
                    }
                    if self.client is not None
                    else None,
                },
            )
        except Exception as exc:
            return GenerationResult(
                raw_response=None,
                status=GenerationStatus.ERROR,
                error_message=str(exc),
                latency_ms=int((time.time() - start) * 1000),
            )

    def generate_batch(
        self,
        prompts: list[str],
        *,
        temperature: float | None = None,
        max_output_tokens: int = 256,
        seed: int | None = None,
        reasoning_mode: str | None = None,
        search_enabled: bool = False,
        response_format: str = "json",
        model_params: dict[str, Any] | None = None,
    ) -> list[GenerationResult]:
        if not prompts:
            return []

        start = time.time()
        try:
            generation_model_params = self._generation_model_params(model_params)

            if seed is not None:
                from transformers import set_seed

                set_seed(seed)

            if self.client is not None:
                if self.cache_client_interface != "generate":
                    return [
                        self.generate(
                            prompt,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                            seed=seed,
                            reasoning_mode=reasoning_mode,
                            search_enabled=search_enabled,
                            response_format=response_format,
                            model_params=model_params,
                        )
                        for prompt in prompts
                    ]

                texts, cache_metadata, input_tokens = self._call_cached_generate_client_batch(
                    prompts,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    model_params=generation_model_params,
                )
                provider_details = self._provider_details(
                    cache_metadata=cache_metadata,
                    batch_count=len(prompts),
                )
            else:
                texts = self._call_pipeline_batch(
                    prompts,
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                    model_params=generation_model_params,
                )
                input_tokens = [self._token_count(prompt) for prompt in prompts]
                provider_details = self._provider_details(
                    batch_count=len(prompts),
                )

            return [
                self._result_from_text(
                    text,
                    start=start,
                    input_tokens=input_count,
                    provider_details=provider_details,
                )
                for text, input_count in zip(texts, input_tokens)
            ]
        except Exception as exc:
            provider_details = self._provider_details(
                batch_count=len(prompts),
            )
            return [
                self._result_from_text(
                    None,
                    start=start,
                    input_tokens=None,
                    provider_details=provider_details,
                    error_message=str(exc),
                )
                for _ in prompts
            ]
