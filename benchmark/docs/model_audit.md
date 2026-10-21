# Canonical model audit

`data/canonical/model_mapping.csv` is the single canonical mapping from model
keys to display names. The `info_url` column links each row to its official
model card or provider documentation.
`data/canonical/model_runs.csv` records saved runner IDs, available request
configuration, adapter-serialization observations, and known unknowns.

## Reasoning labels

`default_reasoning` describes the provider-documented default, not the setting
used in a benchmark request. For example, OpenAI documents medium as the
GPT-5.5 default, while the retained GPT-5.5 configurations explicitly send
`reasoning_effort=none`. Therefore, the GPT-5.5 rows use `medium` for the
provider default and `none` for the actual request setting.

`unspecified` means that the available run configuration and adapter evidence
do not establish an explicit effort setting. It does not mean that reasoning
was disabled. `N/A` is reserved for models without a documented effort
setting. The provider-returned model ID is unknown rather than reconstructed
from a requested ID.

Figures and tables never print `unspecified`. They show a recorded effort as
recorded, `Default` when the request set no effort on a model that has a
reasoning mode (the provider default applied), and `N/A` when the model has no
reasoning mode. Display names follow the same rule: `[default]` marks a
configuration whose request set no effort, as in `Claude Opus 4.7 [default]`.
`analysis.model_names.reasoning_category` implements this resolution; the
mapping and run files keep the recorded `unspecified` value.

## Request serialization evidence

The saved configurations and the retained adapters show why a configured
`reasoning_mode` does not always equal a provider request setting:

- `OpenAIAdapter` serializes a truthy `reasoning_mode` as
  `reasoning_effort`. The saved GPT-5.5 configurations use `none` explicitly.
- `OpenRouterAdapter` does not serialize `reasoning_mode`. It passes
  `model_params` through, so the DeepSeek V4 Pro high run is evidenced by
  `model_params.reasoning.effort=high`; the default V4 runs have no explicit
  request effort.
- `GeminiAdapter` and `CohereAdapter` do not serialize `reasoning_mode`.
- `LocalTransformersAdapter` also ignores `reasoning_mode`; its generation
  settings come from `model_params`.
- The retained Anthropic adapter maps a truthy mode to adaptive thinking and
  high effort. The saved Opus thinking configuration records `thinking`, but
  the exact historical adapter revision is not archived. The non-thinking
  Opus request payload is also unavailable, so its actual setting remains
  unspecified.
- A saved DeepSeek V4 Flash configuration uses provider `deepseek`, but that
  adapter is absent from the retained framework package. The original request
  payload is unavailable, so its actual effort remains unspecified.

These distinctions are represented in `model_runs.csv`; do not convert an
unset or ignored mode into `none`.

## Official source notes

The following primary sources support details that required corrections or
explicit default reasoning evidence:

- [GPT-5.5 model documentation](https://developers.openai.com/api/docs/models/gpt-5.5)
  identifies `gpt-5.5`, lists medium as the default reasoning effort, and
  documents web search support.
- [DeepSeek V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
  reports 284B total parameters and 13B activated parameters. It describes
  non-think, think-high, and think-max modes without establishing a default;
  the mapping therefore uses `unspecified` for the default.
- [DeepSeek V4 Pro model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro)
  reports 1.6T total parameters and 49B activated parameters and describes
  reasoning modes without establishing a default.
- [Olmo 3 32B Think model card](https://huggingface.co/allenai/Olmo-3-32B-Think)
  uses the `Olmo 3 32B Think` name and identifies it as a Think model.
- [Claude Opus 4.7 model overview](https://platform.claude.com/docs/en/models/opus-4-7/overview)
  documents adaptive thinking and high default effort.
- [Gemini thinking documentation](https://ai.google.dev/gemini-api/docs/thinking)
  documents default thinking levels, including high for Gemini 3.1 Pro. The
  retained Gemini adapter doesn't send a `reasoning_mode`, so the benchmark
  request setting is `unspecified`.
- [Gemini 2.5 Flash-Lite model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite)
  confirms the model identity and thinking support but does not establish a
  default effort value.
- [Qwen2.5 14B Instruct AWQ model card](https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-AWQ)
  is the quantized model actually named in the canonical key. `AWQ` is retained
  in its display label; the base-model card is not a substitute for this card.

For every other retained identity, use the official source URL in that row's
`info_url` cell. The mapping uses `unspecified` when that source or the saved
request evidence does not establish an effort value.
