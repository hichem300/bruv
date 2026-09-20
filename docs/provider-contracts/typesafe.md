# TypeSafe Jev provider contract

Source: https://docs.typesafe.ai/introduction/quickstart
Source: https://docs.typesafe.ai/llms.txt
Source: https://docs.typesafe.ai/sdk/python.md
Source: https://docs.typesafe.ai/sdk/python/api/clients/sync.md
Source: https://docs.typesafe.ai/sdk/python/api/types/questions.md
Source: https://docs.typesafe.ai/sdk/python/api/types/responses.md
Source: https://docs.typesafe.ai/sdk/python/api/exceptions.md
Source: https://docs.typesafe.ai/introduction/machine-learning-primer.md
Source: https://docs.typesafe.ai/sdk/python/changelog.md
Source: https://github.com/typesafe-ai/typesafe-sdk-python/releases/tag/v0.7.0
Accessed: 2026-09-20
Version/commit: `typesafe-sdk` 0.7.0, tag `v0.7.0`, commit `2ce5c65f13646cab6e6f782328194c9d85f3300a`.

This note pins bruv's TypeSafe adapter boundary. URLs above were discovered from TypeSafe's official `llms.txt`; release and commit evidence comes from TypeSafe's official SDK repository.

## SDK and authentication

- Distribution package: `typesafe-sdk`.
- Import package: `typesafe_sdk`.
- Pinned contract version: `0.7.0`, latest release listed in official changelog on access date.
- Official installation examples use unversioned `uv add typesafe-sdk` and `pip install typesafe-sdk`; official install docs do not themselves constrain package version.
- Required credential environment variable: `TYPESAFE_API_KEY`.
- Other documented environment variables: `TYPESAFE_BASE_URL`, `TYPESAFE_DEFAULT_MODEL`, and `TYPESAFE_LOG_LEVEL`.
- Default model: `jev-latest`.

## Invocation

Synchronous constructor used by bruv:

```python
TypeSafeClient(
    *,
    api_key: str | None = None,
    model: str | None = None,
    retry: RetryPolicy | None = None,
    timeout: float | httpx2.Timeout | None = None,
    headers: Mapping[str, str] | None = None,
    transport: httpx2.BaseTransport | None = None,
    http_client: httpx2.Client | None = None,
    base_url: str | None = None,
)
```

Pinned `system_one` call signature from SDK 0.7.0:

```python
client.system_one(
    state: JSONContent,
    questions: Mapping[str, Question],
    *,
    model: str | None = None,
    retry: RetryPolicy | None = None,
    timeout: float | httpx2.Timeout | None = None,
    extra_headers: Mapping[str, str] | None = None,
    extra_body: Mapping[str, JSONValue | None] | None = None,
    response_model: type[ResponseT] | None = None,
) -> SystemOneResponse | ResponseT
```

`state` is text, JSON object, or array. `questions` is a nonempty mapping; mapping keys become answer keys. bruv uses default `SystemOneResponse`, not custom `response_model` or forward-compatibility fields.

## Question constructors

All constructors come from `typesafe_sdk`:

```python
Noul(instructions=JSONContent | None, criteria=NoulCriteria | None)
Choice(instructions=JSONContent | None, criteria=Mapping[str, JSONContent | None])
Score(instructions=JSONContent | None, criteria=Sequence[JSONContent])
```

Wire discriminators are `type: "noul"`, `type: "choice"`, and `type: "score"`. `Noul.criteria` may describe `true` and `false`. `Choice.criteria` maps labels to descriptions or `None`. `Score.criteria` is a nonempty ordered sequence. Official Python SDK docs do not state all bruv domain limits, so adapter code must not infer provider limits absent from cited pages.

## Response contract

Default response type is `SystemOneResponse`:

- `model: str`
- `answers: dict[str, Answer]`
- `usage: Usage`
- `request_id`: property sourced from `x-typesafe-request-id`
- `raw_http_response`: underlying response
- filtered properties `nouls`, `choices`, and `scores`

`Usage` fields:

- `input_tokens: int | None`
- `output_tokens: int | None`

Answer fields:

- Noul: `type`, `noul`. `noul` is probability of yes/true from 0 to 1. No `confidence` field is documented.
- Choice: `type`, `choice`, `confidence`, `probabilities`. Probabilities are keyed by choice label and sum approximately to 1.
- Score: `type`, `score`, `confidence`, `legend`, `probabilities`. `score` is probability-weighted expected score and may be fractional; SDK objects expose integer keys for `legend` and `probabilities` after JSON parsing.

No top-level `calibrated` response attribute is documented. Do not invent one in provider-shaped fixtures.

## Documented failures

- `TypeSafeError`: base SDK failure; constructor also uses it for missing API key or invalid timeout, and `system_one` uses it for empty questions or empty Score criteria.
- `ValueError`: both `transport` and `http_client` supplied.
- `TypeSafeAPIError`: unsuccessful HTTP response, with `status`, `body`, `headers`, `endpoint`, and optional `request_id`.
- Specialized HTTP errors: bad request 400, authentication 401, permission 403, not found 404, validation 422, rate limit 429, and internal server 5xx.
- `TypeSafeAPIConnectionError`: no HTTP response.
- `TypeSafeAPITimeoutError`: configured timeout exceeded.
- `TypeSafeAPIResponseValidationError`: successful response lacks required structure or fails response-model validation; includes `field_path`.
- HTTP API reference additionally documents retry with exponential backoff for 429 and 529; SDK default retries handle these.

## Calibration semantics

Official wording says: “Reinforcement learning for calibrated decisions trains TypeSafe to return decisions and calibrated probabilities instead of generated text.” It further defines calibration across groups: outcomes assigned probability 0.8 should occur about 80% of the time, and says these rates are not a guarantee for any single answer.

Therefore bruv may describe TypeSafe probabilities as provider-documented calibrated probabilities, while preserving this population-level meaning. `confidence` is separate: a 0-to-1 statistic derived from distribution shape for Choice and Score. It is not a documented correctness probability and Noul has no separate confidence field.

## Fixture provenance

`tests/fixtures/typesafe/success.json` is synthetic contract data authored for tests. It contains no captured request, customer data, secret, or invented metadata field. Its JSON shape uses only response fields documented above.
