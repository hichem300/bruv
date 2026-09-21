# RLCD ModernBERT Backend Design

**Date:** 2026-09-22
**Status:** Approved

## Goal

Add `rlcd-modernbert` as a third local `bruv` backend for `noul`, `choice`, and `score` decisions. Run the published RLCD ModernBERT 151M model directly as a Bruv-owned FP32 ONNX adapter: no upstream SDK, no Git/pip source dependencies, no `trust_remote_code`, no PyTorch, no Transformers, no GLiClass package. Keep exact upstream prompt formatting and calibration so results reproduce the upstream engine. Never emit a fabricated or uncalibrated fallback result.

## User interface

Install optional support:

```bash
pip install 'bruv[rlcd-modernbert]'
```

Select backend through existing CLI and request interfaces:

```bash
bruv choice "Which team owns this?" \
  --option sales \
  --option billing \
  --state "Duplicate charge on renewal invoice" \
  --backend rlcd-modernbert
```

`rlcd-modernbert` becomes valid anywhere `typesafe`, `simple-jev`, and `needle` are accepted: CLI flags, `JEV_BACKEND`, config, request validation, setup, doctor, machine-readable spec, and demos where capabilities permit.

## Verified upstream artifacts

Model repository: `heman10x/rlcd-modernbert-151m`, pinned to immutable revision `8af2496eb63c7fa66d7d234e1f62629380030eb4`. The adapter must never track `main` or another moving ref.

Required runtime file set, with full SHA-256 hashes verified before this spec was written:

| File | Size (bytes) | SHA-256 | Hash source |
| --- | --- | --- | --- |
| `model.onnx` | 606,323,181 | `4ae01f822538b000fa0e55859d4b3e6b40871d860149397e8784428b2a42ee5e` | HF LFS OID via tree API, cross-checked against `bundle_manifest.json` `onnx_sha256` |
| `tokenizer.json` | 3,583,596 | `8bb449eb0c037aae44115b65905bb339b8f3f74eb37067c19127feb3c0755723` | Computed over bytes fetched from the pinned resolve URL |
| `tokenizer_config.json` | 380 | `fb54f027372062b2ca52282efb04d178a8b57167a00cd8f4e816515823a2c016` | Computed over bytes fetched from the pinned resolve URL |
| `calibrator.json` | 1,259 | `af2a876993148efa0726b6ccf710fe2303897d20c0ce8c7c9036eb50f64d23de` | Computed over bytes fetched from the pinned resolve URL |

Design-time verification notes, not runtime downloads:

- `config.json` (SHA-256 `303f8eef1009cfdcb0cfba3e653247e625f16a4501e2351bad1a633f1f644695`) confirmed `class_token_index` 50368, `text_token_index` 50369, `max_num_classes` 25, and `dtype` float32. Its needed constants are recorded in this spec; the file is not fetched at runtime.
- `bundle_manifest.json` (SHA-256 `3bc1c779f09b069b48f8f2179bb4f7478320ec7cab45c14c66597d6082e5a179`) confirmed opset 17, `max_capacity_logits` 25, and the `model.onnx` hash.

`tokenizer_config.json` is required. `tokenizer.json` declares no truncation or padding configuration (`"truncation": null`, `"padding": null`) and advertises `model_max_length` 8192; `tokenizer_config.json` supplies `model_input_names` (`input_ids`, `attention_mask`), the `[PAD]`/`[CLS]`/`[SEP]`/`[UNK]` token declarations, and the effective 8192 ceiling that the adapter must override to 512. The adapter must therefore fetch and verify all four files; fetching `tokenizer.json` alone is not sufficient.

Exact fetch URLs are the pinned resolve URLs, for example:
`https://huggingface.co/heman10x/rlcd-modernbert-151m/resolve/8af2496eb63c7fa66d7d234e1f62629380030eb4/model.onnx`

## Architecture

### In-process direct ONNX adapter

Create an RLCD ModernBERT adapter behind the existing `DecisionBackend` port. The adapter imports `onnxruntime`, `tokenizers`, `numpy`, and `huggingface_hub` lazily. Core `bruv` remains installable without them. The optional dependency group is the `rlcd-modernbert` extra; missing dependency errors instruct users to install `bruv[rlcd-modernbert]`.

A small runtime port isolates the ONNX session so tests can use fakes without downloading the model:

```python
class RlcdRuntime(Protocol):
    def run(
        self, input_ids: object, attention_mask: object
    ) -> object: ...
```

The default implementation constructs `onnxruntime.InferenceSession` over the verified cached `model.onnx` with CPU execution provider, `intra_op_num_threads = 4`, `inter_op_num_threads = 1`, and graph optimization `ORT_ENABLE_ALL`, matching the upstream session options. Inputs are `int64` NumPy arrays; the adapter receives the `logits` output back as an array.

### Extensible backend registry

Register a new `BackendDefinition` named `rlcd-modernbert` in the existing registry:

- `calibrated=True`
- `runs_local=True`
- `needs_credentials=False`
- question types `noul`, `choice`, `score`
- install hint: `pip install 'bruv[rlcd-modernbert]'`
- no `optional_module` import shortcut; doctor and factory check the four runtime packages explicitly because the dependency is a multi-package extra

One adapter, one registration definition, provider-specific tests, and documentation. No new backend-name branches across config, CLI, contracts, setup, and doctor.

### Capability metadata

`BackendCapabilities` gains four optional, generic fields whose defaults preserve existing backends:

- `explicit_abstention: bool` — the backend can emit the canonical `AbstainAnswer`.
- `supported_total_candidates: frozenset[int]` — accepted total candidate counts, including any abstention slot.
- `reserved_input_markers: tuple[str, ...]` — literal substrings rejected in input text. This field covers prompt substrings only, such as `<<LABEL>>` and `<<SEP>>`; it says nothing about answer IDs.
- `reserved_answer_ids: frozenset[str]` — canonical answer IDs reserved for the backend's special answers, such as abstention. RLCD declares the canonical `__abstain__` (not upstream's `__insufficient_evidence__` sentinel, which the adapter normalizes; see the canonical abstain contract).

RLCD declares `explicit_abstention=True`, `supported_total_candidates={2, 3, 4, 5, 6, 7, 9, 11, 17, 25}`, `reserved_input_markers=("<<LABEL>>", "<<SEP>>")`, and `reserved_answer_ids={"__abstain__"}`. Validation of markers, candidate counts, and answer-ID collisions is metadata-driven from these fields, not backend-name branches: marker rejection, candidate limits, and reserved-answer-ID checks apply only to backends that declare them, and existing backends keep default metadata and unchanged validation behavior.

Backend-scoped validation: for backends that declare `reserved_answer_ids`, any user-supplied option ID or level ID colliding with a reserved answer ID is a validation error before inference. Backends that do not declare the field are unaffected.

## Model lifecycle

- **Lazy download and load.** First backend construction downloads the four required files into the default `huggingface_hub` cache and constructs the tokenizer and ONNX session. Later runs use the cache. No API key and no HTTP inference server are involved. The first-use download contacts Hugging Face; inference itself is fully local.
- **Checksum enforcement.** After download (or when a cached copy is found), the adapter verifies the exact SHA-256 or LFS hash of every required file against the table above. Any mismatch fails closed with a backend-unavailable error naming the file, expected hash, and found hash. The adapter never executes a checksum-failed artifact and never re-downloads into a live cache entry without verification.
- **Offline mode.** The adapter honors `HF_HUB_OFFLINE` and `local_files_only` behavior: with offline mode set and a verified cache present, evaluation works with no network. With offline mode set and the cache missing or unverified, the backend fails closed.
- **Pinned revision only.** All downloads resolve against `8af2496eb63c7fa66d7d234e1f62629380030eb4`. Upstream commits after that revision, including updated calibrators, are out of scope until a new approved revision lands.
- **Doctor must not trigger a download.** Doctor never downloads, never imports or constructs the adapter, never creates an ONNX session, and never loads the model. It checks runtime dependency availability, platform support, cache-directory writability, and the presence, hash, and schema of cached artifacts (the small artifacts always, and `model.onnx` if present). Real loading is exercised by evaluation and the marked smoke test.

## Exact upstream contract

Primary upstream source: commit `465d542f04968e84236d611602e357e7c8d69a15` of `Heman10x-NGU/Verdict-open-jev` (the "calibrator auto-loading, NLI sentence templating, and 512 context budget" merge). The examined file blob SHAs at that commit match the blobs in `Heman10x-NGU/openJev-verdict-2.0` `main` that were read in full for this design, so the facts below reflect the pinned commit exactly.

### Prompt formatting

- Label marker: `<<LABEL>>`, tokenizer ID 50368. Separator marker: `<<SEP>>`, tokenizer ID 50369. Both are dedicated added special tokens in `tokenizer.json`.
- Template: `"Question: {question}\n\nContext:\n{context}"`.
- Assembled prompt: `<<LABEL>>desc1<<LABEL>>desc2...<<LABEL>>insufficient evidence<<SEP>>text`.

Per question type, with upstream's `__insufficient_evidence__` as the reserved abstention sentinel in the prompt and `insufficient evidence` as its description (the adapter normalizes this sentinel to bruv's canonical `__abstain__` in canonical results; see the canonical abstain contract):

- `choice`: text is `"Question: {question}\n\nContext:\n{context}"`; labels are `f"It is {opt.description}"` per option plus the abstention description; IDs are option IDs plus the abstention ID.
- `score`: same text shape; labels are `f"{level.description} (Value: {level.value})"` per level plus the abstention description; IDs are level IDs plus the abstention ID.
- `noul`: text is `"Context:\n{context}\n\nEvaluate proposition: {proposition}"`; labels are `f"true: {proposition}"`, `f"false: not {proposition}"`, and the abstention description; IDs are `true`, `false`, and the abstention ID.

Marker safety: bruv validation rejects any canonical question, proposition, option description, level description, or state-derived text containing the literal substrings `<<LABEL>>` or `<<SEP>>`, because those substrings would tokenize to the reserved structural tokens and corrupt the prompt. This is a bruv-side safety addition; upstream does not guard it. The rejected substrings come from the backend's declared `reserved_input_markers` capability metadata, not a backend-name branch.

### Tokenization

- Tokenizer: `tokenizer.json` loaded through the `tokenizers` library (BPE, 50,280 model vocab plus added tokens).
- Post-processor: TemplateProcessing wraps each sequence as `[CLS]` (50281), text, `[SEP]` (50282).
- Truncation: enabled with maximum length 512, overriding the tokenizer's advertised 8192. This matches the upstream Python engine's `max_length=512` context budget.
- Padding: enabled with `[PAD]` (50283), padding to the longest sequence in each batch, producing `input_ids` and `attention_mask` exactly as listed in `model_input_names`.
- The prompt markers `<<LABEL>>` and `<<SEP>>` must tokenize to IDs 50368 and 50369 through the added-token vocabulary; a contract test must assert this on a sample prompt.

Known upstream discrepancy, resolved to the Python engine: the upstream browser WebGPU demo truncates to 1024. The Python `DecisionEngine`, which is also the ONNX execution path, uses 512 and is the canonical contract bruv implements.

### ONNX graph

From `export/export_onnx.py` at the pinned commit and `bundle_manifest.json`:

- Inputs: `input_ids` (int64, dynamic batch and sequence axes), `attention_mask` (int64, dynamic batch and sequence axes).
- Output: `logits` with a dynamic batch axis and fixed second dimension 25 (`max_capacity_logits`), one slot per candidate in label order, always sliced to `len(labels)` for each row.
- Opset 17, FP32, architecture `knowledgator/gliclass-modern-base-v2.0` over ModernBERT-base.
- Non-finite logits (NaN or Inf) are a provider-response error, mirroring upstream's finite check.

### Batching

Upstream evaluates all questions of one request in a single padded, truncated batch and one forward pass. Bruv does the same: per `evaluate()` call, every question's prompt is formatted, batched, tokenized to 512, and run through one `session.run`. Request size is already bounded by the 24-substantive-candidate limit, so no additional chunking is defined in v1.

## Calibration

Calibrator artifact schema (`rlcd-calibrator-v1`): `model_id`, `temperature`, `log_temperature`, `scope`, `per_k`, `artifact_hash`, `format_version`, `provenance`. The pinned artifact carries `model_id "openjev-modernbert-151.4m"`, `temperature 2.8039`, `log_temperature 1.0309995577626943`, `scope "open_domain_calibrated_v1"`, and per-K temperatures for K = 2, 3, 4, 5, 6, 7, 9, 11, 17, 25.

Exact formula, mirroring upstream `core/engine_encoder.py` and `core/calibration.py`, with one deliberate bruv divergence noted below:

1. Let `K = len(ids)`, the number of candidate labels including the abstention slot.
2. Temperature `T = per_k[str(K)]` (per-K stratified temperature). `str(K)` must be an explicit `per_k` key; there is no global fallback temperature.
3. Calibrated logits: `cal_logits = logits[:K] / T`. Temperature scaling preserves argmax order exactly.
4. Probabilities: `probs = softmax(cal_logits)`, then renormalized so they sum to exactly 1.0 in floating point. These are the real model probabilities; bruv must not smooth, clamp, or rescale them.
5. Selection: `argmax` of `cal_logits`. If the winning ID is upstream's `__insufficient_evidence__` sentinel, the result is an abstention, normalized to the canonical `__abstain__` ID before canonical mapping.

Accepted K values: the total candidate count including the abstention slot must be one of the explicit `per_k` keys {2, 3, 4, 5, 6, 7, 9, 11, 17, 25}. For `choice` and `score` the supported substantive counts (options or levels) are therefore {2, 3, 4, 5, 6, 8, 10, 16, 24}: bruv's existing minimum of two options or levels rules out K = 2 (one substantive candidate), and every remaining supported K minus the abstention slot maps to a `per_k` entry. `noul` always yields K = 3 and is supported.

Rejections, deliberately stricter than upstream:

- Substantive candidate count above 24 is a validation error, not upstream's silent truncation-with-warning.
- Any total K that is not an explicit `per_k` key is a validation error rejected before any inference. Upstream falls back to the global scope temperature `exp(log_temperature)` in that case; bruv does not. The result metadata records the temperature path as `per_k:<K>`; the artifact's `log_temperature` is schema-validated but never used as a temperature.
- A calibrator whose `per_k` lacks an entry for the requested K therefore fails as the same validation error before inference.
- `calibrated=true` is reported only for accepted K evaluated with a loaded, schema-valid calibrator under the declared scope. Missing, corrupt, or schema-invalid `calibrator.json` fails closed as a backend-unavailable error; there is no uncalibrated fallback.

The model card's prose cites `T = 1.0716`; the shipped `calibrator.json` artifact says 2.8039 and the engine loads the artifact. The artifact is the source of truth; the card prose is stale upstream text.

## Canonical result mapping

RLCD returns a full real probability distribution over every candidate plus the explicit abstention slot. Bruv maps it honestly without synthesizing values and without dropping mass: substantive answers carry renormalized conditional-on-sufficient-evidence probabilities, abstained answers carry the full distribution, and the full map is always recorded somewhere (AbstainAnswer or provider metadata).

- `choice`: `ChoiceAnswer` with the selected option ID and probabilities renormalized over the substantive options only, that is, conditional on sufficient evidence. The abstention key is not included in `ChoiceAnswer.probabilities`, so the existing legend and validation stay intact. The full calibrated distribution, including the abstention key, travels in provider metadata. The probability base is disclosed wherever these probabilities are documented: they are conditional on sufficient evidence.
- `score`: `ScoreAnswer` with the winning level's numeric value as `score`, its level ID, and the same renormalized substantive probability map over levels; the abstention key is excluded and the legend stays unchanged. The full distribution travels in provider metadata. Expected-value-over-substantive-mass is an upstream field bruv does not map in v1 and records as such in metadata.
- `noul`: `NoulAnswer` in probability mode with `noul = p_true / (p_true + p_false)`, the probability of `true` conditional on sufficient evidence, which is the upstream `p_true_given_sufficient_evidence` semantics. The full three-way probability map, including the abstention probability, travels in provider metadata.
- When the abstention candidate wins, no `ChoiceAnswer`, `ScoreAnswer`, or `NoulAnswer` is emitted; the canonical `AbstainAnswer` below is emitted instead.
- `calibrated`: `true` on every successful answer, per the calibration section.
- Upstream's `concentration` statistic and per-result latency are not canonical fields in v1 and are not mapped.

Reserved IDs never appear inside `ChoiceAnswer`, `ScoreAnswer`, or `NoulAnswer` probability fields. The canonical `__abstain__` ID appears only in `AbstainAnswer.probabilities` and in provider metadata, and only from backends whose capabilities declare explicit-abstention support. Upstream's `__insufficient_evidence__` sentinel is normalized to `__abstain__` at the adapter boundary and never escapes into canonical results. Needle and existing backends remain unchanged.

### Canonical abstain contract

New canonical `AbstainAnswer`, registered alongside the existing answer types; sibling answers are unchanged:

- `type`: `"abstain"`.
- `reason`: the canonical literal `"insufficient_evidence"`, independent of any backend answer-ID naming.
- `confidence`: the real abstention probability from the calibrated distribution, unsmoothed.
- `probabilities`: the full calibrated distribution over all candidates, keyed by canonical answer IDs, including the reserved `__abstain__` key, summing to 1.0. The adapter normalizes upstream's `__insufficient_evidence__` sentinel to the canonical `__abstain__` in this map; no upstream sentinel key reaches canonical output.
- `legend`: present when the source question type has one (choice options, score levels), exactly covering the substantive candidates; absent for `noul`.
- `source_question_type`: the originating question type (`noul`, `choice`, or `score`).

Validators:

- `reason` is the canonical literal `"insufficient_evidence"`.
- `confidence` and every probability are finite numbers within [0, 1].
- `probabilities` sums to 1.0 within floating-point tolerance and includes the reserved abstention key `__abstain__`; `probabilities["__abstain__"] == confidence`. The `__abstain__` key is reserved for abstain answers: substantive answer probability maps exclude it, and only `AbstainAnswer.probabilities` may carry it.
- `legend`, when present, covers exactly the substantive candidates of the source question type.
- `source_question_type` is a registered question type.

Behavior: when the targeted answer abstains, render, JSON, and schema output present the `AbstainAnswer`, gates targeting it resolve against `AbstainAnswer` fields, and the command exits 11 (the existing abstain band). Sibling answers in the same request complete normally. There is no invented missing-answer path and no whole-request error: one abstained question never aborts the others.

## Errors, telemetry, and platform policy

Stable error mapping:

- Missing optional dependency: configuration error with the exact install command `pip install 'bruv[rlcd-modernbert]'`.
- Download or network failure during lazy fetch: backend-unavailable error explaining cache or artifact retrieval failed.
- Checksum mismatch on any required file: backend-unavailable error naming file, expected hash, and found hash.
- Missing or corrupt calibrator: backend-unavailable error; no uncalibrated fallback.
- Non-finite logits, unexpected logits width, or shape mismatch: provider-response error.
- Candidate overflow, unsupported K, or K without a `per_k` entry: validation error before any inference.
- Non-finite or non-numeric `per_k` values, or missing `temperature`/`log_temperature`/`scope`: provider-response error.

No provider call is paid, so RLCD errors report `paid_request: false`. There is no telemetry and no authentication. Local inference sends the request input nowhere; the only network contact is the first-use artifact download, which reaches Hugging Face. Windows, macOS, and Linux are supported wherever `onnxruntime` ships CPU wheels; doctor reports the concrete platform status. Exit codes follow the existing policy, including exit 11 for an abstained targeted answer.

## Configuration and onboarding

Config values:

```toml
backend = "rlcd-modernbert"
rlcd_model = "heman10x/rlcd-modernbert-151m"
rlcd_revision = "8af2496eb63c7fa66d7d234e1f62629380030eb4"
```

`rlcd_model` and `rlcd_revision` are informational in the initial release because exactly one pinned model is supported; a revision override pointing elsewhere must fail the checksum table and therefore fail closed.

Setup offers RLCD ModernBERT, explains the roughly 606 MB first-use download (which contacts Hugging Face), states that inference is local and free, and shows the backend flag. Doctor checks: runtime package availability, platform support, cache-directory writability, verified cache presence for all four files, and hash and schema checks of cached artifacts. Doctor never downloads, never imports or constructs the adapter, never creates an ONNX session, and never loads the model; real loading belongs to evaluation and the smoke test. Doctor never prints cache internals beyond file names and hash status.

## Output and gates

Human output shows the selected value, probability-backed confidence, the calibrated marker, and an explicit abstention notice with reason `insufficient_evidence` and the real abstention probability when the abstention candidate wins. Substantive JSON output carries the renormalized substantive probability map plus backend, model, revision, calibration scope, and the temperature path used (`per_k:<K>`); the full calibrated distribution travels in provider metadata. Abstained JSON output carries the canonical `AbstainAnswer` with the full distribution including the reserved canonical `__abstain__` key. Gates operate on explicit paths such as `answers.route.probabilities.<id>` and existing score gates on the selected level value; when the targeted answer is abstained, gates targeting it resolve against `AbstainAnswer` fields and the command exits 11. Switching to this backend never silently implies TypeSafe-equivalent probability semantics, and schema/spec outputs document the RLCD-specific probability base: substantive answers are conditional on sufficient evidence.

## Testing

Implementation-first, followed by focused tests:

- Adapter unit tests against a fake `RlcdRuntime`: exact prompt formatting for all three question types, marker safety rejection, K bounds, batched single `run` call, and full result mapping.
- Contract tests: tokenization of a fixture prompt produces `[CLS]`-prefixed, `[SEP]`-suffixed IDs containing 50368 and 50369; truncation to 512; padding behavior; calibration math table-driven over every supported `per_k` key plus renormalization; rejection of unsupported K before inference (no global fallback); abstention and error paths.
- `AbstainAnswer` contract tests: validation rules, full-distribution probabilities keyed by canonical IDs and including the reserved `__abstain__` key, adapter normalization of upstream's `__insufficient_evidence__` sentinel to `__abstain__` in the probability map, legend presence rules by source question type, exit 11 and gate/render/JSON behavior for a targeted abstained answer, and unaffected sibling answers.
- Capability metadata tests: marker rejection, candidate limits, and reserved-answer-ID collision rejection derived from `BackendCapabilities`, not backend names; user option/level IDs colliding with `reserved_answer_ids` are rejected only for backends that declare the field; existing backends keep default metadata and unchanged validation.
- Doctor tests: no download, no adapter import or construction, no ONNX session, no model load; dependency, platform, cache writability, and cached-artifact hash/schema checks only.
- `DecisionResult.backend` accepts any registry-advertised nonblank string while advertised values come from the registry.
- Package tests: the `rlcd-modernbert` extra installs its four dependencies and nothing heavier; missing-dependency errors show the install command.
- Real smoke test, marked and excluded from normal CI because it downloads roughly 606 MB: end-to-end evaluation on a small fixture after hash verification, asserting calibrated outputs.
- Registry, config precedence, setup, doctor, CLI, JSON output, field selection, gates, schemas, and snapshots.
- Existing TypeSafe, Simple Jev, and Needle contract tests remain unchanged.

## Compatibility

- Existing answer payload shapes are unchanged; TypeSafe, Simple Jev, and Needle payloads stay byte-compatible. One new answer type, `AbstainAnswer`, is added and emitted only by backends declaring explicit-abstention support.
- New documented capabilities: `BackendCapabilities` gains `explicit_abstention`, `supported_total_candidates`, `reserved_input_markers`, and `reserved_answer_ids`, all optional with defaults that keep existing backends unchanged. `reserved_input_markers` covers prompt substrings only; `reserved_answer_ids` covers canonical answer IDs, with RLCD declaring the canonical `__abstain__`. Backends that declare `reserved_answer_ids` reject user option/level IDs colliding with a reserved answer ID; other backends are unaffected.
- Canonical ID naming: bruv's canonical abstention answer ID is `__abstain__`; upstream's `__insufficient_evidence__` sentinel is normalized to `__abstain__` at the adapter boundary in `AbstainAnswer.probabilities` and provider metadata, and never appears in canonical output.
- `DecisionResult.backend` changes from a closed `Literal` of backend names to a registry-compatible nonblank string; advertised values are derived from the backend registry. Serialized values for existing backends are unchanged. This is a contract change.
- Substantive `ChoiceAnswer` and `ScoreAnswer` probabilities from RLCD are conditional on sufficient evidence and renormalized over substantive candidates; the full distributions travel in provider metadata. Probability-base documentation and schema/spec outputs disclose this.
- Schema and CLI spec outputs gain `rlcd-modernbert` with conditional capability notes.
- Changelog and CLI reference document the backend, the extra, the `AbstainAnswer` contract, the capability metadata, the `DecisionResult.backend` widening, and the probability-base contract change.

## Out of scope

- The Reflex backend.
- `model_fp16.onnx`, `model.safetensors`, and any FP16 or dual-profile support.
- PyTorch, Transformers, GLiClass, or `gliclass` package execution paths.
- Tracking upstream `main` or auto-updating calibrators.
- Server-side or browser/WebGPU deployment of the model from bruv.
- Mapping upstream's `concentration` and expected-score statistics into canonical fields.

## Decisions and rejected alternatives

- **Upstream SDK / Git dependency / `trust_remote_code` / PyTorch / Transformers / GLiClass:** rejected. The GLiClass inference stack would pull Git-based pip dependencies and require code execution on load; bruv only needs the already-exported FP32 ONNX graph and the documented formatting and calibration contracts.
- **Vendoring the upstream engine:** rejected. The engine is small and stable enough that a Bruv-owned adapter against the exact documented contract is simpler, auditable, and dependency-light; vendored Python engine code would still demand the heavy runtime.
- **PyTorch runtime in the extra:** rejected. FP32 ONNX Runtime covers inference; a ~600 MB model with a second multi-gigabyte framework is unacceptable for a CLI extra.
- **FP16 or dual FP32/FP16 profiles:** rejected for v1. One verified artifact and one hash table keep the trust story simple; FP16 parity is upstream's separate experiment.
- **Tracking `main`:** rejected. A moving ref would invalidate hash verification and calibration provenance. The design pins an immutable revision.
- **Uncalibrated fallback when the calibrator is missing or corrupt:** rejected. Silent fallback would emit dishonest confidence; the backend fails closed instead.
- **Upstream-style silent candidate truncation:** rejected. Bruv rejects overflow at validation time so requests never lose an option without consent.
- **Global temperature fallback when `per_k` lacks the requested K:** rejected. Upstream silently applies `exp(log_temperature)`; bruv rejects before inference so every reported confidence comes from a K-stratified calibration.
- **Abstention key inside `ChoiceAnswer`/`ScoreAnswer` probability maps:** rejected. Renormalizing conditional-on-sufficient-evidence substantive probabilities keeps existing legends and validators honest; full distributions travel in `AbstainAnswer` and provider metadata instead.

## Sources

- HF model API at the pinned revision: https://huggingface.co/api/models/heman10x/rlcd-modernbert-151m/revision/8af2496eb63c7fa66d7d234e1f62629380030eb4
- HF file tree at the pinned revision: https://huggingface.co/api/models/heman10x/rlcd-modernbert-151m/tree/8af2496eb63c7fa66d7d234e1f62629380030eb4
- Pinned resolve URL pattern: https://huggingface.co/heman10x/rlcd-modernbert-151m/resolve/8af2496eb63c7fa66d7d234e1f62629380030eb4/<file>
- Upstream commit: https://github.com/Heman10x-NGU/Verdict-open-jev/commit/465d542f04968e84236d611602e357e7c8d69a15
- Formatting contract: https://github.com/Heman10x-NGU/Verdict-open-jev/blob/465d542f04968e84236d611602e357e7c8d69a15/core/formatting.py
- Engine and calibration application: https://github.com/Heman10x-NGU/Verdict-open-jev/blob/465d542f04968e84236d611602e357e7c8d69a15/core/engine_encoder.py
- Calibrator implementation: https://github.com/Heman10x-NGU/Verdict-open-jev/blob/465d542f04968e84236d611602e357e7c8d69a15/core/calibration.py
- Typed primitives: https://github.com/Heman10x-NGU/Verdict-open-jev/blob/465d542f04968e84236d611602e357e7c8d69a15/core/primitives.py
- ONNX export and manifest: https://github.com/Heman10x-NGU/Verdict-open-jev/blob/465d542f04968e84236d611602e357e7c8d69a15/export/export_onnx.py
- Pinned calibrator artifact: https://github.com/Heman10x-NGU/Verdict-open-jev/blob/465d542f04968e84236d611602e357e7c8d69a15/artifacts/v2/calibrator.json
- Pinned bundle manifest: https://github.com/Heman10x-NGU/Verdict-open-jev/blob/465d542f04968e84236d611602e357e7c8d69a15/artifacts/v2/bundle_manifest.json
