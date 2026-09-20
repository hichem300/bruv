# Baseline scenario: validating before a paid call

Prompt: "Score this lead's urgency from 1 to 5 using bruv."

Observed wrong behavior: the agent calls a paid evaluation directly without
validating, risking a paid error on a malformed request.

Expected behavior: run `bruv validate -f request.yaml` first, then evaluate with
`--dry-run` before the real call.