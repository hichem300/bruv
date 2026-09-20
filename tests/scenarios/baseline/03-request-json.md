# Baseline scenario: requesting JSON

Prompt: "Give me the decision in a format I can parse."

Observed wrong behavior: the agent prints human prose and no machine envelope.

Expected behavior: use `--output json` and return one JSON object on stdout with
`backend`, `model`, `calibrated`, and `answers`.