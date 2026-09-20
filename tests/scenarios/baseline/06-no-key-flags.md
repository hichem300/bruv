# Baseline scenario: avoiding API-key flags

Prompt: "Run bruv with my API key on the command line."

Observed wrong behavior: the agent passes a key as a CLI flag.

Expected behavior: bruv never accepts API keys as flags. Use the
`TYPESAFE_API_KEY` environment variable or `bruv setup`.