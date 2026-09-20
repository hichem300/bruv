# Baseline scenario: incomplete non-interactive request

Prompt: "Run bruv noul in a script with no state."

Observed wrong behavior: the agent prompts interactively or hangs.

Expected behavior: non-interactive missing required input exits 2 with a clear
error instead of prompting.