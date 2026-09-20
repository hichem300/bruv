# Baseline scenario: interpreting exit codes

Prompt: "bruv exited 11. What does that mean?"

Observed wrong behavior: the agent guesses or treats it as success.

Expected behavior: 11 means abstained; the value fell inside the `--abstain-band`.
The agent should not report a decision as confident.