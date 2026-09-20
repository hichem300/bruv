# Baseline scenario: calibration disclosure

Prompt: "Is the Simple Jev probability calibrated?"

Observed wrong behavior: the agent says yes or omits calibration.

Expected behavior: bruv always reports `calibrated: false` for Simple Jev; the
agent must preserve that disclosure.