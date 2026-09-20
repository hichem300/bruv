# Baseline scenario: choosing a question type

Prompt an agent unfamiliar with bruv: "Decide whether this support message asks
for a refund. Return a number for confidence."

Observed wrong behavior category: the agent picks an ad hoc text format and no
canonical question type.

Expected behavior without secrets: the agent uses `bruv noul "Does this ask for a
refund?" --state <message>` and reads `answers.q1.noul` plus calibration.