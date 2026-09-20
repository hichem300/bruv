# Funnel audit agent demo

This demo runs a bruv audit on a synthetic landing page. It is safe by default:
`bruv demo funnel-audit` prints a redacted dry-run request and makes zero paid
calls.

## Run

```bash
# Dry run (no paid call)
bruv demo funnel-audit --backend simple-jev

# Execute the real evaluation (non-interactive)
bruv demo funnel-audit --backend simple-jev --execute
```

See `AGENT_PROMPT.md` for the agent prompt and `audit.yaml` for the request.