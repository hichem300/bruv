# Agent usage

bruv ships a packaged skill that teaches agents when and how to use it. You can
install the skill into Codex, Claude Code, or Pi.

## Install for an agent

```bash
# Pi, user scope (available everywhere)
bruv skill install --target pi --scope user

# Claude Code, inside a project
bruv skill install --target claude --scope project --project-root /path/to/repo

# Codex, user scope
bruv skill install --target codex --scope user
```

The installer prints a plan before it copies. If the destination already
exists, add `--force` to replace it. Non-interactive runs never prompt.

## Manual install

Copy `skills/bruv/SKILL.md` into your agent's skill directory:

- Pi user: `~/.pi/agent/skills/bruv/SKILL.md`
- Claude user: `~/.claude/skills/bruv/SKILL.md`
- Codex user: `~/.codex/skills/bruv/SKILL.md`

For project scope, use the matching dot-directory inside the project root.

## Secret safety

Never pass API keys as CLI flags. Use the `TYPESAFE_API_KEY` environment
variable or `bruv setup`.