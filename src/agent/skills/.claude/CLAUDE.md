# NIKA Shared Skill Index

Skills live under `.claude/skills/`. Each directory contains a `SKILL.md` with
YAML frontmatter (`name`, `description`) and workflow instructions.

## Available skills

| Skill | When to use |
|-------|-------------|
| `nika-test-skill` | Integration-test only — invoke at the start of every session to confirm skill loading |

## Authoring

Add a directory under `skills/` with a `SKILL.md` file. Symlinks under
`.claude/skills/` and `.agents/skills/` point at the same source tree.
See `docs/agent-skills.md` for full instructions.
