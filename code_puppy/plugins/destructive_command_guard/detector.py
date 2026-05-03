"""Pattern detection for destructive shell commands.

Detects dangerous patterns in shell commands using pure regex — no LLM
calls, no caching, no yolo-mode checks. Covers rm -rf root/home,
git push --mirror, git clean -fd, git reset --hard, git checkout/restore .,
SQL DROP via clients, docker prune, and accidental package publishes.
"""

import re
from dataclasses import dataclass


@dataclass
class DestructiveCommandMatch:
    """Result of a destructive command pattern match."""

    pattern_name: str
    description: str


# ---------------------------------------------------------------------------
# Shell-operator regex — same approach as force_push_guard
# ---------------------------------------------------------------------------

# Matches shell operators that precede a new command in a pipeline/chain.
# E.g. "cd foo && rm -rf /" or "true || git reset --hard"
# The capture ensures the command keyword follows a real shell boundary.
_SHELL_OPERATOR_RE = re.compile(r"(?:^|&&|\|\||;|\|)\s*\w+", re.MULTILINE)


def _is_real_command(command: str) -> bool:
    """Check that the destructive keyword is an actual invocation, not a string arg.

    Handles compound commands like "cd foo && rm -rf /" while
    avoiding false positives like "echo 'rm -rf /'".

    Args:
        command: The shell command string to inspect.

    Returns:
        True if the command appears to be a real invocation.
    """
    return bool(_SHELL_OPERATOR_RE.search(command))


# ---------------------------------------------------------------------------
# Cheap pre-filter substrings — if none appear, bail immediately
# ---------------------------------------------------------------------------

_PREFILTER_SUBSTRINGS = (
    "rm",
    "git",
    "docker",
    "drop",
    "npm",
    "yarn",
    "twine",
    "psql",
    "mysql",
    "sqlite3",
)


# ---------------------------------------------------------------------------
# Pattern list — ordered by specificity, first match wins
# ---------------------------------------------------------------------------

# Tier 1 — irreversible, common AI mistakes
# Tier 2 — destructive but less common

_DESTRUCTIVE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    # ── Tier 1 ──────────────────────────────────────────────────────────
    # 1. rm -rf /  /  rm -rf /*  (recursive delete of root filesystem)
    (
        re.compile(r"\brm\b.*\s-rf?\b.*\s/\s*$"),
        "rm -rf /",
        "recursive delete of root filesystem",
    ),
    (
        re.compile(r"\brm\b.*\s-rf?\b.*\s/\*\s*$"),
        "rm -rf /*",
        "recursive delete of root filesystem (glob)",
    ),
    # 2. rm -rf ~  /  rm -rf ~/*  (recursive delete of home directory)
    (
        re.compile(r"\brm\b.*\s-rf?\b.*\s~\s*$"),
        "rm -rf ~",
        "recursive delete of home directory",
    ),
    (
        re.compile(r"\brm\b.*\s-rf?\b.*\s~/\*\s*$"),
        "rm -rf ~/*",
        "recursive delete of home directory (glob)",
    ),
    # 3. git push --mirror  (deletes remote branches not present locally)
    (
        re.compile(r"\bgit\s+push\b.*--mirror\b"),
        "git push --mirror",
        "deletes remote branches not present locally",
    ),
    # 4. git clean -fd  (deletes untracked files and directories)
    (
        re.compile(r"\bgit\s+clean\b.*-f\b.*[dx]"),
        "git clean -fd",
        "deletes untracked files and directories",
    ),
    # 5. git reset --hard  (destroys all uncommitted changes)
    (
        re.compile(r"\bgit\s+reset\b.*--hard\b"),
        "git reset --hard",
        "destroys all uncommitted changes",
    ),
    # 6. git checkout -- .  /  git restore .  (discards all working dir changes)
    (
        re.compile(r"\bgit\s+(?:checkout|restore)\b.*\s--?\s*\.\s*$"),
        "git checkout/restore .",
        "discards all working directory changes",
    ),
    # ── Tier 2 ──────────────────────────────────────────────────────────
    # 7. DROP TABLE/DATABASE/SCHEMA via SQL client
    (
        re.compile(
            r"(?:psql|mysql|sqlite3)\b.*(?:-c|-e)\b.*DROP\s+(?:TABLE|DATABASE|SCHEMA)\b",
            re.IGNORECASE,
        ),
        "DROP via SQL client",
        "drops a table/database/schema via SQL client",
    ),
    (
        re.compile(
            r"DROP\s+(?:TABLE|DATABASE|SCHEMA)\b.*\|\s*(?:psql|mysql|sqlite3)\b",
            re.IGNORECASE,
        ),
        "DROP via SQL pipe",
        "drops a table/database/schema piped to SQL client",
    ),
    # 8. docker system prune -af  /  docker volume prune -f
    (
        re.compile(r"\bdocker\s+(?:system|volume)\s+prune\b.*\s-?[af]\b"),
        "docker prune",
        "nukes Docker resources without confirmation",
    ),
    # 9. npm publish  /  yarn publish  /  twine upload
    (
        re.compile(r"\b(?:npm|yarn)\s+publish\b"),
        "npm/yarn publish",
        "accidental package publishing",
    ),
    (
        re.compile(r"\btwine\s+upload\b"),
        "twine upload",
        "accidental package publishing",
    ),
]


def detect_destructive_command(command: str) -> DestructiveCommandMatch | None:
    """Check if a shell command contains a destructive operation.

    Uses a cheap substring pre-filter before any regex work, then verifies
    the command is a real invocation (not a string argument), then checks
    patterns first-match-wins.

    Args:
        command: The shell command string to inspect.

    Returns:
        DestructiveCommandMatch if a destructive pattern is found, None otherwise.
    """
    # Quick pre-filter: bail if none of the trigger substrings appear
    command_lower = command.lower()
    if not any(sub in command_lower for sub in _PREFILTER_SUBSTRINGS):
        return None

    # Ensure the command is a real invocation, not a string argument
    if not _is_real_command(command):
        return None

    for pattern, name, description in _DESTRUCTIVE_PATTERNS:
        if pattern.search(command):
            return DestructiveCommandMatch(pattern_name=name, description=description)

    return None
