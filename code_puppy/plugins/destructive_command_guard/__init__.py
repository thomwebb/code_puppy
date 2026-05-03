"""Destructive command guard plugin.

Intercepts potentially-destructive shell commands (rm -rf /, git reset --hard,
docker system prune -af, etc.) and prompts the user for approval before
allowing them through. Always active, pure regex, no LLM calls.
"""
