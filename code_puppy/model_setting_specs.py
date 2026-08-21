"""Shared definitions and resolution for configurable model settings."""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from typing import Any

MODEL_SETTING_DEFINITIONS: dict[str, dict[str, Any]] = {
    "temperature": {
        "name": "Temperature",
        "description": "Controls randomness (0.0-1.0). Lower = more deterministic, higher = more creative.",
        "type": "numeric",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "default": None,
        "format": "{:.2f}",
    },
    "seed": {
        "name": "Seed",
        "description": "Random seed for reproducible outputs. Set to the same value for consistent results.",
        "type": "numeric",
        "min": 0,
        "max": 999999,
        "step": 1,
        "default": None,
        "format": "{:.0f}",
    },
    "top_p": {
        "name": "Top-P (Nucleus Sampling)",
        "description": "Controls token diversity from 0.0 (least random) to 1.0 (most random).",
        "type": "numeric",
        "min": 0.0,
        "max": 1.0,
        "step": 0.05,
        "default": None,
        "format": "{:.2f}",
    },
    "reasoning_effort": {
        "name": "Reasoning Effort",
        "description": "Controls how much effort GPT-5 models spend on reasoning.",
        "type": "choice",
        "choices": ["none", "low", "medium", "high", "xhigh", "max"],
        "default": "medium",
    },
    "reasoning_context": {
        "name": "Reasoning Context",
        "description": "Controls which prior reasoning GPT-5.6 Responses models retain.",
        "type": "choice",
        "choices": ["all_turns", "current_turn", "auto"],
        "default": "all_turns",
    },
    "reasoning_mode": {
        "name": "Reasoning Mode",
        "description": "Controls the GPT-5.6 reasoning mode.",
        "type": "choice",
        "choices": ["standard", "pro"],
        "default": "standard",
    },
    "summary": {
        "name": "Reasoning Summary",
        "description": "Controls the detail of OpenAI Responses reasoning summaries.",
        "type": "choice",
        "choices": ["auto", "concise", "detailed"],
        "default": "auto",
    },
    "verbosity": {
        "name": "Verbosity",
        "description": "Controls response length.",
        "type": "choice",
        "choices": ["low", "medium", "high"],
        "default": "medium",
    },
    "extended_thinking": {
        "name": "Extended Thinking",
        "description": "Controls classic, adaptive, or disabled extended thinking.",
        "type": "choice",
        "choices": ["enabled", "adaptive", "off"],
        "default": "enabled",
    },
    "budget_tokens": {
        "name": "Thinking Budget (tokens)",
        "description": "Maximum tokens for classic extended thinking.",
        "type": "numeric",
        "min": 1024,
        "max": 131072,
        "step": 1024,
        "default": 10000,
        "format": "{:.0f}",
    },
    "interleaved_thinking": {
        "name": "Interleaved Thinking",
        "description": (
            "Enable thinking between tool calls on supported Claude 4 models. "
            "Adds a beta header; unsupported Vertex/Bedrock models can reject it."
        ),
        "type": "boolean",
        "default": False,
    },
    "clear_thinking": {
        "name": "Clear Thinking",
        "description": "Controls whether visible thinking blocks are removed.",
        "type": "boolean",
        "default": False,
    },
    "thinking_type": {
        "name": "Thinking Type (GLM)",
        "description": "Enables or disables GLM deep thinking.",
        "type": "choice",
        "choices": ["enabled", "disabled"],
        "default": "enabled",
    },
    "glm_reasoning_effort": {
        "name": "Reasoning Effort (GLM-5.2+)",
        "description": "Controls chain-of-thought effort for supported GLM models.",
        "type": "choice",
        "choices": ["max", "xhigh", "high", "medium", "low", "minimal", "none"],
        "default": "max",
    },
    "thinking_enabled": {
        "name": "Thinking Enabled",
        "description": "Enables thinking mode for supported Gemini models.",
        "type": "boolean",
        "default": True,
    },
    "thinking_level": {
        "name": "Thinking Level",
        "description": "Controls the depth of Gemini thinking.",
        "type": "choice",
        "choices": ["low", "high"],
        "default": "low",
    },
    "effort": {
        "name": "Effort",
        "description": "Controls how much effort adaptive models spend on a response.",
        "type": "choice",
        "choices": ["low", "medium", "high", "xhigh", "max"],
        "default": "high",
    },
}

_ScopedSettings = tuple[
    str,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]
_scoped_model_settings: ContextVar[_ScopedSettings | None] = ContextVar(
    "scoped_model_settings", default=None
)


class ModelSettingsValidationError(ValueError):
    """Raised when agent-scoped model settings are malformed."""


class ModelSettingsCapabilityError(ModelSettingsValidationError):
    """Raised when a valid setting value is unavailable on one model."""


def model_identity(model_name: str, model_config: Mapping[str, Any]) -> str:
    """Return normalized alias + underlying-ID text for family detection."""
    return f"{model_name} {model_config.get('name', '')}".lower()


_ANTHROPIC_MESSAGES_TYPES = frozenset(
    {
        "anthropic",
        "aws_bedrock",
        "azure_foundry",
        "claude_code",
        "custom_anthropic",
    }
)


def uses_anthropic_messages_api(
    model_name: str, model_config: Mapping[str, Any]
) -> bool:
    """Return whether the configured adapter uses Anthropic Messages.

    Family and wire protocol are deliberately separate. A Claude model routed
    through OpenRouter, Copilot, or custom OpenAI still needs OpenAI-shaped
    request settings.
    """
    model_type = model_config.get("type")
    if model_type is not None:
        return model_type in _ANTHROPIC_MESSAGES_TYPES
    identity = model_identity(model_name, model_config)
    return "claude-" in identity or "anthropic-" in identity


def gpt_5_minor_version(model_name: str, model_config: Mapping[str, Any]) -> int | None:
    """Return the GPT-5 minor version from an alias or underlying model ID."""
    match = re.search(r"gpt-5(?:\.(\d+))?", model_identity(model_name, model_config))
    if match is None:
        return None
    return int(match.group(1) or 0)


def supports_xhigh_reasoning(model_name: str, model_config: Mapping[str, Any]) -> bool:
    """Return whether a model supports the xhigh reasoning-effort value."""
    minor = gpt_5_minor_version(model_name, model_config)
    return bool(
        model_config.get("supports_xhigh_reasoning", False)
        or "codex" in model_identity(model_name, model_config)
        or (minor is not None and minor >= 4)
    )


def supports_max_reasoning(model_name: str, model_config: Mapping[str, Any]) -> bool:
    """Return whether a model supports the max reasoning-effort value."""
    minor = gpt_5_minor_version(model_name, model_config)
    return bool(
        model_config.get("supports_max_reasoning", False)
        or (minor is not None and minor >= 6)
    )


def validate_model_settings(
    settings: Mapping[str, Any], *, source: str = "model_settings"
) -> None:
    """Validate known model settings and reject unsafe compound values."""
    for name, value in settings.items():
        if not isinstance(name, str) or not name:
            raise ModelSettingsValidationError(
                f"{source} setting names must be non-empty strings"
            )
        if value is None:
            raise ModelSettingsValidationError(
                f"{source}.{name} must not be null; omit it to inherit"
            )

        definition = MODEL_SETTING_DEFINITIONS.get(name)
        if definition is None:
            if not isinstance(value, (str, int, float, bool)):
                raise ModelSettingsValidationError(
                    f"{source}.{name} must be a string, number, or boolean"
                )
            if isinstance(value, float) and not math.isfinite(value):
                raise ModelSettingsValidationError(f"{source}.{name} must be finite")
            continue

        setting_type = definition["type"]
        if setting_type == "boolean":
            if not isinstance(value, bool):
                raise ModelSettingsValidationError(f"{source}.{name} must be a boolean")
            continue
        if setting_type == "choice":
            choices = definition["choices"]
            if not isinstance(value, str) or value not in choices:
                allowed = ", ".join(choices)
                raise ModelSettingsValidationError(
                    f"{source}.{name} must be one of: {allowed}"
                )
            continue
        if setting_type == "numeric":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ModelSettingsValidationError(f"{source}.{name} must be a number")
            if isinstance(value, float) and not math.isfinite(value):
                raise ModelSettingsValidationError(f"{source}.{name} must be finite")
            if value < definition["min"] or value > definition["max"]:
                raise ModelSettingsValidationError(
                    f"{source}.{name} must be between "
                    f"{definition['min']} and {definition['max']}"
                )
            step = definition.get("step")
            if (
                isinstance(step, int)
                and isinstance(value, float)
                and not value.is_integer()
            ):
                raise ModelSettingsValidationError(
                    f"{source}.{name} must be an integer"
                )


def resolve_model_settings_overrides(
    model_name: str,
    overrides: Mapping[str, Any] | None,
    *,
    models_config: Mapping[str, Any] | None = None,
    source: str = "model_settings",
) -> dict[str, Any]:
    """Validate, capability-filter, and copy overrides for one effective model."""
    if not overrides:
        return {}
    if not isinstance(overrides, Mapping):
        raise ModelSettingsValidationError(f"{source} must be an object")

    validate_model_settings(overrides, source=source)
    if models_config is None:
        from code_puppy.model_factory import ModelFactory

        models_config = ModelFactory.load_config()

    from code_puppy.config import model_supports_setting

    resolved = {
        name: deepcopy(value)
        for name, value in overrides.items()
        if model_supports_setting(
            model_name,
            name,
            models_config=dict(models_config),
        )
    }

    model_config = models_config.get(model_name, {})
    reasoning_effort = resolved.get("reasoning_effort")
    if reasoning_effort == "xhigh" and not supports_xhigh_reasoning(
        model_name, model_config
    ):
        raise ModelSettingsCapabilityError(
            f"{source}.reasoning_effort does not support xhigh for {model_name}"
        )
    if reasoning_effort == "max" and not supports_max_reasoning(
        model_name, model_config
    ):
        raise ModelSettingsCapabilityError(
            f"{source}.reasoning_effort does not support max for {model_name}"
        )

    return resolved


@contextmanager
def model_settings_scope(
    model_name: str,
    settings: Mapping[str, Any] | None,
    *,
    raw_settings: Mapping[str, Any] | None = None,
    models_config: Mapping[str, Any] | None = None,
) -> Iterator[None]:
    """Expose agent settings to construction- and request-time plugin reads."""
    token = _scoped_model_settings.set(
        (
            model_name,
            deepcopy(dict(settings or {})),
            deepcopy(dict(raw_settings or {})),
            deepcopy(dict(models_config or {})),
        )
    )
    try:
        yield
    finally:
        _scoped_model_settings.reset(token)


def get_scoped_model_settings(model_name: str) -> dict[str, Any]:
    """Return a defensive copy of settings scoped to ``model_name``."""
    scoped = _scoped_model_settings.get()
    if scoped is None:
        return {}
    scoped_name, resolved, raw_settings, models_config = scoped
    if scoped_name == model_name:
        return deepcopy(resolved)
    if not raw_settings or model_name not in models_config:
        return {}
    return resolve_model_settings_overrides(
        model_name,
        raw_settings,
        models_config=models_config,
        source="agent model_settings",
    )
