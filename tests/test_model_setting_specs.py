"""Validation and task-local scoping for agent model settings."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from code_puppy.agents._builder import load_model_with_fallback
from code_puppy.config import get_all_model_settings
from code_puppy.model_factory import ModelFactory
from code_puppy.model_setting_specs import (
    ModelSettingsValidationError,
    get_scoped_model_settings,
    model_settings_scope,
    resolve_model_settings_overrides,
    validate_model_settings,
)
from code_puppy.round_robin_model import RoundRobinModel


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"reasoning_effort": []}, "reasoning_effort must be one of"),
        ({"reasoning_effort": "turbo"}, "reasoning_effort must be one of"),
        ({"temperature": 1.5}, "temperature must be between"),
        ({"temperature": 10**1000}, "temperature must be between"),
        ({"budget_tokens": 1024.5}, "budget_tokens must be an integer"),
        ({"interleaved_thinking": "yes"}, "must be a boolean"),
        ({"fast": None}, "fast must not be null"),
        ({"plugin_setting": {"nested": True}}, "string, number, or boolean"),
    ],
)
def test_validate_model_settings_rejects_malformed_values(settings, message):
    with pytest.raises(ModelSettingsValidationError, match=message):
        validate_model_settings(settings)


def test_resolver_filters_unsupported_settings_and_keeps_plugin_scalars():
    models = {
        "plugin-model": {
            "supported_settings": ["reasoning_effort", "fast"],
        }
    }

    resolved = resolve_model_settings_overrides(
        "plugin-model",
        {
            "reasoning_effort": "high",
            "fast": True,
            "unsupported": "ignored",
        },
        models_config=models,
    )

    assert resolved == {"reasoning_effort": "high", "fast": True}


def test_resolver_rejects_model_specific_reasoning_effort():
    models = {
        "ordinary-gpt": {
            "supported_settings": ["reasoning_effort"],
            "supports_xhigh_reasoning": False,
        }
    }

    with pytest.raises(ModelSettingsValidationError, match="does not support xhigh"):
        resolve_model_settings_overrides(
            "ordinary-gpt",
            {"reasoning_effort": "xhigh"},
            models_config=models,
        )


@pytest.mark.asyncio
async def test_model_settings_scope_is_task_local_and_defensively_copied():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def worker(value: bool):
        source = {"fast": value, "nested": {"items": [value]}}
        with model_settings_scope("shared-model", source):
            source["nested"]["items"].append("outside")
            first = get_scoped_model_settings("shared-model")
            first["nested"]["items"].append("returned-copy")
            entered.set()
            await release.wait()
            return get_scoped_model_settings("shared-model")

    first_task = asyncio.create_task(worker(True))
    await entered.wait()
    second_task = asyncio.create_task(worker(False))
    await asyncio.sleep(0)
    release.set()

    first, second = await asyncio.gather(first_task, second_task)
    assert first == {"fast": True, "nested": {"items": [True]}}
    assert second == {"fast": False, "nested": {"items": [False]}}
    assert get_scoped_model_settings("shared-model") == {}


def test_scope_resolves_raw_settings_for_nested_provider_models():
    models = {
        "round-robin": {"type": "round_robin"},
        "claude-child": {"supported_settings": ["interleaved_thinking"]},
    }

    with model_settings_scope(
        "round-robin",
        {},
        raw_settings={"interleaved_thinking": True},
        models_config=models,
    ):
        assert get_scoped_model_settings("claude-child") == {
            "interleaved_thinking": True
        }


def test_scope_owns_an_immutable_model_catalog_snapshot():
    models = {
        "wrapper": {},
        "child": {"supported_settings": ["fast"]},
    }

    with model_settings_scope(
        "wrapper",
        {},
        raw_settings={"fast": True},
        models_config=models,
    ):
        models["child"]["supported_settings"].clear()
        assert get_scoped_model_settings("child") == {"fast": True}


def test_model_construction_observes_resolved_agent_settings():
    models = {
        "plugin-model": {
            "supported_settings": ["interleaved_thinking", "fast"],
        }
    }
    observed = {}

    def fake_get_model(model_name, _models_config):
        observed.update(get_all_model_settings(model_name))
        return object()

    with patch.object(ModelFactory, "get_model", side_effect=fake_get_model):
        model, effective_name = load_model_with_fallback(
            "plugin-model",
            models,
            "group",
            agent_name="plugin-agent",
            model_settings_overrides={
                "interleaved_thinking": True,
                "fast": True,
                "unsupported": True,
            },
        )

    assert model is not None
    assert effective_name == "plugin-model"
    assert observed == {"interleaved_thinking": True, "fast": True}


def test_fallback_construction_resolves_settings_for_selected_model():
    models = {
        "fallback-model": {
            "supported_settings": ["reasoning_effort"],
        }
    }
    observed = {}

    def fake_get_model(model_name, _models_config):
        if model_name == "missing-model":
            raise ValueError("not found in configuration")
        observed.update(get_all_model_settings(model_name))
        return object()

    with (
        patch.object(ModelFactory, "get_model", side_effect=fake_get_model),
        patch(
            "code_puppy.agents._builder.get_global_model_name",
            return_value="fallback-model",
        ),
        patch("code_puppy.agents._builder.emit_warning"),
        patch("code_puppy.agents._builder.emit_info"),
    ):
        _model, effective_name = load_model_with_fallback(
            "missing-model",
            models,
            "group",
            agent_name="fallback-agent",
            model_settings_overrides={"reasoning_effort": "high"},
        )

    assert effective_name == "fallback-model"
    assert observed == {"reasoning_effort": "high"}


def test_fallback_skips_candidate_that_rejects_reasoning_level():
    models = {
        "ordinary": {
            "supported_settings": ["reasoning_effort"],
        },
        "capable": {
            "name": "gpt-5.6-capable",
            "supported_settings": ["reasoning_effort"],
        },
    }

    def fake_get_model(model_name, _models_config):
        if model_name == "missing-model":
            raise ValueError("not found in configuration")
        return object()

    with (
        patch.object(ModelFactory, "get_model", side_effect=fake_get_model),
        patch(
            "code_puppy.agents._builder.get_global_model_name",
            return_value="ordinary",
        ),
        patch("code_puppy.agents._builder.emit_warning"),
        patch("code_puppy.agents._builder.emit_info"),
    ):
        _model, effective_name = load_model_with_fallback(
            "missing-model",
            models,
            "group",
            agent_name="fallback-agent",
            model_settings_overrides={"reasoning_effort": "max"},
        )

    assert effective_name == "capable"


def test_fallback_skips_factory_that_returns_none():
    models = {"empty": {}, "working": {}}

    def fake_get_model(model_name, _models_config):
        if model_name == "missing-model":
            raise ValueError("not found in configuration")
        return None if model_name == "empty" else object()

    with (
        patch.object(ModelFactory, "get_model", side_effect=fake_get_model),
        patch(
            "code_puppy.agents._builder.get_global_model_name",
            return_value="empty",
        ),
        patch("code_puppy.agents._builder.emit_warning"),
        patch("code_puppy.agents._builder.emit_info"),
    ):
        model, effective_name = load_model_with_fallback(
            "missing-model",
            models,
            "group",
            agent_name="fallback-agent",
        )

    assert model is not None
    assert effective_name == "working"


@pytest.mark.asyncio
async def test_round_robin_sends_agent_settings_to_selected_child():
    models = {
        "pool": {
            "type": "round_robin",
            "models": ["luna-child"],
        },
        "luna-child": {
            "type": "openai",
            "name": "gpt-5.6-luna",
            "api": "responses",
            "supported_settings": ["reasoning_effort"],
        },
    }
    child = MagicMock()
    child.model_name = "gpt-5.6-luna"
    child.prepare_request.side_effect = lambda settings, params: (settings, params)
    child.request = AsyncMock(return_value=MagicMock())

    with (
        model_settings_scope(
            "pool",
            {},
            raw_settings={"reasoning_effort": "high"},
            models_config=models,
        ),
        patch("code_puppy.model_factory.get_api_key", return_value="secret"),
        patch("code_puppy.model_factory.make_openai_provider"),
        patch(
            "code_puppy.model_factory.OpenAIResponsesModel",
            return_value=child,
        ),
        patch("code_puppy.model_factory.get_yolo_mode", return_value=True),
        patch("code_puppy.config.get_custom_model_settings", return_value={}),
    ):
        pool = ModelFactory.get_model("pool", models)
        await pool.request([], ModelSettings(max_tokens=1), MagicMock())

    sent_settings = child.request.await_args.args[1]
    assert sent_settings["extra_body"]["reasoning"]["effort"] == "high"


@pytest.mark.asyncio
async def test_nested_round_robin_stream_forwards_leaf_settings_without_mutation():
    nested_extra_body = {"reasoning": {"effort": "high", "trace": ["agent"]}}
    source_settings = ModelSettings(extra_body=deepcopy(nested_extra_body))
    leaf = MagicMock()
    leaf.model_name = "leaf"
    prepared_settings = []

    def prepare_request(settings, params):
        prepared_settings.append(deepcopy(dict(settings or {})))
        settings["extra_body"]["reasoning"]["trace"].append("provider")
        return settings, params

    @asynccontextmanager
    async def request_stream(
        _messages, _settings, _params, _run_context=None, **_kwargs
    ):
        yield MagicMock()

    leaf.prepare_request.side_effect = prepare_request
    leaf.request_stream = request_stream
    inner = RoundRobinModel(
        leaf,
        per_model_settings=[source_settings],
    )
    inner_prepare = MagicMock(wraps=inner.prepare_request)
    inner.prepare_request = inner_prepare
    outer = RoundRobinModel(
        inner,
        per_model_settings=[ModelSettings(temperature=0.25)],
    )

    async with outer.request_stream(
        [], ModelSettings(max_tokens=64), ModelRequestParameters()
    ):
        pass

    assert inner_prepare.call_args.args[0]["temperature"] == 0.25
    assert prepared_settings[0]["extra_body"] == nested_extra_body
    assert source_settings["extra_body"] == nested_extra_body


@pytest.mark.asyncio
async def test_concurrent_nested_round_robin_scopes_are_isolated():
    models = {
        "outer": {"type": "round_robin", "models": ["inner"]},
        "inner": {"type": "round_robin", "models": ["leaf"]},
        "leaf": {
            "type": "openai",
            "name": "gpt-5.6-leaf",
            "api": "responses",
            "supported_settings": ["reasoning_effort"],
        },
    }
    entered = 0
    both_entered = asyncio.Event()
    observed = []

    def make_leaf(*_args, **_kwargs):
        nonlocal entered
        leaf = MagicMock()
        leaf.model_name = "leaf"
        leaf.prepare_request.side_effect = lambda settings, params: (settings, params)

        async def request(_messages, settings, _params):
            nonlocal entered
            entered += 1
            if entered == 2:
                both_entered.set()
            await both_entered.wait()
            observed.append(deepcopy(settings["extra_body"]["reasoning"]))
            return MagicMock()

        leaf.request = request
        return leaf

    async def worker(reasoning_effort):
        raw = {"reasoning_effort": reasoning_effort}
        with model_settings_scope(
            "outer",
            {},
            raw_settings=raw,
            models_config=models,
        ):
            pool = ModelFactory.get_model("outer", models)
            await pool.request(
                [], ModelSettings(max_tokens=64), ModelRequestParameters()
            )

    with (
        patch("code_puppy.model_factory.get_api_key", return_value="secret"),
        patch("code_puppy.model_factory.make_openai_provider"),
        patch(
            "code_puppy.model_factory.OpenAIResponsesModel",
            side_effect=make_leaf,
        ),
        patch("code_puppy.model_factory.get_yolo_mode", return_value=True),
        patch("code_puppy.config.get_custom_model_settings", return_value={}),
    ):
        await asyncio.gather(worker("high"), worker("low"))

    assert sorted(item["effort"] for item in observed) == ["high", "low"]
    assert all(item["context"] == "all_turns" for item in observed)


def test_round_robin_rejects_unavailable_child():
    models = {
        "pool": {"type": "round_robin", "models": ["unavailable"]},
        "unavailable": {"type": "anthropic", "name": "claude-test"},
    }

    with patch("code_puppy.model_factory.get_api_key", return_value=None):
        with pytest.raises(ValueError, match="child 'unavailable'.*initialized"):
            ModelFactory.get_model("pool", models)


def test_anthropic_construction_uses_agent_interleaved_thinking_header():
    models = {
        "friendly-claude": {
            "type": "anthropic",
            "name": "claude-sonnet-4-5",
            "provider": "anthropic",
            "supported_settings": ["interleaved_thinking"],
        }
    }
    anthropic_client = MagicMock()

    with (
        model_settings_scope("friendly-claude", {"interleaved_thinking": True}),
        patch("code_puppy.model_factory.get_api_key", return_value="secret"),
        patch("code_puppy.model_factory.ClaudeCacheAsyncClient"),
        patch(
            "code_puppy.model_factory.AsyncAnthropic",
            return_value=anthropic_client,
        ) as async_anthropic,
        patch("code_puppy.model_factory.make_anthropic_provider"),
        patch("code_puppy.model_factory.AnthropicModel", return_value=MagicMock()),
    ):
        ModelFactory.get_model("friendly-claude", models)

    async_anthropic.assert_called_once()
    assert async_anthropic.call_args.kwargs["default_headers"] == {
        "anthropic-beta": "interleaved-thinking-2025-05-14"
    }
