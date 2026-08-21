"""Tests for agent-scoped model request settings."""

import json
from unittest.mock import MagicMock, patch

import pytest

from code_puppy.agents._builder import (
    _assemble_instructions,
    build_pydantic_agent,
    build_tool_probe_for_agent,
)
from code_puppy.agents._runtime import _should_prepend_system_prompt
from code_puppy.agents.json_agent import JSONAgent
from code_puppy.model_factory import (
    ModelFactory,
    make_model_settings,
    model_uses_openai_responses_api,
)


def _json_agent_config(**overrides):
    config = {
        "name": "test-agent",
        "description": "Exercises agent model settings",
        "system_prompt": "Test prompt",
        "tools": [],
    }
    config.update(overrides)
    return config


def test_json_agent_returns_defensive_model_settings_copy(tmp_path):
    agent_file = tmp_path / "test-agent.json"
    agent_file.write_text(
        json.dumps(
            _json_agent_config(
                model_settings={"reasoning_effort": "high", "verbosity": "low"}
            )
        )
    )

    agent = JSONAgent(str(agent_file))
    settings = agent.get_model_settings_overrides()

    assert settings == {"reasoning_effort": "high", "verbosity": "low"}
    settings["reasoning_effort"] = "low"
    assert agent.get_model_settings_overrides()["reasoning_effort"] == "high"


def test_json_agent_rejects_non_object_model_settings(tmp_path):
    agent_file = tmp_path / "test-agent.json"
    agent_file.write_text(json.dumps(_json_agent_config(model_settings="high")))

    with pytest.raises(ValueError, match="'model_settings' must be an object"):
        JSONAgent(str(agent_file))


def test_json_agent_rejects_invalid_model_setting_value(tmp_path):
    agent_file = tmp_path / "test-agent.json"
    agent_file.write_text(
        json.dumps(_json_agent_config(model_settings={"reasoning_effort": []}))
    )

    with pytest.raises(ValueError, match="reasoning_effort must be one of"):
        JSONAgent(str(agent_file))


def test_agent_creator_rejects_non_object_model_settings():
    from code_puppy.agents.agent_creator_agent import AgentCreatorAgent

    errors = AgentCreatorAgent().validate_agent_json(
        _json_agent_config(model_settings="high")
    )

    assert "'model_settings' must be an object" in errors


def test_agent_creator_rejects_invalid_model_setting_value():
    from code_puppy.agents.agent_creator_agent import AgentCreatorAgent

    errors = AgentCreatorAgent().validate_agent_json(
        _json_agent_config(model_settings={"temperature": 4})
    )

    assert any("temperature must be between" in error for error in errors)


def test_agent_settings_override_per_model_values_before_provider_translation():
    model_config = {
        "gpt-5-test": {
            "type": "openai",
            "name": "gpt-5-test",
            "supported_settings": ["reasoning_effort", "verbosity"],
        }
    }

    with (
        patch.object(ModelFactory, "load_config", return_value=model_config),
        patch(
            "code_puppy.config.get_effective_model_settings",
            return_value={"reasoning_effort": "low", "verbosity": "low"},
        ),
        patch("code_puppy.config.get_custom_model_settings", return_value={}),
        patch("code_puppy.model_factory.get_yolo_mode", return_value=True),
    ):
        settings = make_model_settings(
            "gpt-5-test",
            max_tokens=4096,
            overrides={"reasoning_effort": "high", "unsupported": "ignored"},
        )

    assert settings["openai_reasoning_effort"] == "high"
    assert settings["extra_body"]["verbosity"] == "low"
    assert "unsupported" not in settings


def test_friendly_alias_uses_underlying_gpt_5_6_provider_translation():
    model_config = {
        "luna": {
            "type": "codex",
            "name": "gpt-5.6-luna",
            "supported_settings": [
                "reasoning_effort",
                "reasoning_context",
                "reasoning_mode",
                "summary",
                "verbosity",
            ],
        }
    }

    with (
        patch.object(ModelFactory, "load_config", return_value=model_config),
        patch("code_puppy.config.get_effective_model_settings", return_value={}),
        patch("code_puppy.config.get_custom_model_settings", return_value={}),
        patch("code_puppy.model_factory.get_yolo_mode", return_value=True),
    ):
        settings = make_model_settings(
            "luna",
            max_tokens=4096,
            overrides={"reasoning_effort": "high"},
            models_config=model_config,
        )

    assert settings["extra_body"]["reasoning"] == {
        "effort": "high",
        "summary": "auto",
        "context": "all_turns",
        "mode": "standard",
    }


def test_friendly_anthropic_alias_infers_settings_without_capability_metadata():
    model_config = {
        "friendly": {
            "type": "anthropic",
            "name": "claude-sonnet-4-5",
        }
    }

    with (
        patch("code_puppy.config.get_effective_model_settings", return_value={}),
        patch("code_puppy.config.get_custom_model_settings", return_value={}),
        patch("code_puppy.model_factory.get_yolo_mode", return_value=True),
    ):
        settings = make_model_settings(
            "friendly",
            max_tokens=4096,
            overrides={"extended_thinking": "off", "budget_tokens": 2048},
            models_config=model_config,
        )

    assert "anthropic_thinking" not in settings


@pytest.mark.parametrize("model_type", ["openrouter", "custom_openai"])
def test_openai_compatible_claude_ids_do_not_get_anthropic_wire_settings(model_type):
    model_config = {
        "friendly": {
            "type": model_type,
            "name": "anthropic/claude-sonnet-4",
        }
    }

    with (
        patch("code_puppy.config.get_effective_model_settings", return_value={}),
        patch("code_puppy.config.get_custom_model_settings", return_value={}),
        patch("code_puppy.model_factory.get_yolo_mode", return_value=True),
    ):
        settings = make_model_settings(
            "friendly",
            max_tokens=4096,
            overrides={"extended_thinking": "enabled"},
            models_config=model_config,
        )

    assert "anthropic_thinking" not in settings
    assert "anthropic_cache_instructions" not in settings


@pytest.mark.parametrize("model_type", ["anthropic", "custom_anthropic"])
def test_native_anthropic_adapters_keep_anthropic_wire_settings(model_type):
    model_config = {
        "friendly": {
            "type": model_type,
            "name": "claude-sonnet-4-5",
            "supported_settings": ["extended_thinking", "budget_tokens"],
        }
    }

    with (
        patch("code_puppy.config.get_effective_model_settings", return_value={}),
        patch("code_puppy.config.get_custom_model_settings", return_value={}),
        patch("code_puppy.model_factory.get_yolo_mode", return_value=True),
    ):
        settings = make_model_settings(
            "friendly",
            max_tokens=4096,
            overrides={"extended_thinking": "enabled", "budget_tokens": 2048},
            models_config=model_config,
        )

    assert settings["anthropic_thinking"] == {
        "type": "enabled",
        "budget_tokens": 2048,
    }


@pytest.mark.parametrize(
    ("underlying_name", "uses_responses"),
    [("gpt-4.1", False), ("gpt-5", True), ("gpt-5.6-chat", True)],
)
def test_azure_foundry_api_selection_matches_deployment_family(
    underlying_name, uses_responses
):
    assert (
        model_uses_openai_responses_api(
            "friendly-foundry",
            {
                "type": "azure_foundry_openai",
                "name": underlying_name,
                "api": "responses",  # Plugin selection, not generic metadata, wins.
            },
        )
        is uses_responses
    )


def test_openai_construction_and_settings_share_responses_api_detection():
    model_config = {
        "friendly": {
            "type": "openai",
            "name": "gpt-5.6-codex",
            "api": "responses",
        }
    }
    responses_model = MagicMock()

    with (
        patch("code_puppy.model_factory.get_api_key", return_value="secret"),
        patch("code_puppy.model_factory.make_openai_provider"),
        patch(
            "code_puppy.model_factory.OpenAIResponsesModel",
            return_value=responses_model,
        ) as responses_cls,
        patch("code_puppy.model_factory.OpenAIChatModel") as chat_cls,
    ):
        model = ModelFactory.get_model("friendly", model_config)

    assert model is responses_model
    responses_cls.assert_called_once()
    chat_cls.assert_not_called()


def test_prompt_assembly_uses_resolved_model_and_agent_settings():
    agent = MagicMock()
    agent.get_full_system_prompt.return_value = "instructions"
    agent.get_available_tools.return_value = []
    overrides = {"extended_thinking": "adaptive"}

    with (
        patch("code_puppy.agents._builder.load_puppy_rules", return_value=None),
        patch(
            "code_puppy.tools.has_extended_thinking_active", return_value=True
        ) as thinking_active,
        patch("code_puppy.model_utils.prepare_prompt_for_model") as prepare,
    ):
        prepare.return_value = MagicMock(instructions="prepared")
        result = _assemble_instructions(agent, "fallback-claude", overrides)

    assert result == "prepared"
    assert "extended thinking enabled" in agent._resolved_system_prompt
    thinking_active.assert_called_once_with(
        "fallback-claude", settings_overrides=overrides
    )
    prepare.assert_called_once()
    assert prepare.call_args.args[0] == "fallback-claude"
    assert "extended thinking enabled" in prepare.call_args.args[1]


def test_first_turn_prompt_uses_resolved_identity_and_assembled_guidance():
    agent = MagicMock()
    agent._message_history = []
    agent._last_model_name = "fallback-claude"
    agent._resolved_system_prompt = "assembled extended-thinking guidance"
    agent.get_model_name.return_value = "missing-pin"

    with patch("code_puppy.model_utils.prepare_prompt_for_model") as prepare:
        prepare.return_value = MagicMock(user_prompt="prepared user prompt")
        result = _should_prepend_system_prompt(agent, "hello")

    assert result == "prepared user prompt"
    assert prepare.call_args.kwargs == {
        "model_name": "fallback-claude",
        "system_prompt": "assembled extended-thinking guidance",
        "user_prompt": "hello",
        "prepend_system_to_user": True,
    }


def test_custom_params_remain_the_final_wire_level_override():
    model_config = {
        "gpt-5-test": {
            "type": "openai",
            "name": "gpt-5-test",
            "supported_settings": ["verbosity"],
        }
    }

    with (
        patch.object(ModelFactory, "load_config", return_value=model_config),
        patch("code_puppy.config.get_effective_model_settings", return_value={}),
        patch(
            "code_puppy.config.get_custom_model_settings",
            return_value={"verbosity": "high"},
        ),
        patch("code_puppy.model_factory.get_yolo_mode", return_value=True),
    ):
        settings = make_model_settings(
            "gpt-5-test",
            max_tokens=4096,
            overrides={"verbosity": "low"},
        )

    assert settings["extra_body"]["verbosity"] == "high"


def test_tool_probe_uses_agent_settings_and_capability_aware_fallback():
    agent = MagicMock()
    agent.name = "probe-agent"
    agent.get_model_name.return_value = "ordinary"
    agent.get_model_settings_overrides.return_value = {"reasoning_effort": "max"}
    agent.get_available_tools.return_value = ["read_file"]
    models = {
        "ordinary": {
            "supported_settings": ["reasoning_effort"],
        },
        "capable": {
            "name": "gpt-5.6-capable",
            "supported_settings": ["reasoning_effort"],
        },
    }
    observed_construction_settings = {}
    built_model = MagicMock()
    probe = MagicMock()

    def get_model(model_name, _models_config):
        from code_puppy.config import get_all_model_settings

        observed_construction_settings[model_name] = get_all_model_settings(model_name)
        return built_model

    with (
        patch.object(ModelFactory, "load_config", return_value=models),
        patch.object(ModelFactory, "get_model", side_effect=get_model),
        patch(
            "code_puppy.agents._builder.get_global_model_name",
            return_value="ordinary",
        ),
        patch("code_puppy.agents._builder.emit_warning"),
        patch("code_puppy.agents._builder.emit_info"),
        patch("code_puppy.agents._builder.PydanticAgent", return_value=probe),
        patch("code_puppy.tools.register_tools_for_agent") as register_tools,
    ):
        result = build_tool_probe_for_agent(agent)

    assert result is probe
    assert observed_construction_settings == {"capable": {"reasoning_effort": "max"}}
    register_tools.assert_called_once_with(
        probe,
        ["read_file"],
        model_name="capable",
        agent_name="probe-agent",
        settings_overrides={"reasoning_effort": "max"},
    )


def test_main_agent_builder_passes_agent_model_settings():
    agent = MagicMock()
    agent.name = "test-agent"
    agent.get_model_name.return_value = "gpt-5-test"
    agent.get_model_settings_overrides.return_value = {"reasoning_effort": "high"}
    agent.get_available_tools.return_value = []

    model = MagicMock()
    probe = MagicMock()
    probe._tools = {}
    final = MagicMock()
    final._tools = {}
    observed_wrap_settings = {}

    def wrap_agent(_agent, built, **_kwargs):
        from code_puppy.config import get_all_model_settings

        observed_wrap_settings.update(get_all_model_settings("gpt-5-test"))
        return built

    with (
        patch.object(
            ModelFactory,
            "load_config",
            return_value={"gpt-5-test": {"supported_settings": ["reasoning_effort"]}},
        ),
        patch(
            "code_puppy.agents._builder.load_model_with_fallback",
            return_value=(model, "gpt-5-test"),
        ),
        patch(
            "code_puppy.agents._builder._assemble_instructions",
            return_value="instructions",
        ) as assemble_instructions,
        patch("code_puppy.agents._builder.load_mcp_servers", return_value=[]),
        patch("code_puppy.agents._builder.make_model_settings") as make_settings,
        patch(
            "code_puppy.agents._builder.make_history_processor",
            return_value=MagicMock(),
        ),
        patch(
            "code_puppy.agents._builder.make_steer_history_processor",
            return_value=MagicMock(),
        ),
        patch("code_puppy.agents._builder.build_tool_output_limits", return_value=[]),
        patch("code_puppy.agents._builder.build_response_clamp"),
        patch(
            "code_puppy.agents._builder.PydanticAgent",
            side_effect=[probe, final],
        ),
        patch("code_puppy.tools.register_tools_for_agent") as register_tools,
        patch(
            "code_puppy.agents._builder.on_wrap_pydantic_agent",
            side_effect=wrap_agent,
        ),
    ):
        result = build_pydantic_agent(agent)

    assert result is final
    assert observed_wrap_settings == {"reasoning_effort": "high"}
    assert [
        call.kwargs["settings_overrides"] for call in register_tools.call_args_list
    ] == [
        {"reasoning_effort": "high"},
        {"reasoning_effort": "high"},
    ]
    assemble_instructions.assert_called_once_with(
        agent,
        "gpt-5-test",
        {"reasoning_effort": "high"},
    )
    make_settings.assert_called_once_with(
        "gpt-5-test",
        overrides={"reasoning_effort": "high"},
        models_config={"gpt-5-test": {"supported_settings": ["reasoning_effort"]}},
    )
