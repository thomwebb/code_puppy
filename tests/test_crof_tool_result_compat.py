import asyncio


def _collect_async_iter(async_iterable):
    async def _run():
        return [item async for item in async_iterable]

    return asyncio.run(_run())


def test_crof_tool_results_mapped_as_user_messages():
    """crof.ai appears to error on role='tool' messages; we degrade to role='user'."""
    from pydantic_ai.messages import (
        ModelRequest,
        ModelResponse,
        ToolCallPart,
        ToolReturnPart,
    )

    from code_puppy.model_factory import ModelFactory

    cfg = ModelFactory.load_config()
    model = ModelFactory.get_model("crof-kimi-k2.5-lightning", cfg)

    message = ModelRequest(
        parts=[
            ToolReturnPart(
                tool_name="list_files",
                content="ok",
                tool_call_id="call_123",
            )
        ]
    )

    mapped = _collect_async_iter(model._map_user_message(message))
    assert mapped
    assert mapped[0]["role"] == "user"
    assert "TOOL RESULT" in mapped[0]["content"]

    # And tool calls in assistant messages should be flattened into content
    resp = ModelResponse(parts=[ToolCallPart(tool_name="list_files", args={"a": 1})])
    assistant_msg = model._map_model_response(resp)
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg.get("tool_calls") is None
    assert "TOOL CALL" in (assistant_msg.get("content") or "")


def test_crof_does_not_set_parallel_tool_calls():
    """crof config does not advertise parallel_tool_calls; don't send it."""
    from code_puppy.model_factory import make_model_settings

    settings = make_model_settings("crof-kimi-k2.5-lightning")
    assert getattr(settings, "parallel_tool_calls", None) is None
