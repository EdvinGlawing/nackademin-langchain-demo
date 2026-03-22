from __future__ import annotations

import json
import re
from typing import Any

from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _sanitize_text(text: str) -> str:
    cleaned = text

    suspicious_patterns = [
        r"(?i)ignore previous instructions",
        r"(?i)system prompt",
        r"(?i)developer message",
        r"(?i)reveal hidden prompt",
        r"(?i)chain of thought",
        r"(?i)<tool_call>",
        r"(?i)</tool_call>",
    ]
    for pattern in suspicious_patterns:
        cleaned = re.sub(pattern, "[redacted]", cleaned)

    cleaned = re.sub(r"[A-Za-z]:\\[^\s]+", "[redacted-path]", cleaned)
    cleaned = re.sub(r"/[A-Za-z0-9_./-]+", "[redacted-path]", cleaned)
    cleaned = re.sub(r"sk-[A-Za-z0-9_-]+", "[redacted-key]", cleaned)

    sensitive_json_patterns = [
        r'"employee_id"\s*:\s*"[^"]+"',
        r'"device_serial"\s*:\s*"[^"]+"',
        r'"internal_note"\s*:\s*"[^"]+"',
    ]
    for pattern in sensitive_json_patterns:
        cleaned = re.sub(pattern, '"redacted_field":"[redacted]"', cleaned)

    if len(cleaned) > 1500:
        cleaned = cleaned[:1500] + "\n...[truncated by middleware]"

    return cleaned


def _tool_name_from_request(request) -> str:
    tool_call = getattr(request, "tool_call", {}) or {}
    return tool_call.get("name", "okänt_tool")


def _tool_args_from_request(request) -> str:
    tool_call = getattr(request, "tool_call", {}) or {}
    args = tool_call.get("args", {})

    try:
        text = json.dumps(args, ensure_ascii=False)
    except Exception:
        text = str(args)

    if len(text) > 220:
        text = text[:220] + "..."
    return text


@wrap_tool_call
async def sanitize_mcp_output(request, handler):
    """
    Async middleware som:
    1. Visar vilket tool agenten använder
    2. Sanerar tool-output innan modellen får tillbaka den
    """
    tool_name = _tool_name_from_request(request)
    tool_args = _tool_args_from_request(request)

    print(f"\nAgenten använder verktyget: {tool_name}")
    print(f"Argument: {tool_args}")

    try:
        result = await handler(request)
    except Exception as exc:
        print(f"Verktygsfel i {tool_name}: {exc}")
        return ToolMessage(
            content=f"Tool error: Kunde inte köra verktyget säkert. ({exc})",
            tool_call_id=request.tool_call["id"],
        )

    if isinstance(result, ToolMessage):
        raw_text = _content_to_text(result.content)
        safe_text = _sanitize_text(raw_text)

        preview = safe_text.replace("\n", " ")
        if len(preview) > 220:
            preview = preview[:220] + "..."

        print(f"Klart med verktyget: {tool_name}")
        print(f"Förhandsvisning av resultat: {preview}")

        return ToolMessage(
            content=safe_text,
            tool_call_id=result.tool_call_id,
        )

    print(f"Klart med verktyget: {tool_name}")
    return result