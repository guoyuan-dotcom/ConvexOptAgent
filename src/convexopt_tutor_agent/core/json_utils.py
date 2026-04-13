from __future__ import annotations

import json


def extract_first_json_object(text: str) -> dict:
    text = text.strip()
    if not text:
        raise ValueError("LLM returned empty content.")

    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value

    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in LLM response.")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                return json.loads(candidate)

    raise ValueError("Unable to extract a complete JSON object from LLM response.")

