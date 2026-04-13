from __future__ import annotations

from dataclasses import dataclass
from time import sleep
from typing import Callable

from convexopt_tutor_agent.core.json_utils import extract_first_json_object
from convexopt_tutor_agent.core.schema import AppSettings

try:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError
except ImportError:  # pragma: no cover - handled at runtime
    OpenAI = None
    APIConnectionError = Exception
    APIStatusError = Exception
    APITimeoutError = Exception
    RateLimitError = Exception


@dataclass(slots=True)
class KimiClientConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: int


class KimiClient:
    def supports_json_mode(self, model: str) -> bool:
        return "thinking" not in model.lower()

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        settings: AppSettings,
        progress: Callable[[str], None] | None = None,
    ) -> dict:
        if OpenAI is None:
            raise RuntimeError("The `openai` package is missing. Run `py -m pip install -r requirements.txt` first.")
        if not settings.api_key:
            raise RuntimeError("Moonshot API Key is missing.")

        client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.request_timeout_seconds,
        )

        request_kwargs = {
            "model": settings.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": settings.temperature,
        }
        if self.supports_json_mode(settings.model):
            request_kwargs["response_format"] = {"type": "json_object"}

        retry_delays = (2, 5, 8)
        last_error: Exception | None = None

        for attempt in range(len(retry_delays) + 1):
            try:
                response = client.chat.completions.create(**request_kwargs)
                message = response.choices[0].message
                content = message.content or ""
                return extract_first_json_object(content)
            except RateLimitError as exc:
                last_error = exc
                if not _is_engine_overloaded(exc):
                    raise RuntimeError("Moonshot API rate limit reached. Please try again later or check your quota.") from exc
            except (APITimeoutError, APIConnectionError) as exc:
                last_error = exc
            except APIStatusError as exc:
                last_error = exc
                if not _should_retry_status_error(exc):
                    raise RuntimeError(f"Moonshot API request failed: {_compact_error_message(exc)}") from exc

            if attempt >= len(retry_delays):
                break

            delay = retry_delays[attempt]
            if progress is not None:
                progress(
                    f"Kimi is busy or the network is unstable. Retrying in {delay} seconds ({attempt + 1}/{len(retry_delays)})."
                )
            sleep(delay)

        if last_error is not None and _is_engine_overloaded(last_error):
            raise RuntimeError("Kimi is currently overloaded. Automatic retries were exhausted. Please try again later.") from last_error
        if isinstance(last_error, APITimeoutError):
            raise RuntimeError("Request to Kimi timed out. Automatic retries were exhausted. Please try again later.") from last_error
        if isinstance(last_error, APIConnectionError):
            raise RuntimeError("Unable to reach the Moonshot API. Check your network connection or Base URL setting.") from last_error
        if last_error is not None:
            raise RuntimeError(f"Moonshot API request failed: {_compact_error_message(last_error)}") from last_error
        raise RuntimeError("Moonshot API request failed.")

    def describe_status(self, settings: AppSettings) -> str:
        if self.supports_json_mode(settings.model):
            return "The current model uses JSON mode."
        return "The current model does not support JSON mode and will fall back to plain-text JSON extraction."


def _compact_error_message(exc: Exception) -> str:
    return " ".join(str(exc).split())


def _is_engine_overloaded(exc: Exception) -> bool:
    message = _compact_error_message(exc).lower()
    return "engine_overloaded_error" in message or "overloaded" in message


def _should_retry_status_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    return _is_engine_overloaded(exc)
