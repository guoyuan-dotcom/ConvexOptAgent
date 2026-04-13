from __future__ import annotations

import json
import os
from pathlib import Path

from convexopt_tutor_agent.core.schema import AppSettings


class SettingsStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or self._default_path()

    def _default_path(self) -> Path:
        local_app_data = os.getenv("LOCALAPPDATA")
        if local_app_data:
            base_path = Path(local_app_data) / "ConvexOptAgent"
        else:
            base_path = Path.home() / ".convexopt_agent"
        return base_path / "settings.json"

    def load(self) -> AppSettings:
        defaults = AppSettings()
        if not self.path.exists():
            return defaults

        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return defaults

        return AppSettings(
            api_key=str(raw.get("api_key", "")),
            base_url=str(raw.get("base_url", defaults.base_url)),
            model=str(raw.get("model", defaults.model)),
            temperature=float(raw.get("temperature", defaults.temperature)),
            reasoning_mode=str(raw.get("reasoning_mode", defaults.reasoning_mode)),
            request_timeout_seconds=int(
                raw.get("request_timeout_seconds", defaults.request_timeout_seconds)
            ),
            execution_timeout_seconds=int(
                raw.get("execution_timeout_seconds", defaults.execution_timeout_seconds)
            ),
        )

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(settings.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
