from __future__ import annotations

import traceback
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal


class TaskThread(QThread):
    progress = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, task: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self._task = task
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._task(*self._args, progress=self.progress.emit, **self._kwargs)
        except Exception as exc:  # pragma: no cover - UI threading path
            detail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(detail)
            return
        self.succeeded.emit(result)
