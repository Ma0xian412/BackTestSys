"""运行控制对象：支持优雅中断请求。"""

from __future__ import annotations

import threading
from typing import Optional


DEFAULT_INTERRUPT_REASON = "external_request"


class RunControl:
    """线程安全的 run 停止控制器。"""

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._stop_reason: Optional[str] = None
        self._lock = threading.Lock()

    def request_stop(self, reason: str = DEFAULT_INTERRUPT_REASON) -> None:
        normalized_reason = reason.strip() if isinstance(reason, str) else ""
        if not normalized_reason:
            normalized_reason = DEFAULT_INTERRUPT_REASON
        with self._lock:
            if self._stop_event.is_set():
                return
            self._stop_reason = normalized_reason
            self._stop_event.set()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    @property
    def stop_reason(self) -> Optional[str]:
        return self._stop_reason
