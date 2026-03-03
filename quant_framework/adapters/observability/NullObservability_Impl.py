"""NullObservability 实现。"""

from __future__ import annotations

from .ReceiptLogger_Impl import (
    ReceiptLogger_Impl,
    _DEFAULT_HISTORY_DIR,
    _DEFAULT_SUBSCRIBER_MEMORY,
)


class NullObservability_Impl(ReceiptLogger_Impl):
    """无输出观测实现（保留诊断与流式能力）。"""

    def __init__(
        self,
        history_dir: str = _DEFAULT_HISTORY_DIR,
        keep_history_files: bool = False,
        default_subscriber_memory_bytes: int = _DEFAULT_SUBSCRIBER_MEMORY,
    ) -> None:
        super().__init__(
            output_file=None,
            verbose=False,
            callback=None,
            history_dir=history_dir,
            keep_history_files=keep_history_files,
            default_subscriber_memory_bytes=default_subscriber_memory_bytes,
        )
