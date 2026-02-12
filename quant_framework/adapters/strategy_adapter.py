"""策略端口适配器。"""

from __future__ import annotations

from typing import Any, List

from ..core.runtime import EVENT_KIND_SNAPSHOT_ARRIVAL


class LegacyStrategyAdapter:
    """为旧策略补齐 on_event 下的“首快照注入撤单”语义。"""

    def __init__(self, strategy: Any):
        self._strategy = strategy
        self._cancels_injected = False

    def __getattr__(self, item: str) -> Any:
        return getattr(self._strategy, item)

    def on_event(self, e: Any, ctx: Any) -> List[Any]:
        if hasattr(self._strategy, "on_event"):
            actions = list(self._strategy.on_event(e, ctx) or [])
        elif getattr(e, "kind", None) == EVENT_KIND_SNAPSHOT_ARRIVAL and hasattr(self._strategy, "on_snapshot"):
            actions = list(self._strategy.on_snapshot(e.snapshot, ctx.omsView) or [])
        elif getattr(e, "kind", None) == "ReceiptDelivery" and hasattr(self._strategy, "on_receipt"):
            actions = list(self._strategy.on_receipt(e.receipt, ctx.snapshot, ctx.omsView) or [])
        else:
            actions = []

        if (
            not self._cancels_injected
            and getattr(e, "kind", None) == EVENT_KIND_SNAPSHOT_ARRIVAL
            and hasattr(self._strategy, "get_pending_cancels")
        ):
            for sent_time, cancel_req in self._strategy.get_pending_cancels():
                cancel_req.create_time = int(sent_time)
                actions.append(cancel_req)
            self._cancels_injected = True
        return actions
