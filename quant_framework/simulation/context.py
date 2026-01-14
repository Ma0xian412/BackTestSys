from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional

from ..core.types import Order, NormalizedSnapshot
from .ports import IOrderIntervalView, IMarketIntervalView


@dataclass(frozen=True)
class OrderIntervalView(IOrderIntervalView):
    """Order-focused port for an interval (t0, t1].

    Kept small on purpose: simulation models should not depend on runner/execution details.
    """

    t0: int
    t1: int

    active_orders: Sequence[Order]
    pending_orders: Sequence[Tuple[int, Order]]

    def due_pending_orders(self) -> List[Order]:
        """Pending orders whose due time falls inside (t0, t1]."""
        return [
            o
            for due_ts, o in self.pending_orders
            if self.t0 < int(due_ts) <= self.t1 and getattr(o, "remaining_qty", 0) > 0
        ]

    def relevant_orders(self) -> List[Order]:
        """Active orders + due pending orders (remaining_qty > 0)."""
        active = [
            o for o in self.active_orders
            if getattr(o, "is_active", False) and getattr(o, "remaining_qty", 0) > 0
        ]
        return active + self.due_pending_orders()

    @property
    def needs_micro_sim(self) -> bool:
        """Whether a micro simulation between endpoints can affect results."""
        return bool(self.relevant_orders())


@dataclass(frozen=True)
class MarketIntervalView(IMarketIntervalView):
    """Market-focused port for an interval (t0, t1]."""

    t0: int
    t1: int

    prev: NormalizedSnapshot
    curr: NormalizedSnapshot

    # Optional: allow the runner/config layer to pass the instrument tick size.
    # If omitted, simulation models may fall back to inferring tick from endpoints.
    tick_size: Optional[float] = None


@dataclass(frozen=True)
class IntervalContext:
    """Composite context made of two small ports: orders + market.

    This object is still convenient to pass around as `context=...`, but keeps
    responsibilities separated. It also provides backward-compatible attribute
    delegates so older models that expect `context.t0`, `context.active_orders`,
    etc. can still work.
    """

    orders: IOrderIntervalView
    market: IMarketIntervalView

    # --- Backward-compatible delegates ---
    @property
    def t0(self) -> int:
        return int(self.orders.t0)

    @property
    def t1(self) -> int:
        return int(self.orders.t1)

    @property
    def active_orders(self) -> Sequence[Order]:
        return self.orders.active_orders

    @property
    def pending_orders(self) -> Sequence[Tuple[int, Order]]:
        return self.orders.pending_orders

    def due_pending_orders(self) -> Sequence[Order]:
        return list(self.orders.due_pending_orders())

    @property
    def needs_micro_sim(self) -> bool:
        return bool(self.orders.needs_micro_sim)
