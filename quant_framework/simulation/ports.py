from __future__ import annotations

"""Simulation layer ports (role interfaces).

These Protocols define *what* simulation models may need for an interval, without
coupling the core simulation logic to the runner / execution-engine details.

Design notes:
- Interface Segregation Principle (ISP): avoid "fat" context objects; keep role
  interfaces small and focused.
- Ports & Adapters (Hexagonal) / Clean Architecture: the runner supplies adapter
  objects that implement these ports.

In Python we use `typing.Protocol` as a lightweight way to express these ports.
"""

from typing import Protocol, Sequence, Tuple

from ..core.types import Order, NormalizedSnapshot


class IOrderIntervalView(Protocol):
    """Read-only view of orders relevant to an interval (t0, t1]."""

    t0: int
    t1: int

    # Orders already present at the exchange
    active_orders: Sequence[Order]

    # Orders not yet at the exchange: (due_ts, order)
    pending_orders: Sequence[Tuple[int, Order]]

    def due_pending_orders(self) -> Sequence[Order]:
        """Pending orders whose due time falls inside (t0, t1]."""
        ...

    @property
    def needs_micro_sim(self) -> bool:
        """Whether a micro simulation between endpoints can affect results."""
        ...

    def relevant_orders(self) -> Sequence[Order]:
        """Active orders + due pending orders (remaining_qty > 0)."""
        ...


class IMarketIntervalView(Protocol):
    """Read-only view of the market interval endpoints."""

    t0: int
    t1: int

    prev: NormalizedSnapshot
    curr: NormalizedSnapshot
