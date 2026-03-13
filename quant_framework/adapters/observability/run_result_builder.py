"""回测最终结果汇总器。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from ...core.data_structure import OrderStatus
from ...core.run_result import (
    BacktestRunResult,
    CancelRequestRecord,
    DoneInfo,
    ExecutionDetail,
    OrderInfo,
    RunResultMetadata,
)

_BUY_DIRECTION = "Buy"
_SELL_DIRECTION = "Sell"
_FILLED_RECEIPT_TYPES = {"FILL", "PARTIAL"}


def _to_output_order_id(order_id: object) -> int:
    try:
        return int(str(order_id))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"order_id must be convertible to int, got {order_id!r}") from exc


def _to_output_direction(side: object) -> str:
    value = str(side).strip().upper()
    if value == "BUY":
        return _BUY_DIRECTION
    if value == "SELL":
        return _SELL_DIRECTION
    raise ValueError(f"unsupported order direction: {side!r}")


def _to_output_contract_id(contract_id: object) -> str:
    value = str(contract_id).strip()
    if value == "0":
        return ""
    return value


def _trade_state(order: Any) -> str:
    qty = int(getattr(order, "qty", 0))
    filled_qty = int(getattr(order, "filled_qty", 0))
    status = getattr(order, "status", None)
    if qty > 0 and (
        filled_qty >= qty
        or status == OrderStatus.FILLED
        or getattr(status, "value", "") == OrderStatus.FILLED.value
    ):
        return "A"
    if filled_qty > 0:
        return "P"
    return "N"


class RunResultBuilder:
    """根据观测事件构造最终回测结果。"""

    def __init__(self, metadata: Optional[RunResultMetadata] = None) -> None:
        self._metadata = metadata or RunResultMetadata()
        self._contract_id = _to_output_contract_id(self._metadata.contract_id)
        self.reset()

    def reset(self) -> None:
        self._orders: list[OrderInfo] = []
        self._cancels: list[CancelRequestRecord] = []
        self._executions: list[ExecutionDetail] = []
        self._last_recv_time_by_order: dict[str, int] = {}
        self._submitted_order_ids: list[str] = []

    def record_order_submitted(self, event_time: int, payload: Mapping[str, object]) -> None:
        order_id_raw = str(payload["order_id"])
        self._submitted_order_ids.append(order_id_raw)
        self._orders.append(
            OrderInfo(
                PartitionDay=self._metadata.partition_day,
                ContractId=self._contract_id,
                OrderId=_to_output_order_id(order_id_raw),
                LimitPrice=float(payload["price"]),
                Volume=int(payload["qty"]),
                OrderDirection=_to_output_direction(payload["side"]),
                SentTime=int(event_time),
                MachineName=self._metadata.machine_name,
            )
        )

    def record_cancel_submitted(self, event_time: int, payload: Mapping[str, object]) -> None:
        self._cancels.append(
            CancelRequestRecord(
                PartitionDay=self._metadata.partition_day,
                ContractId=self._contract_id,
                OrderId=_to_output_order_id(payload["order_id"]),
                CancelSentTime=int(payload.get("create_time", event_time)),
                MachineName=self._metadata.machine_name,
            )
        )

    def record_receipt_delivered(self, payload: Mapping[str, object], order_lookup: Mapping[str, Any]) -> None:
        order_id_raw = str(payload["order_id"])
        recv_tick = int(payload.get("recv_time") or payload["timestamp"])
        self._last_recv_time_by_order[order_id_raw] = recv_tick

        receipt_type = str(payload["receipt_type"])
        if receipt_type not in _FILLED_RECEIPT_TYPES:
            return

        order = order_lookup.get(order_id_raw)
        if order is None:
            raise ValueError(f"missing order for receipt order_id={order_id_raw!r}")
        self._executions.append(
            ExecutionDetail(
                PartitionDay=self._metadata.partition_day,
                RecvTick=recv_tick,
                ExchTick=int(payload["timestamp"]),
                OrderId=_to_output_order_id(order_id_raw),
                ContractId=self._contract_id,
                Price=float(payload["fill_price"]),
                Volume=int(payload["fill_qty"]),
                OrderDirection=_to_output_direction(getattr(order.side, "value", order.side)),
                MachineName=self._metadata.machine_name,
            )
        )

    def build(self, oms: Optional[Any], final_time: int) -> BacktestRunResult:
        orders = getattr(oms, "orders", {}) if oms is not None else {}
        done_info = self._build_done_info(orders, final_time)
        return BacktestRunResult(
            DoneInfo=tuple(done_info),
            ExecutionDetail=tuple(self._executions),
            OrderInfo=tuple(self._orders),
            CancelRequest=tuple(self._cancels),
        )

    def _build_done_info(self, orders: Mapping[str, Any], final_time: int) -> list[DoneInfo]:
        seen: set[str] = set()
        output: list[DoneInfo] = []

        for order_id_raw in self._submitted_order_ids:
            if order_id_raw in seen:
                continue
            seen.add(order_id_raw)
            order = orders.get(order_id_raw)
            if order is None:
                continue
            output.append(self._make_done_info(order_id_raw, order, final_time))

        for order_id_raw, order in orders.items():
            if order_id_raw in seen:
                continue
            output.append(self._make_done_info(str(order_id_raw), order, final_time))
        return output

    def _make_done_info(self, order_id_raw: str, order: Any, final_time: int) -> DoneInfo:
        done_time = self._last_recv_time_by_order.get(order_id_raw, int(final_time))
        return DoneInfo(
            PartitionDay=self._metadata.partition_day,
            ContractId=self._contract_id,
            OrderId=_to_output_order_id(order_id_raw),
            DoneTime=int(done_time),
            OrderTradeState=_trade_state(order),
            MachineName=self._metadata.machine_name,
        )
