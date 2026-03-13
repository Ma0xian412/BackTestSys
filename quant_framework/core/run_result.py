"""回测最终结果数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class RunResultMetadata:
    """回测结果公共元数据。"""

    partition_day: int = 0
    contract_id: str = ""
    machine_name: str = ""


@dataclass(frozen=True)
class DoneInfo:
    PartitionDay: int
    ContractId: str
    OrderId: int
    DoneTime: int
    OrderTradeState: str
    MachineName: str


@dataclass(frozen=True)
class ExecutionDetail:
    PartitionDay: int
    RecvTick: int
    ExchTick: int
    OrderId: int
    ContractId: str
    Price: float
    Volume: int
    OrderDirection: str
    MachineName: str


@dataclass(frozen=True)
class OrderInfo:
    PartitionDay: int
    ContractId: str
    OrderId: int
    LimitPrice: float
    Volume: int
    OrderDirection: str
    SentTime: int
    MachineName: str


@dataclass(frozen=True)
class CancelRequestRecord:
    PartitionDay: int
    ContractId: str
    OrderId: int
    CancelSentTime: int
    MachineName: str


@dataclass(frozen=True)
class BacktestRunResult:
    DoneInfo: Tuple[DoneInfo, ...] = field(default_factory=tuple)
    ExecutionDetail: Tuple[ExecutionDetail, ...] = field(default_factory=tuple)
    OrderInfo: Tuple[OrderInfo, ...] = field(default_factory=tuple)
    CancelRequest: Tuple[CancelRequestRecord, ...] = field(default_factory=tuple)
