"""核心模块。

导出核心类型、接口和DTO。
"""

from .types import (
    Price, Qty, OrderId, Timestamp,
    Side, OrderStatus, TimeInForce, ReceiptType,
    Level, NormalizedSnapshot, Order, Fill,
    TapeSegment, OrderReceipt, FillDetail, OrderDiagnostics,
)

from .interfaces import (
    IQueueModel, IMarketDataFeed, ISimulationModel,
    ITradeTapeReconstructor, IStrategy,
    ITapeBuilder, IExchangeSimulator,
    IStrategyNew, IStrategyDTO,
    IReadOnlyOrderManager, IOrderManager,
)

from .dto import (
    LevelDTO, SnapshotDTO, OrderInfoDTO, PortfolioDTO,
    ReadOnlyOMSView, to_snapshot_dto,
)

from .events import EventType, SimulationEvent
