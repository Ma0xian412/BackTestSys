"""核心模块。

导出核心类型、接口和DTO。
"""

from .types import (
    Price, Qty, OrderId, Timestamp,
    TICK_PER_MS, SNAPSHOT_MIN_INTERVAL_TICK, DEFAULT_SNAPSHOT_TOLERANCE_TICK,
    Side, OrderStatus, TimeInForce, RequestType, ReceiptType,
    Level, NormalizedSnapshot, Order, CancelRequest, Fill,
    TapeSegment, OrderReceipt, FillDetail, OrderDiagnostics,
)

from .interfaces import (
    IQueueModel, IMarketDataFeed, ISimulationModel,
    ITradeTapeReconstructor,
    ITapeBuilder, IExchangeSimulator,
    IStrategy, IOrderManager,
)

from .dto import (
    LevelDTO, SnapshotDTO, OrderInfoDTO, PortfolioDTO,
    ReadOnlyOMSView, to_snapshot_dto,
)

from .events import EventType, SimulationEvent

from .data_loader import (
    CsvMarketDataFeed,
    PickleMarketDataFeed,
    SnapshotDuplicatingFeed,
)
