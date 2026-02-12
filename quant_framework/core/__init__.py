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
    IMarketDataFeed,
    ITapeBuilder,
    IExecutionVenue, IOMS, ITimeModel, IObservabilitySinks, StepOutcome,
    IStrategy,
)

from .dto import (
    LevelDTO, SnapshotDTO, OrderInfoDTO, PortfolioDTO,
    ReadOnlyOMSView, to_snapshot_dto,
)

from .runtime import (
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    EVENT_KIND_ACTION_ARRIVAL,
    EVENT_KIND_RECEIPT_DELIVERY,
    Event,
    StrategyContext,
    EventSpecRegistry,
    RuntimeContext,
)
from .actions import Action, PlaceOrderAction, CancelOrderAction
from .scheduler import HeapScheduler
from .dispatcher import Dispatcher, IEventHandler
from .handlers import SnapshotArrivalHandler, ActionArrivalHandler, ReceiptDeliveryHandler
from .kernel import EventLoopKernel
from .app import RuntimeBuildConfig, CompositionRoot, BacktestApp

from .data_loader import (
    CsvMarketDataFeed,
    PickleMarketDataFeed,
    SnapshotDuplicatingFeed,
)
