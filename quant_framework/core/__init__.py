"""核心模块导出。"""

from .types import (
    Price, Qty, OrderId, Timestamp,
    TICK_PER_MS, SNAPSHOT_MIN_INTERVAL_TICK, DEFAULT_SNAPSHOT_TOLERANCE_TICK,
    Side, OrderStatus, TimeInForce, RequestType, ReceiptType,
    Level, NormalizedSnapshot, Order, CancelRequest, Fill,
    TapeSegment, OrderReceipt, FillDetail, OrderDiagnostics,
)

from .port import (
    IMarketDataFeed,
    ITapeBuilder,
    IExecutionVenue, IOMS, ITimeModel, IObservabilitySinks, StepOutcome,
    IStrategy,
)

from .read_only_view import (
    OrderSnapshot, PortfolioSnapshot,
    ReadOnlyOMSView,
)

from .model import (
    EVENT_KIND_SNAPSHOT_ARRIVAL,
    EVENT_KIND_ACTION_ARRIVAL,
    EVENT_KIND_RECEIPT_DELIVERY,
    ActionType,
    Action,
    Event,
    StrategyContext,
    EventSpecRegistry,
    RuntimeContext,
    reset_event_seq,
)
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
