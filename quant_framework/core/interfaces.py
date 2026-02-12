"""接口定义模块。

本模块定义回测系统中的核心抽象接口：
- IQueueModel: 队列模型接口
- IMarketDataFeed: 行情数据源接口
- ISimulationModel: 仿真模型接口
- ITradeTapeReconstructor: 成交带重建接口
- ITapeBuilder: Tape构建器接口
- IExchangeSimulator: 交易所模拟器接口
- IStrategy: 策略接口（使用DTO，支持回执处理）
- IOrderManager: 订单管理器接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Any, TYPE_CHECKING
from .types import Order, NormalizedSnapshot, Price, Qty, Side, TapeSegment, OrderReceipt
from .events import SimulationEvent

if TYPE_CHECKING:
    from .dto import SnapshotDTO, ReadOnlyOMSView


class IQueueModel(ABC):
    """队列模型接口。"""

    @abstractmethod
    def init_order(self, order: Order, level_qty: Qty) -> None:
        """初始化订单在队列中的位置。"""
        pass

    @abstractmethod
    def advance_on_trade(self, order: Order, trade_px: Price, trade_qty: Qty, before: Qty, after: Qty) -> Qty:
        """成交时推进队列位置。"""
        pass

    def advance_on_quote(self, order: Order, before: Qty, after: Qty) -> None:
        """盘口变化时推进队列位置（可选实现）。

        用于处理无成交的盘口变化（主要承载撤单/新增导致的排队位置变化）。
        若实现未覆盖该方法，则默认不更新（保持与旧实现兼容）。
        """
        return


class IMarketDataFeed(ABC):
    """行情数据源接口。"""

    @abstractmethod
    def next(self) -> Optional[NormalizedSnapshot]:
        """获取下一个快照。"""
        pass

    @abstractmethod
    def reset(self):
        """重置数据源（支持多次回测）。"""
        pass


class ISimulationModel(ABC):
    """仿真模型接口：支持随机种子初始化。"""

    @abstractmethod
    def generate_events(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot, context: Any = None) -> Iterator[SimulationEvent]:
        """生成仿真事件。"""
        pass


class ITradeTapeReconstructor(ABC):
    """成交带重建接口。"""

    @abstractmethod
    def reconstruct(self, prev: Optional[NormalizedSnapshot], curr: NormalizedSnapshot) -> List[Tuple[Price, Qty]]:
        """重建成交带。"""
        pass


class ITapeBuilder(ABC):
    """Tape构建器接口（纯函数，无状态）。"""

    @abstractmethod
    def build(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> List[TapeSegment]:
        """从A/B快照构建Tape段。

        Args:
            prev: 前一个快照（A）
            curr: 当前快照（B）

        Returns:
            按时间排序的TapeSegment列表
        """
        pass


class IExchangeSimulator(ABC):
    """交易所模拟器接口（FIFO队列管理）。"""

    @abstractmethod
    def reset(self) -> None:
        """重置模拟器状态（用于新区间）。"""
        pass

    @abstractmethod
    def on_order_arrival(self, order: Order, arrival_time: int, market_qty: Qty) -> Optional[OrderReceipt]:
        """处理订单到达交易所。

        Args:
            order: 到达的订单
            arrival_time: 到达时间
            market_qty: 订单价位的当前市场队列深度

        Returns:
            立即拒绝的回执（None表示已接受）
        """
        pass

    @abstractmethod
    def on_cancel_arrival(self, order_id: str, arrival_time: int) -> OrderReceipt:
        """处理撤单请求到达。

        Args:
            order_id: 要撤销的订单ID
            arrival_time: 撤单到达时间

        Returns:
            撤单操作的回执
        """
        pass

    @abstractmethod
    def advance(self, t_from: int, t_to: int, segment: TapeSegment) -> Tuple[List[OrderReceipt], int]:
        """使用tape段推进仿真从t_from到t_to。

        Args:
            t_from: 切片开始时间
            t_to: 切片结束时间
            segment: 包含该区间M和C的Tape段

        Returns:
            (该时段最早成交回执列表, 停止时间)
        """
        pass

    @abstractmethod
    def align_at_boundary(self, snapshot: NormalizedSnapshot) -> None:
        """在区间边界对齐内部状态。

        Args:
            snapshot: 边界处的快照，用于对齐
        """
        pass

    @abstractmethod
    def get_queue_depth(self, side: Side, price: Price) -> Qty:
        """获取某价位的当前队列深度。

        Args:
            side: 买卖方向
            price: 价格档位

        Returns:
            该档位的队列深度
        """
        pass


@dataclass(frozen=True)
class StepOutcome:
    """执行场所 step 结果。"""

    next_time: int
    receipts_generated: List[OrderReceipt]


class IExecutionVenue(ABC):
    """执行场所端口（由核心事件循环调用）。"""

    @abstractmethod
    def startSession(self) -> None:
        """开启回测会话。"""
        raise NotImplementedError

    @abstractmethod
    def beginInterval(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> None:
        """开始处理区间 [prev.ts_recv, curr.ts_recv]。"""
        raise NotImplementedError

    @abstractmethod
    def onActionArrival(self, action: Any, t_arrive: int) -> List[OrderReceipt]:
        """动作到达执行场所，返回即时生成的回执。"""
        raise NotImplementedError

    @abstractmethod
    def step(self, t_cur: int, t_limit: int) -> StepOutcome:
        """推进执行场所内部时钟到 t_limit（或更早停止点）。"""
        raise NotImplementedError

    @abstractmethod
    def endInterval(self, snapshot_end: NormalizedSnapshot) -> object:
        """结束当前区间，返回统计对象。"""
        raise NotImplementedError


class IOMS(ABC):
    """OMS 端口（核心只依赖该抽象）。"""

    @abstractmethod
    def submit_action(self, action: Any, send_time: int) -> None:
        """登记策略动作（通常是挂单），用于后续回执匹配。"""
        raise NotImplementedError

    @abstractmethod
    def apply_receipt(self, receipt: OrderReceipt) -> None:
        """应用回执，推进订单状态机。"""
        raise NotImplementedError

    @abstractmethod
    def view(self) -> 'ReadOnlyOMSView':
        """返回只读 OMS 视图供策略查询。"""
        raise NotImplementedError


class ITimeModel(ABC):
    """时延模型端口。"""

    @abstractmethod
    def action_arrival_time(self, send_time: int, action: Any) -> int:
        """策略动作到达执行场所的时间。"""
        raise NotImplementedError

    @abstractmethod
    def receipt_delivery_time(self, receipt: OrderReceipt) -> int:
        """回执送达策略的时间。"""
        raise NotImplementedError


class IObservabilitySinks(ABC):
    """可观测性汇聚端口。"""

    @abstractmethod
    def on_receipt_generated(self, receipt: OrderReceipt) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_receipt_delivered(self, receipt: OrderReceipt) -> None:
        raise NotImplementedError

    @abstractmethod
    def on_interval_end(self, stats: object) -> None:
        raise NotImplementedError


class IStrategy(ABC):
    """策略接口（兼容新旧两种回调模式）。

    新模式：
    - on_event(event, strategy_context) -> actions

    旧模式（向后兼容）：
    - on_snapshot(snapshot_dto, oms_view) -> orders
    - on_receipt(receipt, snapshot_dto, oms_view) -> orders
    """

    def on_event(self, e: Any, ctx: Any) -> List[Any]:
        """统一事件入口（默认桥接到旧接口）。"""
        if getattr(e, "kind", None) == "SnapshotArrival" and hasattr(e, "snapshot"):
            return self.on_snapshot(e.snapshot, ctx.omsView)
        if getattr(e, "kind", None) == "ReceiptDelivery" and hasattr(e, "receipt"):
            return self.on_receipt(e.receipt, ctx.snapshot, ctx.omsView)
        raise NotImplementedError("Strategy must implement on_event or legacy callbacks.")

    def on_snapshot(self, snapshot: 'SnapshotDTO', oms_view: 'ReadOnlyOMSView') -> List[Order]:
        """旧接口：快照到达回调（默认桥接到 on_event）。"""
        from .runtime import SnapshotStrategyEvent, StrategyContext

        strategy_ctx = StrategyContext(
            t=int(getattr(snapshot, "ts_recv", 0)),
            snapshot=snapshot,
            omsView=oms_view,
        )
        return self.on_event(SnapshotStrategyEvent(time=strategy_ctx.t, snapshot=snapshot), strategy_ctx)

    def on_receipt(self, receipt: OrderReceipt, snapshot: 'SnapshotDTO', oms_view: 'ReadOnlyOMSView') -> List[Order]:
        """旧接口：回执到达回调（默认桥接到 on_event）。"""
        from .runtime import ReceiptStrategyEvent, StrategyContext

        recv_time = int(receipt.recv_time) if receipt.recv_time is not None else int(receipt.timestamp)
        strategy_ctx = StrategyContext(
            t=recv_time,
            snapshot=snapshot,
            omsView=oms_view,
        )
        return self.on_event(ReceiptStrategyEvent(time=recv_time, receipt=receipt), strategy_ctx)


class IOrderManager(ABC):
    """订单管理器接口。

    提供订单管理功能，EventLoop使用此接口来管理订单。
    """

    @abstractmethod
    def get_active_orders(self) -> List[Order]:
        """获取所有活跃订单。

        Returns:
            活跃订单列表
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[Order]:
        """根据ID获取订单。

        Args:
            order_id: 订单ID

        Returns:
            订单（如果存在），否则返回None
        """
        pass

    @abstractmethod
    def submit(self, order: Order, submit_time: int) -> None:
        """提交新订单。

        Args:
            order: 要提交的订单
            submit_time: 提交时间
        """
        pass

    @abstractmethod
    def on_receipt(self, receipt: OrderReceipt) -> None:
        """处理订单回执。

        Args:
            receipt: 要处理的订单回执
        """
        pass

    @abstractmethod
    def get_pending_orders(self) -> List[Order]:
        """获取所有待到达的订单（arrival_time is None）。

        Returns:
            待到达订单列表
        """
        pass

    @abstractmethod
    def get_arrived_orders(self) -> List[Order]:
        """获取所有已到达的订单（arrival_time is not None）。

        Returns:
            已到达订单列表
        """
        pass

    @abstractmethod
    def mark_order_arrived(self, order_id: str, arrival_time: int) -> None:
        """标记订单已到达交易所。

        Args:
            order_id: 订单ID
            arrival_time: 到达时间
        """
        pass
