"""接口定义模块。

本模块定义回测系统中的核心抽象接口：
- IQueueModel: 队列模型接口
- IMarketDataFeed: 行情数据源接口
- ISimulationModel: 仿真模型接口
- ITradeTapeReconstructor: 成交带重建接口
- IStrategy: 旧版策略接口
- ITapeBuilder: Tape构建器接口
- IExchangeSimulator: 交易所模拟器接口
- IStrategyNew: 新版策略接口（支持回执处理）
- IStrategyDTO: 使用DTO的策略接口（完全解耦，只读视图）
- IOrderManager: 订单管理器接口
- IReadOnlyOrderManager: 只读订单管理器接口（供策略使用）
"""

from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Any
from .types import Order, NormalizedSnapshot, Price, Qty, Side, TapeSegment, OrderReceipt
from .events import SimulationEvent


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


class IStrategy(ABC):
    """旧版策略接口。"""

    @abstractmethod
    def on_market_tick(self, book: Any, oms: Any) -> List[Order]:
        """行情tick回调。"""
        pass


# ============================================================================
# 新版统一架构接口
# ============================================================================

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
    def advance(self, t_from: int, t_to: int, segment: TapeSegment) -> List[OrderReceipt]:
        """使用tape段推进仿真从t_from到t_to。

        Args:
            t_from: 切片开始时间
            t_to: 切片结束时间
            segment: 包含该区间M和C的Tape段

        Returns:
            该时段内的成交回执列表
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


class IReadOnlyOrderManager(ABC):
    """只读订单管理器接口（供策略使用）。
    
    该接口只提供查询方法，不提供修改方法。
    策略通过此接口查询订单状态，但不能直接操作OMS。
    策略需要提交订单时，应通过返回Order列表的方式，
    由EventLoop负责调用OMS的submit方法。
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


class IStrategyNew(ABC):
    """新版策略接口（必须处理回执）。
    
    注意：虽然此接口传入oms参数，但策略应只使用其查询方法。
    推荐使用IStrategyDTO接口，它通过只读视图强制这一约束。
    """

    @abstractmethod
    def on_snapshot(self, snapshot: NormalizedSnapshot, oms: 'IOrderManager') -> List[Order]:
        """快照到达时回调。

        Args:
            snapshot: 新的行情快照
            oms: 订单管理器（用于查询订单状态）

        Returns:
            要提交的新订单列表
        """
        pass

    @abstractmethod
    def on_receipt(self, receipt: OrderReceipt, snapshot: NormalizedSnapshot, oms: 'IOrderManager') -> List[Order]:
        """订单回执到达时回调。

        Args:
            receipt: 订单回执（成交、撤单等）
            snapshot: 当前行情快照
            oms: 订单管理器（用于查询订单状态）

        Returns:
            要提交的新订单列表
        """
        pass


class IStrategyDTO(ABC):
    """使用DTO的策略接口（完全解耦，只读视图）。

    该接口使用只读视图和DTO，实现策略与系统其他组件的完全解耦：
    - 使用SnapshotDTO代替NormalizedSnapshot，确保策略无法修改原始数据
    - 使用ReadOnlyOMSView代替IOrderManager，策略只能查询不能直接操作OMS

    设计理念：
    - 策略的输入是只读的（SnapshotDTO, ReadOnlyOMSView）
    - 策略的输出是Order列表，由EventLoop负责提交到OMS
    - 策略不能直接调用OMS的submit/on_receipt等修改方法

    这种设计的优点：
    1. 类型安全：编译时即可发现策略尝试修改只读数据的错误
    2. 清晰边界：明确了策略的输入（只读）和输出（Order列表）
    3. 易于测试：可以轻松构造DTO进行单元测试
    4. 避免副作用：策略无法直接操作OMS，减少了潜在的bug
    """

    @abstractmethod
    def on_snapshot(self, snapshot: 'SnapshotDTO', oms_view: 'ReadOnlyOMSView') -> List[Order]:
        """快照到达时回调（使用只读视图）。

        Args:
            snapshot: 行情快照DTO（不可变）
            oms_view: OMS只读视图（只能查询，不能操作）

        Returns:
            要提交的新订单列表
        """
        pass

    @abstractmethod
    def on_receipt(self, receipt: OrderReceipt, snapshot: 'SnapshotDTO', oms_view: 'ReadOnlyOMSView') -> List[Order]:
        """订单回执到达时回调（使用只读视图）。

        Args:
            receipt: 订单回执（成交、撤单等）
            snapshot: 当前行情快照DTO（不可变）
            oms_view: OMS只读视图（只能查询，不能操作）

        Returns:
            要提交的新订单列表
        """
        pass


class IOrderManager(IReadOnlyOrderManager):
    """订单管理器接口（新架构）。
    
    继承自IReadOnlyOrderManager，增加了修改方法。
    EventLoop使用此接口来管理订单。
    """

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
