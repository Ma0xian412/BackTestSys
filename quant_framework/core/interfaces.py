from abc import ABC, abstractmethod
from typing import Iterator, List, Optional, Tuple, Any
from .types import Order, NormalizedSnapshot, Price, Qty
from .events import SimulationEvent

class IQueueModel(ABC):
    @abstractmethod
    def init_order(self, order: Order, level_qty: Qty) -> None: pass
    @abstractmethod
    def advance_on_trade(self, order: Order, trade_px: Price, trade_qty: Qty, before: Qty, after: Qty) -> Qty: pass

    # 可选：用于处理“无成交的盘口变化”（主要承载撤单/新增导致的排队位置变化）
    # 若实现未覆盖该方法，则默认不更新（保持与旧实现兼容）。
    def advance_on_quote(self, order: Order, before: Qty, after: Qty) -> None:
        return

class IMarketDataFeed(ABC):
    @abstractmethod
    def next(self) -> Optional[NormalizedSnapshot]: pass
    @abstractmethod
    def reset(self): pass # 支持多次回测重置

class ISimulationModel(ABC):
    """仿真模型接口：支持随机种子初始化"""
    @abstractmethod
    def generate_events(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot, context: Any = None) -> Iterator[SimulationEvent]: pass

class ITradeTapeReconstructor(ABC):
    @abstractmethod
    def reconstruct(self, prev: Optional[NormalizedSnapshot], curr: NormalizedSnapshot) -> List[Tuple[Price, Qty]]: pass

class IStrategy(ABC):
    @abstractmethod
    def on_market_tick(self, book: Any, oms: Any) -> List[Order]: pass