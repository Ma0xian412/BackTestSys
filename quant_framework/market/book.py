"""订单簿视图模块。

提供市场数据的视图封装。
"""

from typing import Optional
from ..core.types import NormalizedSnapshot, Price, Qty, Side, Level


class BookView:
    """订单簿视图。

    封装当前和前一个快照，提供便捷的价格查询方法。

    Attributes:
        prev: 前一个快照
        cur: 当前快照
    """

    def __init__(self):
        """初始化订单簿视图。"""
        self.prev: Optional[NormalizedSnapshot] = None
        self.cur: Optional[NormalizedSnapshot] = None

    def apply_snapshot(self, snapshot: NormalizedSnapshot, synthetic: bool = False):
        """应用新的快照。

        Args:
            snapshot: 新的快照数据
            synthetic: 是否为合成快照（默认False）
        """
        self.prev = self.cur
        self.cur = snapshot

    def get_best_price(self, side: Side) -> Optional[Price]:
        """获取最优价格。

        Args:
            side: 买卖方向

        Returns:
            最优价格，如果无数据则返回None
        """
        if not self.cur:
            return None
        levels = self.cur.bids if side == Side.BUY else self.cur.asks
        return levels[0].price if levels else None

    def find_qty_at(self, snapshot: Optional[NormalizedSnapshot], price: Price, side: Side) -> Qty:
        """查询指定价位的数量。

        Args:
            snapshot: 快照数据
            price: 价格
            side: 买卖方向

        Returns:
            该价位的数量，如果不存在则返回0
        """
        if not snapshot:
            return 0
        levels = snapshot.bids if side == Side.BUY else snapshot.asks
        for lvl in levels:
            if abs(lvl.price - price) < 1e-8:
                return lvl.qty
        return 0