from typing import List
from ..core.interfaces import IStrategy
from ..core.types import Order, Side
from ..market.book import BookView

class LadderTestStrategy(IStrategy):
    def __init__(self, name="Ladder"):
        self.name = name
        self.count = 0

    def on_market_tick(self, book: BookView, oms) -> List[Order]:
        if not book.cur: return []
        self.count += 1
        if self.count % 50 != 0: return []
        
        # 简单示例：只挂单不撤单
        bid = book.get_best_price(Side.BUY)
        return [Order(f"{self.name}-{self.count}", Side.BUY, bid, 1)] if bid else []