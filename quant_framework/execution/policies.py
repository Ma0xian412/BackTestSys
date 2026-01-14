from typing import List, Tuple
from ..core.types import Order, Price, Qty, Side
from ..market.book import BookView
from ..core.interfaces import IQueueModel

class TakerPolicy:
    def match(self, order: Order, book: BookView) -> List[Tuple[Price, Qty]]:
        fills = []
        if not book.cur: return fills
        
        remaining = order.remaining_qty
        levels = book.cur.asks if order.side == Side.BUY else book.cur.bids
        
        for lvl in levels:
            if remaining <= 0: break
            is_match = (order.price >= lvl.price) if order.side == Side.BUY else (order.price <= lvl.price)
            if is_match:
                exec_qty = min(remaining, lvl.qty)
                fills.append((lvl.price, exec_qty))
                remaining -= exec_qty
                order.filled_qty += exec_qty
            else:
                break
        return fills

class MakerPolicy:
    def __init__(self, queue_model: IQueueModel):
        self.queue_model = queue_model

    def on_new_order(self, order: Order, book: BookView):
        qty = book.find_qty_at(book.cur, order.price, order.side)
        self.queue_model.init_order(order, qty)

    def process_trade_tick(self, order: Order, px: Price, qty: Qty, book: BookView) -> Qty:
        q_before = book.find_qty_at(book.prev, px, order.side)
        q_after = book.find_qty_at(book.cur, px, order.side)
        return self.queue_model.advance_on_trade(order, px, qty, q_before, q_after)

    def process_quote_update(self, order: Order, book: BookView) -> None:
        """处理无成交的盘口更新，用于更新排队位置（撤单/新增）。"""
        if not book.cur or not book.prev:
            return
        q_before = book.find_qty_at(book.prev, order.price, order.side)
        q_after = book.find_qty_at(book.cur, order.price, order.side)
        self.queue_model.advance_on_quote(order, q_before, q_after)