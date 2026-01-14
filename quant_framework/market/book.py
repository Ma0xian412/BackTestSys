from typing import Optional
from ..core.types import NormalizedSnapshot, Price, Qty, Side, Level

class BookView:
    def __init__(self):
        self.prev: Optional[NormalizedSnapshot] = None
        self.cur: Optional[NormalizedSnapshot] = None

    def apply_snapshot(self, snapshot: NormalizedSnapshot, synthetic: bool = False):
        self.prev = self.cur
        self.cur = snapshot

    def get_best_price(self, side: Side) -> Optional[Price]:
        if not self.cur: return None
        levels = self.cur.bids if side == Side.BUY else self.cur.asks
        return levels[0].price if levels else None

    def find_qty_at(self, snapshot: Optional[NormalizedSnapshot], price: Price, side: Side) -> Qty:
        if not snapshot: return 0
        levels = snapshot.bids if side == Side.BUY else snapshot.asks
        for lvl in levels:
            if abs(lvl.price - price) < 1e-8: return lvl.qty
        return 0