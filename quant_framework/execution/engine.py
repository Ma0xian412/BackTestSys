import uuid
from typing import List
from ..core.types import Order, Fill, OrderStatus
from ..core.events import SimulationEvent, EventType
from ..market.book import BookView
from .policies import MakerPolicy, TakerPolicy

class ReactiveExecutionEngine:
    def __init__(self, maker: MakerPolicy, taker: TakerPolicy):
        self.maker = maker
        self.taker = taker
        self.local_book = BookView()
        self.active_orders: List[Order] = []
        self.ts = 0

    def process_event(self, event: SimulationEvent) -> List[Fill]:
        self.ts = event.ts
        fills = []

        if event.type in [EventType.SNAPSHOT_ARRIVAL, EventType.QUOTE_UPDATE]:
            self.local_book.apply_snapshot(event.data, synthetic=(event.type==EventType.QUOTE_UPDATE))
            # 盘口变化（撤单/新增）也会影响 maker 排队位置
            for o in list(self.active_orders):
                if o.is_active:
                    self.maker.process_quote_update(o, self.local_book)
            fills.extend(self._match_taker())
            
        elif event.type == EventType.TRADE_TICK:
            # 兼容两种 payload:
            # 1) (px, qty)
            # 2) {"px":..., "qty":..., "book_after": NormalizedSnapshot}
            px = qty = None
            book_after = None
            if isinstance(event.data, dict):
                px = event.data.get('px')
                qty = event.data.get('qty')
                book_after = event.data.get('book_after')
            else:
                try:
                    px, qty = event.data
                except Exception:
                    pass

            if book_after is not None:
                # 将“成交后的盘口”作为 synthetic 更新写入 book，便于 before/after 推断
                self.local_book.apply_snapshot(book_after, synthetic=True)

            if px is not None and qty is not None:
                fills.extend(self._match_maker(px, qty))

            # 成交后盘口已更新，可能触发 taker 立即成交
            fills.extend(self._match_taker())
            
        return fills

    def register_orders(self, orders: List[Order]):
        for o in orders:
            if o not in self.active_orders:
                self.active_orders.append(o)
                if self.local_book.cur:
                    self.maker.on_new_order(o, self.local_book)

    def _match_maker(self, px, qty) -> List[Fill]:
        fills = []
        if not self.local_book.cur: return fills
        for o in list(self.active_orders):
            if not o.is_active: 
                self.active_orders.remove(o)
                continue
                
            f_qty = self.maker.process_trade_tick(o, px, qty, self.local_book)
            if f_qty > 0:
                fills.append(self._make_fill(o, px, f_qty, "MAKER"))
                if o.remaining_qty == 0: 
                    o.status = OrderStatus.FILLED
        return fills

    def _match_taker(self) -> List[Fill]:
        fills = []
        if not self.local_book.cur: return fills
        for o in list(self.active_orders):
            if not o.is_active: continue
            
            res = self.taker.match(o, self.local_book)
            for px, q in res:
                fills.append(self._make_fill(o, px, q, "TAKER"))
                
            if o.remaining_qty == 0: 
                o.status = OrderStatus.FILLED
            elif o.filled_qty > 0:
                o.status = OrderStatus.PARTIALLY_FILLED
        return fills

    def _make_fill(self, o: Order, px, qty, liq) -> Fill:
        return Fill(str(uuid.uuid4()), o.order_id, o.side, px, qty, self.ts, liq)