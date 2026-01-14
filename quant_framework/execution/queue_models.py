import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..core.interfaces import IQueueModel
from ..core.types import Order, Price, Qty, Side


@dataclass
class _LevelState:
    """维护同一 (side, price) 上【自有订单序列】的 FIFO 一致性。

    我们只有 L2（聚合）数量，因此无法知道撤单发生在队列的精确位置。
    这里采取一个工程化折中：

    - 对【新增】：默认全部加在队尾（不影响你前方量）。
    - 对【减少】（撤单/其它非你单成交造成的减少）：按一个单调 CDF F(u) 把减少分配到
      外部队列的不同区间（越靠后撤单概率越大/越小由 k 控制）。
    - 对【成交】：按 FIFO 在“外部队列 + 你的订单”构成的联合队列上推进，保证同价位
      多笔挂单不会出现“后单跑到前单前面”的顺序破坏。

    重要：订单簿快照不包含你的订单（无冲击假设），因此我们把外部队列（来自快照的 qty）
    与自有订单（Order.remaining_qty）分开存储。
    """

    side: Side
    price: Price
    k: float

    # 外部队列的“分段”表示（都不包含你的订单）
    market_ahead: int = 0                 # 在你最早一笔订单之前的外部量
    gaps: List[int] = field(default_factory=list)  # 你的订单之间夹着的外部量（len = n_orders-1）
    tail: int = 0                         # 在你最后一笔订单之后的外部量

    # 自有订单 FIFO 列表（按到达交易所顺序）
    orders: List[Order] = field(default_factory=list)

    # 为了在同一个事件里被多次调用（每个 order 调一次）而不重复处理：
    quote_sig: Optional[Tuple[int, int]] = None
    quote_served: set = field(default_factory=set)

    trade_sig: Optional[Tuple[float, int, int, int]] = None
    trade_fill_map: Dict[str, int] = field(default_factory=dict)
    trade_served: set = field(default_factory=set)
    trade_expected: set = field(default_factory=set)

    def _alpha(self) -> float:
        return math.exp(self.k)

    def _F(self, u: float) -> float:
        """CDF on [0,1]. 这里沿用旧实现的 x**alpha 形式。"""
        u = max(0.0, min(1.0, u))
        return u ** self._alpha()

    # ---------- housekeeping ----------

    def _external_total(self) -> int:
        return int(self.market_ahead + sum(self.gaps) + self.tail)

    def _clean_orders(self) -> None:
        """移除已无剩余量的订单，并维护 gaps/market_ahead/tail 的一致性。"""
        i = 0
        while i < len(self.orders):
            o = self.orders[i]
            if o.remaining_qty > 0 and o.is_active:
                i += 1
                continue

            # 移除 orders[i]
            if len(self.orders) == 1:
                # 没有自有单了：把所有外部量收敛到 market_ahead
                self.market_ahead = self._external_total()
                self.gaps = []
                self.tail = 0
                self.orders = []
                break

            if i == 0:
                # 第一笔被移除：gap0 变成新的 market_ahead 的一部分
                if self.gaps:
                    self.market_ahead += self.gaps[0]
                    self.gaps.pop(0)
                self.orders.pop(0)
                # i 不变（新首单来到 i=0）
                continue

            if i == len(self.orders) - 1:
                # 最后一笔被移除：最后一个 gap 变成 tail
                if self.gaps:
                    self.tail += self.gaps[-1]
                    self.gaps.pop(-1)
                self.orders.pop(-1)
                break  # 最后一个删了，结束
            else:
                # 中间的订单被移除：合并两侧 gap
                # gaps[i-1] 在 orders[i-1] 与 orders[i] 之间
                # gaps[i]   在 orders[i]   与 orders[i+1] 之间
                self.gaps[i - 1] += self.gaps[i]
                self.gaps.pop(i)
                self.orders.pop(i)
                # i 不变（原 orders[i+1] 移到 i）
                continue

    def _reconcile_external_to(self, target_ext: int) -> None:
        """把外部段总和强制对齐到 target_ext（快照观测值）。

        只在内部做一个“最不伤害排队位置”的调整：
        - 先动 tail（对前方量影响最小）
        - 不够再动 gaps（从最靠后开始）
        - 最后才动 market_ahead
        """
        target_ext = max(0, int(target_ext))
        cur = self._external_total()
        diff = target_ext - cur
        if diff == 0:
            return
        if diff > 0:
            self.tail += diff
            return

        # diff < 0: 需要移除 -diff
        need = -diff

        # 1) tail
        take = min(self.tail, need)
        self.tail -= take
        need -= take
        if need == 0:
            return

        # 2) gaps from back
        for j in range(len(self.gaps) - 1, -1, -1):
            if need == 0:
                break
            take = min(self.gaps[j], need)
            self.gaps[j] -= take
            need -= take

        if need == 0:
            return

        # 3) market_ahead
        take = min(self.market_ahead, need)
        self.market_ahead -= take
        need -= take
        # 若仍有 need>0，说明 target_ext=0 且我们已经归零，忽略即可。

    # ---------- init / positions ----------

    def add_order_at_tail(self, order: Order, observed_ext_qty: int) -> None:
        """在该价位新增一笔自有订单（默认排在队尾）。"""
        self._clean_orders()
        self._reconcile_external_to(observed_ext_qty)

        # 防重复
        if any(o.order_id == order.order_id for o in self.orders):
            return

        if not self.orders:
            # 第一笔：所有外部量都在它前面
            self.market_ahead = int(observed_ext_qty)
            self.gaps = []
            self.tail = 0
            self.orders = [order]
        else:
            # 新订单插在尾部：此前 tail 成为 last_order 与 new_order 的 gap
            self.gaps.append(int(self.tail))
            self.tail = 0
            self.orders.append(order)

        # 新事件 cache 清空（结构变化）
        self.quote_sig = None
        self.quote_served.clear()
        self.trade_sig = None
        self.trade_served.clear()
        self.trade_fill_map.clear()
        self.trade_expected.clear()

    def compute_front_for(self, order_id: str) -> int:
        """返回该订单在该价位的“前方量”（含外部与更早自有订单）。"""
        self._clean_orders()
        front = self.market_ahead
        for idx, o in enumerate(self.orders):
            if o.order_id == order_id:
                return max(0, int(front))
            # 经过该自有单本体
            front += o.remaining_qty
            # 经过 gap
            if idx < len(self.gaps):
                front += self.gaps[idx]
        return max(0, int(front))

    # ---------- quote update (no trade) ----------

    def _allocate_removal_across_external(self, remove_qty: int, total_before: int) -> None:
        """把外部减少量按 CDF 在 market_ahead / gaps / tail 上分配并扣减。"""
        remove_qty = max(0, int(remove_qty))
        total_before = max(1, int(total_before))  # 避免除零
        if remove_qty == 0:
            return

        segs = [("ahead", self.market_ahead)]
        for i, g in enumerate(self.gaps):
            segs.append((f"gap{i}", g))
        segs.append(("tail", self.tail))

        # 期望分配（float），然后 largest remainder 转整数
        expected = []
        prefix = 0
        for name, size in segs:
            u_l = prefix / total_before
            u_r = (prefix + size) / total_before
            frac = self._F(u_r) - self._F(u_l)
            expected.append((name, size, frac))
            prefix += size

        # 若我们的 external_total 与 total_before 不一致（可能），prefix 可能!=total_before
        # 这里仍按 total_before 做归一化；后续会 reconcile 到 after。

        raw = [(name, remove_qty * frac) for name, _, frac in expected]
        floors = {name: int(math.floor(v)) for name, v in raw}
        rem = remove_qty - sum(floors.values())

        # 余数按小数部分从大到小补齐
        frac_parts = sorted(((name, v - floors[name]) for name, v in raw), key=lambda x: x[1], reverse=True)
        for name, _ in frac_parts:
            if rem <= 0:
                break
            floors[name] += 1
            rem -= 1

        # 扣减并 cap（如果 cap 导致扣不完，剩余从 tail->...->ahead 再扣）
        left = remove_qty
        # apply in order ahead->gaps->tail
        def apply_to(name: str, amount: int):
            nonlocal left
            amount = max(0, int(amount))
            if amount == 0:
                return
            if name == "ahead":
                take = min(self.market_ahead, amount)
                self.market_ahead -= take
                left -= take
            elif name.startswith("gap"):
                idx = int(name[3:])
                take = min(self.gaps[idx], amount)
                self.gaps[idx] -= take
                left -= take
            elif name == "tail":
                take = min(self.tail, amount)
                self.tail -= take
                left -= take

        for name in floors:
            apply_to(name, floors[name])

        # 如果因为 cap 没扣够：从队尾开始补扣
        if left > 0:
            # tail
            take = min(self.tail, left)
            self.tail -= take
            left -= take
            # gaps back
            for j in range(len(self.gaps) - 1, -1, -1):
                if left <= 0:
                    break
                take = min(self.gaps[j], left)
                self.gaps[j] -= take
                left -= take
            # ahead
            if left > 0:
                take = min(self.market_ahead, left)
                self.market_ahead -= take
                left -= take

    def apply_quote_update(self, before_ext: int, after_ext: int) -> None:
        """处理该价位一次“无成交”的盘口变化（外部队列的 before/after）。"""
        self._clean_orders()
        self._reconcile_external_to(before_ext)

        before_ext = max(0, int(before_ext))
        after_ext = max(0, int(after_ext))
        delta = after_ext - before_ext
        if delta > 0:
            # 新增：默认队尾
            self.tail += delta
        elif delta < 0:
            # 减少：按 CDF 分配到外部段
            self._allocate_removal_across_external(-delta, total_before=max(1, before_ext))

        # 强制对齐到 after_ext（把残差丢到 tail，保持数值一致）
        self._reconcile_external_to(after_ext)

    # ---------- trade tick ----------

    def apply_trade_tick(self, trade_qty: int, before_ext: int, after_ext: int) -> None:
        """处理一次成交打印（在该价位发生的 trade_qty），并让外部总量对齐到 after_ext。"""
        self._clean_orders()
        self._reconcile_external_to(before_ext)

        t = max(0, int(trade_qty))

        # FIFO 推进：market_ahead -> O1 -> gap1 -> O2 -> ... -> On -> tail
        # 注意：这里“可能”会填到你的订单；这是一种无冲击回测中的常见假设。
        # 记录每笔订单本次 tick 的 fill
        self.trade_fill_map = {}

        # 1) market_ahead
        take = min(self.market_ahead, t)
        self.market_ahead -= take
        t -= take

        # 2) interleave orders and gaps
        for i, o in enumerate(list(self.orders)):
            if t <= 0:
                self.trade_fill_map[o.order_id] = 0
            else:
                fill = min(o.remaining_qty, t)
                if fill > 0:
                    o.filled_qty += fill
                self.trade_fill_map[o.order_id] = int(fill)
                t -= fill

            if t <= 0:
                # 仍需把后续订单 fill_map 填 0
                for j in range(i + 1, len(self.orders)):
                    self.trade_fill_map[self.orders[j].order_id] = 0
                break

            # consume gap_i after order i (if exists)
            if i < len(self.gaps):
                take = min(self.gaps[i], t)
                self.gaps[i] -= take
                t -= take

        # 3) tail
        if t > 0:
            take = min(self.tail, t)
            self.tail -= take
            t -= take

        # 现在用 after_ext 作为观测：把外部总量对齐
        cur_ext = self._external_total()
        after_ext = max(0, int(after_ext))
        if cur_ext > after_ext:
            # 额外减少：当作撤单/其它减少，按 CDF 再分配一次（更真实地影响前方量）
            self._allocate_removal_across_external(cur_ext - after_ext, total_before=max(1, cur_ext))
        elif cur_ext < after_ext:
            # 额外增加：默认队尾
            self.tail += (after_ext - cur_ext)

        self._reconcile_external_to(after_ext)

        # 清理可能被完全成交的订单
        self._clean_orders()


class ProbabilisticQueueModel(IQueueModel):
    """基于 L2 的概率队列模型（支持同价位多笔自有订单的 FIFO 一致性）。

    对外接口保持不变：init_order / advance_on_quote / advance_on_trade。
    """

    def __init__(self, k: float = 0.0):
        # k 越大，撤单越偏向队尾；k=0 为均匀
        self.k = float(k)
        self.positions: Dict[str, int] = {}  # 兼容旧逻辑：order_id -> front
        self._levels: Dict[Tuple[Side, Price], _LevelState] = {}

    def _get_level(self, side: Side, price: Price) -> _LevelState:
        key = (side, float(price))
        st = self._levels.get(key)
        if st is None:
            st = _LevelState(side=side, price=float(price), k=self.k)
            self._levels[key] = st
        return st

    def init_order(self, order: Order, level_qty: Qty) -> None:
        st = self._get_level(order.side, order.price)
        st.add_order_at_tail(order, int(level_qty))
        self.positions[order.order_id] = st.compute_front_for(order.order_id)

    def advance_on_quote(self, order: Order, before: Qty, after: Qty) -> None:
        st = self._get_level(order.side, order.price)

        # 同一事件会被多次调用（同价位的每笔订单调用一次）
        sig = (int(before), int(after))
        if st.quote_sig != sig:
            st.quote_sig = sig
            st.quote_served.clear()
            st.apply_quote_update(int(before), int(after))
            # 更新 positions
            for o in st.orders:
                self.positions[o.order_id] = st.compute_front_for(o.order_id)

        st.quote_served.add(order.order_id)
        # 当该价位所有自有订单都服务过后，清掉 sig，避免未来“相同 before/after”误判同一事件
        expected = {o.order_id for o in st.orders}
        if st.quote_served.issuperset(expected):
            st.quote_sig = None
            st.quote_served.clear()

    def advance_on_trade(self, order: Order, trade_px: Price, trade_qty: Qty, before: Qty, after: Qty) -> Qty:
        # 只处理发生在该订单价位的成交
        if not math.isclose(float(order.price), float(trade_px), abs_tol=1e-9):
            return 0

        st = self._get_level(order.side, order.price)

        sig = (float(trade_px), int(trade_qty), int(before), int(after))
        if st.trade_sig != sig:
            st.trade_sig = sig
            st.trade_served.clear()
            st.trade_fill_map.clear()
            st.trade_expected = {o.order_id for o in st.orders}

            st.apply_trade_tick(int(trade_qty), int(before), int(after))

            # positions 对齐
            for o in st.orders:
                self.positions[o.order_id] = st.compute_front_for(o.order_id)

            # trade_fill_map 已在 state 中填好；若某些订单在 _clean_orders 中被移除，
            # 仍保留其本次 fill 供返回（因为引擎这轮仍会为它生成 Fill）
            # 因此 expected 用“应用 trade 前”那批订单
            if not st.trade_expected:
                st.trade_expected = set(st.trade_fill_map.keys())

        # 返回该订单在该成交 tick 的 fill（每个订单只允许返回一次）
        if order.order_id in st.trade_served:
            return 0
        st.trade_served.add(order.order_id)
        fill = int(st.trade_fill_map.get(order.order_id, 0))

        # 清理 sig，避免未来相同 sig 被误判为同一事件
        if st.trade_served.issuperset(st.trade_expected):
            st.trade_sig = None
            st.trade_served.clear()
            st.trade_fill_map.clear()
            st.trade_expected.clear()

        return fill
