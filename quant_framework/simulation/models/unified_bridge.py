"""Unified bridge model (snapshot-only L2 backtest).

目标：在相邻两帧快照 A(t0), B(t1) 之间，生成一段“机制一致”的微观事件流：
- QUOTE_UPDATE：盘口(Top5) 在中间时刻的变化（主要承载撤单/新增/队列变化）
- TRADE_TICK：模拟的成交打印（承载 maker 队列推进）
- SNAPSHOT_ARRIVAL：真实快照 B 到达

核心思路：
1) 用低维 Top-of-Book (best bid/ask 价与队列量) 的队列模型做主干（参考 Markovian LOB / queue-reactive 视角）。
2) 在已知端点 (S_A, S_B) 的条件下，用一个轻量级的 SMC bridge 抽样中间路径。
   这对应“先定义目标条件路径分布，再用 SMC 近似采样”的实现路线。

注意：这是 MVP 版本，目标是“可跑起来 + 机制一致 + 方便后续加约束”。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple, Dict

import numpy as np

from ...core.interfaces import ISimulationModel
from ...core.types import NormalizedSnapshot, Level, Price, Qty
from ...core.events import SimulationEvent, EventType


@dataclass
class _State:
    """桥接过程中使用的低维状态（更贴近 L2/MBP 语义）。

    与 tick-grid 不同，L2/MBP（Top5 快照）给的是“最优 N 个**非空**价格档位”的聚合深度，
    价格可能出现跳档（中间若干 tick 价位队列为空、不会出现在快照里）。

    因此这里把“档位”定义为：按价格竞争力排序的非空 price levels（并允许用 tick 做有限外推）。

    变量含义：
    - bp[i]: bid 侧第 i 档价格（i=0 为 best bid；严格递减）
    - ap[i]: ask 侧第 i 档价格（i=0 为 best ask；严格递增）
    - bq[i]: 与 bp[i] 对应的聚合队列量
    - aq[i]: 与 ap[i] 对应的聚合队列量
    - cum_vol: 区间内累计模拟成交量（手）

    说明：bp/ap/bq/aq 只维护“需要的档位数”（<=5）。输出快照时若不足 5 档，
    会用端点快照与 tick 做回填/外推。
    """

    bp: List[Price]
    ap: List[Price]
    bq: List[Qty]
    aq: List[Qty]
    cum_vol: int


class UnifiedBridgeModel(ISimulationModel):
    """Only-snapshot Monte-Carlo simulator using a guided particle bridge.

    参数（可按你的系统配置暴露给用户）：
    - seed: 随机种子（每次 Monte-Carlo run 不同）
    - num_steps: 在两快照之间切分的微步数 N（越大越细，但更慢）
    - num_particles: SMC 粒子数 M（越大越稳定，但更慢）
    - ess_ratio: 触发重采样的 ESS 阈值（0~1）
    - base_trade_intensity: 每步的成交强度基线（越大越“活跃”）
    - base_add_cancel_intensity: 每步新增/撤单强度基线
    """

    def __init__(
        self,
        seed: int = 42,
        num_steps: int = 50,
        num_particles: int = 200,
        ess_ratio: float = 0.5,
        base_trade_intensity: float = 1.0,
        base_add_cancel_intensity: float = 2.0,
        tick_size: Optional[float] = None,
        tick_size_by_contract: Optional[Dict[str, float]] = None,
    ):
        self.rng = np.random.default_rng(seed)
        self.num_steps = int(max(5, num_steps))
        self.num_particles = int(max(50, num_particles))
        self.ess_ratio = float(min(0.95, max(0.05, ess_ratio)))
        self.base_trade_intensity = float(max(0.0, base_trade_intensity))
        self.base_add_cancel_intensity = float(max(0.0, base_add_cancel_intensity))

        # Optional external tick-size configuration.
        # - tick_size: a single global tick size (useful for single-instrument backtests)
        # - tick_size_by_contract: per-contract tick size map (used when snapshots carry a contract id)
        self._tick_size_override: Optional[float] = float(tick_size) if (tick_size is not None and tick_size > 0) else None
        self._tick_size_by_contract: Dict[str, float] = dict(tick_size_by_contract) if tick_size_by_contract else {}

        # 订单驱动窗口：默认只模拟 best bid/ask 一档。
        self._required_bid_levels = 1
        self._required_ask_levels = 1

    def set_tick_size(self, tick_size: Optional[float]) -> None:
        """Optionally override tick size used by this model.

        If set to a positive float, the model will stop inferring tick from endpoints.
        """
        self._tick_size_override = float(tick_size) if (tick_size is not None and tick_size > 0) else None

    def _get_tick_from_context_or_config(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot, context) -> Optional[float]:
        """Return a user-provided tick size if available, else None."""
        # 1) Direct override on the model
        if self._tick_size_override is not None:
            return float(self._tick_size_override)

        # 2) Optional tick_size on the market port (runner/config can supply this)
        market_view = getattr(context, "market", None) if context is not None else None
        tick_ctx = getattr(market_view, "tick_size", None)
        if tick_ctx is not None and float(tick_ctx) > 0:
            return float(tick_ctx)

        # 3) Per-contract map, when snapshots carry a contract id/symbol
        if self._tick_size_by_contract:
            for s in (prev, curr):
                cid = getattr(s, "contract_id", None) or getattr(s, "contract", None) or getattr(s, "symbol", None)
                if cid is not None:
                    key = str(cid)
                    if key in self._tick_size_by_contract and float(self._tick_size_by_contract[key]) > 0:
                        return float(self._tick_size_by_contract[key])
        return None

    def set_required_levels(self, bid_levels: int = 1, ask_levels: int = 1) -> None:
        """由外部 Runner 动态设置需要模拟的档位深度（<=5）。

        直觉：若你的订单挂在 bid 第 3 档，则必须同时模拟 bid 第 1~3 档
        的队列消耗/撤单，否则无法判断“何时轮到你”。这符合价格优先
        (price priority) 的成交机制：更优价格档位必须先被消耗完，才轮到
        更差价格档位。
        """
        self._required_bid_levels = int(np.clip(int(bid_levels), 1, 5))
        self._required_ask_levels = int(np.clip(int(ask_levels), 1, 5))


    def _infer_required_levels_from_context(self, prev: NormalizedSnapshot, context) -> tuple[int, int]:
        """Infer required bid/ask depth (<=5) from orders affecting this interval.

        Runner passes an IntervalContext-like object (active orders + pending (due_ts, order)).
        The bridge model decides:
        - whether to generate micro events
        - how many levels are necessary for queue positioning
        """
        if context is None:
            return 1, 1

        # Prefer the dedicated order port when available; fall back to legacy context attributes.
        orders_view = getattr(context, "orders", context)

        active_orders = [
            o for o in getattr(orders_view, "active_orders", [])
            if getattr(o, "is_active", False) and getattr(o, "remaining_qty", 0) > 0
        ]

        # pending orders due inside (t0, t1]
        due_orders = []
        if hasattr(orders_view, "due_pending_orders"):
            try:
                due_orders = list(orders_view.due_pending_orders())
            except Exception:
                due_orders = []
        else:
            t0 = int(getattr(orders_view, "t0", int(prev.ts_exch)))
            t1 = int(getattr(orders_view, "t1", int(prev.ts_exch)))
            for due_ts, o in getattr(orders_view, "pending_orders", []):
                if t0 < int(due_ts) <= t1 and getattr(o, "remaining_qty", 0) > 0:
                    due_orders.append(o)

        orders = active_orders + due_orders
        if not orders:
            return 1, 1

        # 更贴近 L2/MBP 语义：depth 用“快照实际出现的非空价格档位顺序”定义，
        # 而不是用 tick 距离（tick 距离会把中间空档误当成真实 level）。
        #
        # bid_prices: 价格从高到低的 Top5 非空价位
        # ask_prices: 价格从低到高的 Top5 非空价位
        market_view = getattr(context, "market", None)
        curr = getattr(market_view, "curr", None)
        curr = curr if isinstance(curr, NormalizedSnapshot) else None

        bid_prices = sorted({float(l.price) for l in (prev.bids[:5] + (curr.bids[:5] if curr else []))}, reverse=True)
        ask_prices = sorted({float(l.price) for l in (prev.asks[:5] + (curr.asks[:5] if curr else []))})
        bid_prices = bid_prices[:5]
        ask_prices = ask_prices[:5]

        best_bid = bid_prices[0] if bid_prices else None
        best_ask = ask_prices[0] if ask_prices else None

        def bid_rank(px: float) -> int:
            """在 L2 价位列表里，px 对应的“档位深度”（1..5）。

            定义：rank = 更优 bid 价位数量 + 1。
            - 若 px 恰好等于某一档价：返回该档的 index+1
            - 若 px 位于两档之间（快照未出现的空价位）：视为“虚拟插入档”，返回插入后的位置
            - 若 px 更优于 best_bid（inside spread/new best）：返回 1
            """
            if not bid_prices or best_bid is None:
                return 1
            if px >= best_bid:
                return 1
            better = sum(1 for p in bid_prices if p > px + 1e-12)
            return int(np.clip(better + 1, 1, 5))

        def ask_rank(px: float) -> int:
            """在 L2 价位列表里，px 对应的“档位深度”（1..5），卖单用 ask 侧对称定义。"""
            if not ask_prices or best_ask is None:
                return 1
            if px <= best_ask:
                return 1
            better = sum(1 for p in ask_prices if p < px - 1e-12)
            return int(np.clip(better + 1, 1, 5))

        bid_need, ask_need = 1, 1
        for o in orders:
            side = getattr(o, "side", None)
            side_val = getattr(side, "value", side)
            px = float(getattr(o, "price", 0.0))
            if side_val == "BUY":
                bid_need = max(bid_need, bid_rank(px))
            elif side_val == "SELL":
                ask_need = max(ask_need, ask_rank(px))

        return int(bid_need), int(ask_need)

    # ----------------------------- public API -----------------------------

    def generate_events(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot, context=None) -> Iterator[SimulationEvent]:
        # Prefer the dedicated market port when available; fall back to snapshot timestamps.
        market_view = getattr(context, "market", None) if context is not None else None
        t0 = int(getattr(market_view, "t0", int(prev.ts_exch)))
        t1 = int(getattr(market_view, "t1", int(curr.ts_exch)))

        # Decide whether micro-simulation is needed for this interval.
        # If no order is active (already at exchange) and no pending order becomes active within (t0, t1],
        # the bridge can be skipped without affecting results.
        needs_micro = True
        if context is not None:
            orders_view = getattr(context, "orders", context)
            if hasattr(orders_view, "needs_micro_sim"):
                needs_micro = bool(getattr(orders_view, "needs_micro_sim"))
            else:
                active_orders = [
                    o for o in getattr(orders_view, "active_orders", [])
                    if getattr(o, "is_active", False) and getattr(o, "remaining_qty", 0) > 0
                ]
                t0c = int(getattr(orders_view, "t0", t0))
                t1c = int(getattr(orders_view, "t1", t1))
                due_pending = []
                for due_ts, o in getattr(orders_view, "pending_orders", []):
                    if t0c < int(due_ts) <= t1c and getattr(o, "remaining_qty", 0) > 0:
                        due_pending.append(o)
                needs_micro = bool(active_orders or due_pending)

        if not needs_micro:
            yield SimulationEvent(t1, EventType.SNAPSHOT_ARRIVAL, curr)
            return

        # Order-driven depth window (<=5): simulate only the price levels that matter for queue positioning
        # of active/due orders; default is best-only if context is missing.
        bid_lvls, ask_lvls = self._infer_required_levels_from_context(prev, context)
        self.set_required_levels(bid_levels=bid_lvls, ask_levels=ask_lvls)

        if t1 <= t0:
            yield SimulationEvent(t1, EventType.SNAPSHOT_ARRIVAL, curr)
            return

        # 目标成交量（如果数据没有 volume，则用 LastVolSplit 的总量作为弱约束）
        delta_vol = self._infer_interval_volume(prev, curr)

        # Tick size (最小变动单位)：
        # 1) 若用户/配置层提供，则直接使用（避免用端点“最小价差”误判 tick）
        # 2) 否则用端点快照的价格差最小正值做近似推断
        tick_cfg = self._get_tick_from_context_or_config(prev, curr, context)
        tick = float(tick_cfg) if tick_cfg is not None else self._infer_tick(prev, curr)

        # 生成中间时间点（不包含 t1，t1 留给真实快照 B）
        times = self._linspace_int(t0, t1, self.num_steps)
        # 中间事件时间点：排除 t0 与 t1。t1 由真实快照 B 覆盖。
        mid_times = times[1:-1]

        if not mid_times:
            yield SimulationEvent(t1, EventType.SNAPSHOT_ARRIVAL, curr)
            return

        # 抽样一条桥路径（每个时间点一个状态 + 该步是否发生成交）
        path_states, path_trades = self._sample_path(
            prev,
            curr,
            tick=tick,
            delta_vol=delta_vol,
            times=mid_times,
            bid_levels=self._required_bid_levels,
            ask_levels=self._required_ask_levels,
        )

        # 输出事件流（每步至少一个事件，便于外部注入“下单延迟”）
        for ts, st, trade in zip(mid_times, path_states, path_trades):
            snap = self._state_to_snapshot(st, prev, curr, tick=tick, ts=ts)
            if trade is None:
                yield SimulationEvent(ts, EventType.QUOTE_UPDATE, snap)
            else:
                px, qty = trade
                yield SimulationEvent(ts, EventType.TRADE_TICK, {"px": px, "qty": qty, "book_after": snap})

        # 末端：使用真实快照
        yield SimulationEvent(t1, EventType.SNAPSHOT_ARRIVAL, curr)

    # ----------------------------- core bridge -----------------------------

    def _sample_path(
        self,
        prev: NormalizedSnapshot,
        curr: NormalizedSnapshot,
        tick: float,
        delta_vol: int,
        times: List[int],
        bid_levels: int = 1,
        ask_levels: int = 1,
    ) -> Tuple[List[_State], List[Optional[Tuple[Price, Qty]]]]:
        """用简化 SMC bridge 抽样一条路径。

        返回：
        - states[k]: 第 k 个中间时刻的状态
        - trades[k]: 该步是否发生一笔“聚合成交”(price, qty)，用于驱动 maker 队列
        """

        bid_levels = int(np.clip(int(bid_levels), 1, 5))
        ask_levels = int(np.clip(int(ask_levels), 1, 5))

        # 起点与终点（Top-K 非空价位及对应队列量；更贴近 L2/MBP 语义）
        bp0, bq0_vec = self._top_levels(prev, side="bid", levels=bid_levels, tick=tick)
        ap0, aq0_vec = self._top_levels(prev, side="ask", levels=ask_levels, tick=tick)
        bpT, bqT_vec = self._top_levels(curr, side="bid", levels=bid_levels, tick=tick)
        apT, aqT_vec = self._top_levels(curr, side="ask", levels=ask_levels, tick=tick)

        # 端点价位池：用于“队列耗尽 -> 下一档”时尽量跳到已观测到的非空价位
        bid_pool = sorted({float(l.price) for l in (prev.bids[:5] + curr.bids[:5])}, reverse=True)
        ask_pool = sorted({float(l.price) for l in (prev.asks[:5] + curr.asks[:5])})

        # 初始化粒子
        M = self.num_particles
        N = len(times)
        bp = np.tile(np.asarray(bp0, dtype=float), (M, 1))  # shape (M, Kb)
        ap = np.tile(np.asarray(ap0, dtype=float), (M, 1))  # shape (M, Ka)
        bq = np.tile(np.asarray(bq0_vec, dtype=int), (M, 1))
        aq = np.tile(np.asarray(aq0_vec, dtype=int), (M, 1))
        cum_vol = np.zeros(M, dtype=int)

        w = np.full(M, 1.0 / M, dtype=float)

        # 记录祖先用于回溯路径
        anc = np.zeros((N, M), dtype=int)
        rec_bp = np.zeros((N, M, bid_levels), dtype=float)
        rec_ap = np.zeros((N, M, ask_levels), dtype=float)
        rec_bq = np.zeros((N, M, bid_levels), dtype=int)
        rec_aq = np.zeros((N, M, ask_levels), dtype=int)
        rec_tv = np.zeros((N, M), dtype=int)
        rec_trade_px = np.full((N, M), np.nan, dtype=float)
        rec_trade_qty = np.zeros((N, M), dtype=int)

        # 权重尺度（越小越“强约束”，越大越“软”）
        sigma_p = max(tick, 1.0 * tick)
        # 队列的尺度：取起点与终点 best 档位的平均量级
        q_scale = float(np.mean([bq0_vec[0], aq0_vec[0], bqT_vec[0], aqT_vec[0]]))
        sigma_q = max(10.0, 0.25 * q_scale)
        sigma_v = max(1.0, 0.25 * max(1, delta_vol))

        for k in range(N):
            r = max(1, N - k)  # remaining steps

            # 1) proposal：从当前状态采样下一步
            for i in range(M):
                st, trade = self._step(
                    _State(bp[i, :].tolist(), ap[i, :].tolist(), bq[i, :].tolist(), aq[i, :].tolist(), int(cum_vol[i])),
                    target=_State(bpT, apT, bqT_vec, aqT_vec, delta_vol),
                    tick=tick,
                    remaining_steps=r,
                    bid_pool=bid_pool,
                    ask_pool=ask_pool,
                )
                cum_vol[i] = st.cum_vol
                bp[i, :] = np.asarray(st.bp, dtype=float)
                ap[i, :] = np.asarray(st.ap, dtype=float)
                bq[i, :] = np.asarray(st.bq, dtype=int)
                aq[i, :] = np.asarray(st.aq, dtype=int)
                if trade is not None:
                    rec_trade_px[k, i] = trade[0]
                    rec_trade_qty[k, i] = int(trade[1])

            # 2) incremental weight：把粒子拉向终点（bridge 观测）
            #    这里用“软约束”近似 h-transform / endpoint conditioning
            d_bp = (bp - np.asarray(bpT, dtype=float)) / sigma_p
            d_ap = (ap - np.asarray(apT, dtype=float)) / sigma_p
            d_bq = (bq - np.asarray(bqT_vec, dtype=int)) / sigma_q
            d_aq = (aq - np.asarray(aqT_vec, dtype=int)) / sigma_q

            # 成交量进度：在第 k 步期望达到的累计量（线性进度）
            v_target_k = int(round(delta_vol * (k + 1) / max(1, N)))
            d_v = (cum_vol - v_target_k) / sigma_v

            logw_inc = -0.5 * (
                np.sum(d_bp ** 2, axis=1)
                + np.sum(d_ap ** 2, axis=1)
                + np.sum(d_bq ** 2, axis=1)
                + np.sum(d_aq ** 2, axis=1)
                + d_v ** 2
            )
            w = w * np.exp(logw_inc - np.max(logw_inc))
            w_sum = float(np.sum(w))
            if w_sum <= 0 or not np.isfinite(w_sum):
                w = np.full(M, 1.0 / M, dtype=float)
            else:
                w = w / w_sum

            # 3) 记录当前 step 的粒子（用于回溯）
            rec_bp[k, :, :] = bp
            rec_ap[k, :, :] = ap
            rec_bq[k, :, :] = bq
            rec_aq[k, :, :] = aq
            rec_tv[k, :] = cum_vol

            # 4) 重采样（避免粒子退化）
            ess = 1.0 / np.sum(w ** 2)
            if ess < self.ess_ratio * M:
                idx = self._systematic_resample(w)
                anc[k, :] = idx
                # 重要：记录矩阵必须与“重采样后的粒子索引”一致，否则回溯会错位
                rec_bp[k, :, :] = rec_bp[k, idx, :]
                rec_ap[k, :, :] = rec_ap[k, idx, :]
                rec_bq[k, :, :] = rec_bq[k, idx, :]
                rec_aq[k, :, :] = rec_aq[k, idx, :]
                rec_tv[k, :] = rec_tv[k, idx]
                rec_trade_px[k, :] = rec_trade_px[k, idx]
                rec_trade_qty[k, :] = rec_trade_qty[k, idx]

                bp, ap, bq, aq, cum_vol = bp[idx, :], ap[idx, :], bq[idx, :], aq[idx, :], cum_vol[idx]
                w = np.full(M, 1.0 / M, dtype=float)
            else:
                anc[k, :] = np.arange(M, dtype=int)

        # 5) 选一个粒子并回溯得到单条路径
        final_idx = int(self.rng.choice(np.arange(M), p=w))

        states: List[_State] = []
        trades: List[Optional[Tuple[Price, Qty]]] = []

        j = final_idx
        for k in reversed(range(N)):
            st = _State(
                bp=[float(x) for x in rec_bp[k, j, :].tolist()],
                ap=[float(x) for x in rec_ap[k, j, :].tolist()],
                bq=[int(x) for x in rec_bq[k, j, :].tolist()],
                aq=[int(x) for x in rec_aq[k, j, :].tolist()],
                cum_vol=int(rec_tv[k, j]),
            )
            states.append(st)
            if np.isfinite(rec_trade_px[k, j]) and rec_trade_qty[k, j] > 0:
                trades.append((float(rec_trade_px[k, j]), int(rec_trade_qty[k, j])))
            else:
                trades.append(None)
            j = int(anc[k, j])

        states.reverse()
        trades.reverse()
        return states, trades

    def _step(
        self,
        st: _State,
        target: _State,
        tick: float,
        remaining_steps: int,
        bid_pool: List[float],
        ask_pool: List[float],
    ) -> Tuple[_State, Optional[Tuple[Price, Qty]]]:
        """一步状态推进（proposal），按 L2/MBP “非空价位档”语义推进。

        机制解释：
        - 撤单 / 新增：扰动 bq/aq（按深度衰减）
        - 成交：打掉 best bid 或 best ask 的队列量，产生 TRADE_TICK
        - best 队列耗尽：best 档位出队，下一档上移成为新 best；新尾档用价位池/外推补齐

        注意：我们不假设价格档位必须连续 tick；当 best 耗尽时，优先“跳到已观测到的下一非空价位”。
        """

        bp = np.asarray(st.bp, dtype=float).copy()  # shape (Kb,)
        ap = np.asarray(st.ap, dtype=float).copy()  # shape (Ka,)
        bq = np.asarray(st.bq, dtype=int).copy()
        aq = np.asarray(st.aq, dtype=int).copy()
        v = int(st.cum_vol)

        bqT = np.asarray(target.bq, dtype=int)
        aqT = np.asarray(target.aq, dtype=int)
        bpT = np.asarray(target.bp, dtype=float)
        apT = np.asarray(target.ap, dtype=float)

        # --- 1) 指导成交量：让总成交量更容易对齐 delta_vol ---
        remaining_vol = max(0, int(target.cum_vol) - v)
        mean_trade = float(self.base_trade_intensity)
        if remaining_steps > 0:
            mean_trade += remaining_vol / float(remaining_steps)
        trade_qty = int(self.rng.poisson(mean_trade))
        trade_qty = max(0, min(trade_qty, remaining_vol))

        trade: Optional[Tuple[Price, Qty]] = None

        # --- 2) 新增/撤单噪声（队列随机扰动，按深度衰减） ---
        lam0 = float(self.base_add_cancel_intensity)
        decay = 0.65
        for i in range(len(bq)):
            lam = lam0 * (decay ** i)
            add_i = int(self.rng.poisson(lam))
            cancel_i = int(self.rng.poisson(lam))
            bq[i] = max(0, int(bq[i]) + add_i - cancel_i)
        for i in range(len(aq)):
            lam = lam0 * (decay ** i)
            add_i = int(self.rng.poisson(lam))
            cancel_i = int(self.rng.poisson(lam))
            aq[i] = max(0, int(aq[i]) + add_i - cancel_i)

        # --- 3) 成交发生在 bid 还是 ask（简单 50/50，可按微价格偏置） ---
        if trade_qty > 0:
            if self.rng.random() < 0.5:
                # 市价卖：打 best bid 队列
                trade = (float(bp[0]), trade_qty)
                bq[0] -= trade_qty
            else:
                # 市价买：打 best ask 队列
                trade = (float(ap[0]), trade_qty)
                aq[0] -= trade_qty
            v += trade_qty

        # --- 4) 队列耗尽 => 档位出队、下一档上移 ---
        refill_base = float(np.median(np.concatenate([bq[bq > 0], aq[aq > 0]]))) if (np.any(bq > 0) or np.any(aq > 0)) else 1.0
        refill_mean = int(max(1, refill_base))

        while bq[0] <= 0:
            last = float(bp[-1])
            bp[:-1] = bp[1:]
            bq[:-1] = bq[1:]
            bp[-1] = self._next_price_from_pool(side="bid", pool=bid_pool, last=last, tick=tick)
            bq[-1] = int(self.rng.poisson(refill_mean))

        while aq[0] <= 0:
            last = float(ap[-1])
            ap[:-1] = ap[1:]
            aq[:-1] = aq[1:]
            ap[-1] = self._next_price_from_pool(side="ask", pool=ask_pool, last=last, tick=tick)
            aq[-1] = int(self.rng.poisson(refill_mean))

        # --- 5) bridge guidance：对 best 价位做轻量均值回复，并把同样的 move 应用到整条梯子 ---
        bp = self._guide_ladder_prices(bp, float(bpT[0]), tick=tick, remaining_steps=remaining_steps)
        ap = self._guide_ladder_prices(ap, float(apT[0]), tick=tick, remaining_steps=remaining_steps)

        # 保证 best bid < best ask
        if bp[0] >= ap[0]:
            # 只抬 ask 梯子（更保守），保持阶梯间距不变
            shift = (bp[0] - ap[0]) + tick
            ap = ap + shift

        # 数量层面：按剩余步数把 bq/aq 往目标拉一点（逐档）
        bq = self._guide_qty_vec(bq, bqT, remaining_steps)
        aq = self._guide_qty_vec(aq, aqT, remaining_steps)

        return _State(bp.tolist(), ap.tolist(), bq.tolist(), aq.tolist(), v), trade

    # ----------------------------- utilities -----------------------------

    def _infer_interval_volume(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> int:
        # 优先使用交易所累计量（更像硬约束）
        if prev.volume is not None and curr.volume is not None:
            try:
                dv = int(curr.volume) - int(prev.volume)
                return max(0, dv)
            except Exception:
                pass
        # 次选：LastVolSplit 的总量（更像弱约束）
        if curr.last_vol_split:
            return max(0, int(sum(q for _, q in curr.last_vol_split)))
        return 0

    def _infer_tick(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> float:
        prices: List[float] = []
        for s in (prev, curr):
            for lvl in (s.bids + s.asks):
                prices.append(float(lvl.price))
        prices = sorted(set(prices))
        diffs = [prices[i+1] - prices[i] for i in range(len(prices)-1) if prices[i+1] - prices[i] > 1e-12]
        if diffs:
            return float(min(diffs))
        # fallback：如果只有一档，就用 1
        return 1.0

    def _top_levels(self, snap: NormalizedSnapshot, side: str, levels: int, tick: float) -> Tuple[List[Price], List[Qty]]:
        """从快照中提取 top-K 档位的 (price_vector, qty_vector)。

        这里的“档位”指 **快照里出现的非空 price levels（MBP Top-N）**，而不是 tick 网格的连续档。

        - levels: K (1..5)
        - 返回的 price_vector/qty_vector 长度均为 K：i=0 表示 best 档位
        - 若快照实际给出的档位不足 K：用 tick 做有限外推（仅用于让数组维度固定）
        """
        K = int(np.clip(int(levels), 1, 5))

        # fallback：用该快照所有档位 qty 的中位数（避免缺失导致 0）
        all_qty = [int(lvl.qty) for lvl in (snap.bids + snap.asks) if lvl.qty is not None]
        fallback = int(np.median(all_qty)) if all_qty else 1

        if side == "bid":
            if not snap.bids:
                # 没有盘口时，用 0 价位 + 回填量占位
                return [0.0 - i * tick for i in range(K)], [fallback for _ in range(K)]
            bids = sorted(snap.bids, key=lambda x: float(x.price), reverse=True)
            p = [float(bids[i].price) for i in range(min(K, len(bids)))]
            q = [int(bids[i].qty) for i in range(min(K, len(bids)))]
            # 不足 K：向更差价位外推
            while len(p) < K:
                p.append(float(p[-1] - tick))
                q.append(int(fallback))
            return p, q

        # ask
        if not snap.asks:
            return [0.0 + i * tick for i in range(K)], [fallback for _ in range(K)]
        asks = sorted(snap.asks, key=lambda x: float(x.price))
        p = [float(asks[i].price) for i in range(min(K, len(asks)))]
        q = [int(asks[i].qty) for i in range(min(K, len(asks)))]
        while len(p) < K:
            p.append(float(p[-1] + tick))
            q.append(int(fallback))
        return p, q

    def _best(self, snap: NormalizedSnapshot, side: str) -> Tuple[Price, Qty]:
        if side == "bid":
            if not snap.bids:
                return 0.0, 1
            best = max(snap.bids, key=lambda x: x.price)
            return float(best.price), int(best.qty)
        else:
            if not snap.asks:
                return 0.0, 1
            best = min(snap.asks, key=lambda x: x.price)
            return float(best.price), int(best.qty)

    def _next_price_from_pool(self, side: str, pool: List[float], last: float, tick: float) -> float:
        """给定“当前最差一档价格”last，选择下一更差价位。

        - bid: 选择 pool 里小于 last 的最大值；若不存在则 last - tick 外推
        - ask: 选择 pool 里大于 last 的最小值；若不存在则 last + tick 外推

        说明：这比固定 ±tick 更贴近 L2 的语义（价格档位可能跳档）。
        """
        if side == "bid":
            cand = [p for p in pool if p < last - 1e-12]
            return float(max(cand)) if cand else float(last - tick)
        cand = [p for p in pool if p > last + 1e-12]
        return float(min(cand)) if cand else float(last + tick)

    def _guide_ladder_prices(self, ladder: np.ndarray, target_best: float, tick: float, remaining_steps: int) -> np.ndarray:
        """对整个价位梯子做轻量 guidance。

        直觉：我们只对 best 价位做一个很小的“均值回复”，然后把同样的 price move
        应用到整条梯子上，从而保持梯子内部的相对间距（允许跳档）。
        """
        if ladder.size == 0:
            return ladder
        best = float(ladder[0])
        guided_best = self._guide_price(best, float(target_best), tick=tick, remaining_steps=remaining_steps)
        delta = float(guided_best - best)
        if abs(delta) < 1e-12:
            return ladder
        return ladder + delta

    def _guide_price(self, p: float, p_target: float, tick: float, remaining_steps: int) -> float:
        if remaining_steps <= 0:
            return p
        delta_ticks = int(round((p_target - p) / tick))
        # 每步最多走 1 tick；概率与剩余差/剩余步数相关
        desired = np.clip(delta_ticks / float(max(1, remaining_steps)), -1.0, 1.0)
        move_scale = 0.35
        p_up = move_scale * max(0.0, desired)
        p_dn = move_scale * max(0.0, -desired)
        u = self.rng.random()
        if u < p_up:
            return p + tick
        if u < p_up + p_dn:
            return p - tick
        return p

    def _guide_qty(self, q: int, q_target: int, remaining_steps: int) -> int:
        if remaining_steps <= 0:
            return max(1, int(q))
        drift = (q_target - q) / float(max(1, remaining_steps))
        q2 = int(round(q + 0.5 * drift))
        return max(1, q2)

    def _guide_qty_vec(self, q: np.ndarray, q_target: np.ndarray, remaining_steps: int) -> np.ndarray:
        """逐档位的队列 guidance。

        - q[i] 表示该侧第 i 档队列量
        - guidance 是“软拉回”而非硬对齐：避免破坏随机性，同时让端点约束更容易满足
        """
        if remaining_steps <= 0:
            return np.maximum(0, q.astype(int))
        q = q.astype(float)
        q_target = q_target.astype(float)
        drift = (q_target - q) / float(max(1, remaining_steps))
        q2 = np.round(q + 0.5 * drift).astype(int)
        return np.maximum(0, q2)

    def _linspace_int(self, t0: int, t1: int, n: int) -> List[int]:
        # 生成递增的整数时间序列，长度 n+1，最后一个等于 t1
        arr = np.linspace(t0, t1, n + 1)
        out = np.maximum.accumulate(arr.astype(int)).tolist()
        out[-1] = int(t1)
        return out

    def _systematic_resample(self, w: np.ndarray) -> np.ndarray:
        M = len(w)
        positions = (self.rng.random() + np.arange(M)) / M
        cumsum = np.cumsum(w)
        idx = np.zeros(M, dtype=int)
        i = j = 0
        while i < M:
            if positions[i] < cumsum[j]:
                idx[i] = j
                i += 1
            else:
                j += 1
                if j >= M:
                    j = M - 1
        return idx

    def _qty_from_snapshots(self, price: float, side: str, prev: NormalizedSnapshot, curr: NormalizedSnapshot) -> Optional[int]:
        levels = (prev.bids + curr.bids) if side == "bid" else (prev.asks + curr.asks)
        for lvl in levels:
            if abs(float(lvl.price) - float(price)) < 1e-9:
                return int(lvl.qty)
        return None

    def _state_to_snapshot(self, st: _State, prev: NormalizedSnapshot, curr: NormalizedSnapshot, tick: float, ts: int) -> NormalizedSnapshot:
        # 生成 Top5（按 L2/MBP：最优的 5 个“非空价位档”）的近似快照。
        #
        # 说明：桥状态只维护“需要的档位数”（<=5）。当不足 5 档时：
        # - 优先用端点快照 (prev/curr) 出现过的价位做补齐（更贴近 L2 的非空价位语义）
        # - 若仍不足：用 tick 做有限外推
        bid_lvls: List[Level] = []
        ask_lvls: List[Level] = []

        # 用 prev/curr 的平均深度作为缺失回填
        all_qty = [lvl.qty for lvl in (prev.bids + prev.asks + curr.bids + curr.asks) if lvl.qty is not None]
        fallback = int(np.median(all_qty)) if all_qty else 1

        # 端点价位池
        bid_pool = sorted({float(l.price) for l in (prev.bids[:5] + curr.bids[:5])}, reverse=True)
        ask_pool = sorted({float(l.price) for l in (prev.asks[:5] + curr.asks[:5])})

        # --- bids ---
        bid_prices = [float(x) for x in st.bp]
        bid_qty = [int(x) for x in st.bq]
        while len(bid_prices) < 5:
            last = bid_prices[-1] if bid_prices else (bid_pool[0] if bid_pool else 0.0)
            # 选“下一更差的已观测价位”，否则 tick 外推
            cand = [p for p in bid_pool if p < last - 1e-12 and p not in bid_prices]
            nxt = max(cand) if cand else (last - tick)
            bid_prices.append(float(nxt))
            bid_qty.append(int(self._qty_from_snapshots(nxt, "bid", prev, curr) or fallback))

        for p, q in zip(bid_prices[:5], bid_qty[:5]):
            bid_lvls.append(Level(float(p), int(max(1, q))))

        # --- asks ---
        ask_prices = [float(x) for x in st.ap]
        ask_qty = [int(x) for x in st.aq]
        while len(ask_prices) < 5:
            last = ask_prices[-1] if ask_prices else (ask_pool[0] if ask_pool else 0.0)
            cand = [p for p in ask_pool if p > last + 1e-12 and p not in ask_prices]
            nxt = min(cand) if cand else (last + tick)
            ask_prices.append(float(nxt))
            ask_qty.append(int(self._qty_from_snapshots(nxt, "ask", prev, curr) or fallback))

        for p, q in zip(ask_prices[:5], ask_qty[:5]):
            ask_lvls.append(Level(float(p), int(max(1, q))))

        return NormalizedSnapshot(
            ts_exch=int(ts),
            bids=bid_lvls,
            asks=ask_lvls,
            last_vol_split=[],
        )
