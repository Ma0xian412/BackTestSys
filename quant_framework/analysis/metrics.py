from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from ..core.types import Fill, Order


class ExecutionQualityMetrics:
    """订单执行质量指标（以订单为中心）。

    设计目标：
    - L2-only 回测中，优先关注：订单 fill（是否成交/成交比例）与生命周期（等待时间）。
    - PnL/cash 等账户类指标在此版本不做强绑定；runner/portfolio 可自行扩展并注入。
    """

    def __init__(self) -> None:
        # 所有 fill 记录（按时间追加）
        self.fills: List[Fill] = []

        # 已提交订单（策略发单时登记）：order_id -> meta
        # meta = {"order": Order, "submit_ts": int, "due_ts": int, "arrival_ts": Optional[int]}
        self.submitted: Dict[str, Dict[str, Any]] = {}

        # 已到达交易所并注册到撮合器的订单（oms.register_orders 时登记）
        self.arrived: Dict[str, Order] = {}

        # 每个订单的 first/last fill 时间（从 fills 推导）
        self._first_fill_ts: Dict[str, int] = {}
        self._last_fill_ts: Dict[str, int] = {}

        # 用于计算“未成交订单生命周期”的区间终点
        self.last_ts: int = 0

    # ---------------- hooks from runner ----------------

    def update_time(self, ts: int) -> None:
        self.last_ts = int(ts)

    def on_order_submitted(self, o: Order, submit_ts: int, due_ts: int) -> None:
        """策略生成订单时调用（订单尚未到交易所）。"""
        oid = o.order_id
        if oid in self.submitted:
            return
        self.submitted[oid] = {
            "order": o,
            "submit_ts": int(submit_ts),
            "due_ts": int(due_ts),
            "arrival_ts": None,
        }

    def on_order_new(self, o: Order) -> None:
        """订单到达交易所并注册到 OMS/撮合器时调用。"""
        oid = o.order_id
        self.arrived[oid] = o

        # 若 runner 已设置 arrival_time，优先使用
        arr_ts = None
        try:
            arr_ts = int(o.arrival_time) if o.arrival_time is not None else None
        except Exception:
            arr_ts = None

        if oid in self.submitted:
            self.submitted[oid]["arrival_ts"] = arr_ts
        else:
            # 兜底：若没有显式 submitted hook，也把它当作 submitted
            submit_ts = int(getattr(o, "create_time", 0) or 0)
            self.submitted[oid] = {"order": o, "submit_ts": submit_ts, "due_ts": arr_ts or submit_ts, "arrival_ts": arr_ts}

    def on_fill(self, f: Fill) -> None:
        self.fills.append(f)
        oid = f.order_id
        ts = int(f.ts)
        if oid not in self._first_fill_ts:
            self._first_fill_ts[oid] = ts
        self._last_fill_ts[oid] = ts

    # ---------------- internal helpers ----------------

    def _fill_qty_map(self) -> Dict[str, int]:
        m: Dict[str, int] = {}
        for f in self.fills:
            m[f.order_id] = m.get(f.order_id, 0) + int(f.qty)
        return m

    def _order_iter(self) -> List[Dict[str, Any]]:
        """返回提交订单的 meta 列表（稳定顺序）。"""
        items = list(self.submitted.items())
        items.sort(key=lambda kv: str(kv[0]))
        return [meta for _, meta in items]

    # ---------------- public tables & summaries ----------------

    def order_table(self, end_ts: Optional[int] = None) -> pd.DataFrame:
        """逐订单明细表（用于分析/导出）。"""
        if end_ts is None:
            end_ts = int(self.last_ts)

        fill_qty = self._fill_qty_map()

        rows: List[Dict[str, Any]] = []
        for meta in self._order_iter():
            o: Order = meta["order"]
            oid = o.order_id

            submit_ts = int(meta.get("submit_ts") or getattr(o, "create_time", 0) or 0)
            due_ts = int(meta.get("due_ts") or 0)
            arrival_ts = meta.get("arrival_ts", None)
            if arrival_ts is None:
                try:
                    arrival_ts = int(o.arrival_time) if o.arrival_time is not None else None
                except Exception:
                    arrival_ts = None

            qty = int(getattr(o, "qty", 0) or 0)
            filled = int(fill_qty.get(oid, 0))
            filled = min(filled, max(0, qty))

            first_fill = self._first_fill_ts.get(oid)
            last_fill = self._last_fill_ts.get(oid)

            if qty <= 0:
                status = "INVALID_QTY"
                fill_ratio = None
            else:
                if filled <= 0:
                    status = "UNFILLED"
                elif filled < qty:
                    status = "PARTIAL"
                else:
                    status = "FULL"
                fill_ratio = filled / qty

            # 生命周期相关：需要 arrival_ts
            ttf = (first_fill - arrival_ts) if (arrival_ts is not None and first_fill is not None) else None
            ttf_full = (last_fill - arrival_ts) if (arrival_ts is not None and status == "FULL" and last_fill is not None) else None

            # lifetime：若有 fill，用 last_fill；否则用 end_ts（代表“到回测结束仍未成交”）
            if arrival_ts is not None:
                end_for_life = last_fill if last_fill is not None else int(end_ts)
                lifetime = int(end_for_life) - int(arrival_ts)
            else:
                lifetime = None

            rows.append({
                "order_id": oid,
                "side": getattr(o.side, "value", str(o.side)),
                "price": float(getattr(o, "price", 0.0) or 0.0),
                "qty": qty,
                "filled_qty": filled,
                "fill_ratio": fill_ratio,
                "status": status,
                "submit_ts": submit_ts,
                "due_ts": due_ts,
                "arrival_ts": arrival_ts,
                "first_fill_ts": first_fill,
                "last_fill_ts": last_fill,
                "time_to_first_fill": ttf,
                "time_to_full_fill": ttf_full,
                "lifetime": lifetime,
            })

        cols = ['order_id', 'side', 'price', 'qty', 'filled_qty', 'fill_ratio', 'status', 'submit_ts', 'due_ts', 'arrival_ts', 'first_fill_ts', 'last_fill_ts', 'time_to_first_fill', 'time_to_full_fill', 'lifetime']
        return pd.DataFrame(rows, columns=cols)

    def get_summary(self, end_ts: Optional[int] = None) -> Dict[str, Any]:
        """单次 run 的汇总指标（不做跨 run 的 mean/std）。"""
        df = self.order_table(end_ts=end_ts)

        num_orders = int(len(df))
        total_qty = int(df["qty"].fillna(0).sum()) if num_orders else 0
        total_filled_qty = int(df["filled_qty"].fillna(0).sum()) if num_orders else 0

        # 订单口径 fill rate
        any_fill = int((df["filled_qty"] > 0).sum()) if num_orders else 0
        full_fill = int((df["status"] == "FULL").sum()) if num_orders else 0

        arrived_mask = df["arrival_ts"].notna() if num_orders else None
        num_arrived = int(arrived_mask.sum()) if num_orders else 0
        total_qty_arrived = int(df.loc[arrived_mask, "qty"].fillna(0).sum()) if num_orders else 0
        total_filled_qty_arrived = int(df.loc[arrived_mask, "filled_qty"].fillna(0).sum()) if num_orders else 0
        any_fill_arrived = int(((df["filled_qty"] > 0) & arrived_mask).sum()) if num_orders else 0
        full_fill_arrived = int(((df["status"] == "FULL") & arrived_mask).sum()) if num_orders else 0

        # 平均 fill ratio（逐订单均值）
        avg_fill_ratio = float(df["fill_ratio"].dropna().mean()) if num_orders else 0.0

        # 生命周期（需要 arrival_ts）
        avg_ttf = float(df["time_to_first_fill"].dropna().mean()) if num_orders else 0.0
        avg_ttf_full = float(df["time_to_full_fill"].dropna().mean()) if num_orders else 0.0
        avg_lifetime = float(df["lifetime"].dropna().mean()) if num_orders else 0.0

        # 成交记录数
        total_fills = int(len(self.fills))

        return {
            "num_orders": num_orders,
            "num_orders_arrived": num_arrived,
            "num_fills": total_fills,

            "total_submitted_qty": total_qty,
            "total_filled_qty": total_filled_qty,
            "qty_fill_rate": (total_filled_qty / total_qty) if total_qty else 0.0,

            "total_arrived_qty": total_qty_arrived,
            "total_filled_qty_arrived": total_filled_qty_arrived,
            "qty_fill_rate_arrived": (total_filled_qty_arrived / total_qty_arrived) if total_qty_arrived else 0.0,

            "order_fill_rate_any": (any_fill / num_orders) if num_orders else 0.0,
            "order_fill_rate_full": (full_fill / num_orders) if num_orders else 0.0,
            "order_fill_rate_any_arrived": (any_fill_arrived / num_arrived) if num_arrived else 0.0,
            "order_fill_rate_full_arrived": (full_fill_arrived / num_arrived) if num_arrived else 0.0,

            "avg_fill_ratio": avg_fill_ratio,
            "avg_time_to_first_fill": avg_ttf,
            "avg_time_to_full_fill": avg_ttf_full,
            "avg_lifetime": avg_lifetime,
        }

    # ---------------- exports ----------------

    def export_fills_csv(self, out_path: str, run_id: Optional[int] = None) -> str:
        """导出 fill 逐笔记录到 CSV（每次 run 一个文件更稳）。"""
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        rows: List[Dict[str, Any]] = []
        for f in self.fills:
            oid = f.order_id
            meta = self.submitted.get(oid)
            submit_ts = None
            arrival_ts = None
            if meta is not None:
                submit_ts = meta.get("submit_ts")
                arrival_ts = meta.get("arrival_ts")
            rows.append({
                "run_id": run_id,
                "fill_id": f.fill_id,
                "order_id": oid,
                "side": getattr(f.side, "value", str(f.side)),
                "price": float(f.price),
                "qty": int(f.qty),
                "ts": int(f.ts),
                "liquidity": str(f.liquidity),
                "submit_ts": submit_ts,
                "arrival_ts": arrival_ts,
            })

        cols = ['run_id', 'fill_id', 'order_id', 'side', 'price', 'qty', 'ts', 'liquidity', 'submit_ts', 'arrival_ts']
        pd.DataFrame(rows, columns=cols).to_csv(out_path, index=False)
        return out_path

    def export_orders_csv(self, out_path: str, end_ts: Optional[int] = None) -> str:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        self.order_table(end_ts=end_ts).to_csv(out_path, index=False)
        return out_path
