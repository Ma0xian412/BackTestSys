"""CSV格式的行情数据源实现。"""

import csv
import ast
import math
import logging
from typing import List, Any, Optional, Tuple
from ...core.port import IMarketDataFeed
from ...core.data_structure import NormalizedSnapshot, Level

# 设置模块级logger
logger = logging.getLogger(__name__)


class CsvMarketDataFeed_Impl(IMarketDataFeed):
    """CSV格式的行情数据源。
    
    读取CSV文件，按RecvTick排序后提供快照数据。
    字段与PickleMarketDataFeed_Impl兼容：
    - ExchTick / RecvTick
    - Bid, Bid2..Bid5 与 BidVol, BidVol2..BidVol5
    - Ask, Ask2..Ask5 与 AskVol, AskVol2..AskVol5
    - Last, Volume, Turnover, AveragePrice
    - LastVolSplit: [(price, qty), ...]（CSV中存储为字符串形式）
    """
    
    def __init__(self, file_path: str):
        """初始化CSV数据源。
        
        Args:
            file_path: CSV文件路径
        """
        self.file_path = file_path
        self.data: List[Any] = []
        self._load_data()
        self.idx = 0
    
    def _load_data(self):
        """加载CSV数据并按RecvTick排序。"""
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                self.data = list(reader)
            
            # 按RecvTick排序（处理可能的空值和类型转换）
            def get_recv_tick(row):
                val = row.get('RecvTick', '')
                if val == '' or val is None:
                    return 0
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    return 0
            
            self.data.sort(key=get_recv_tick)
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            self.data = []
    
    def next(self) -> Optional[NormalizedSnapshot]:
        """获取下一个快照。"""
        if self.idx >= len(self.data):
            return None
        row = self.data[self.idx]
        self.idx += 1
        return self._parse_row(row)
    
    def reset(self):
        """重置数据源。"""
        self.idx = 0
    
    def __len__(self) -> int:
        """快照条数（用于进度条等场景）。"""
        return int(len(self.data))
    
    def _parse_row(self, row) -> Optional[NormalizedSnapshot]:
        """把CSV行解析成标准快照。
        
        兼容字段：
        - ExchTick / RecvTick
        - Bid, Bid2..Bid5 与 BidVol, BidVol2..BidVol5
        - Ask, Ask2..Ask5 与 AskVol, AskVol2..AskVol5
        - Last, Volume, Turnover, AveragePrice
        - LastVolSplit: [(price, qty), ...]（CSV中存储为字符串形式）
        """
        
        def _is_nan(x: Any) -> bool:
            try:
                if x is None or x == '':
                    return True
                if isinstance(x, float) and math.isnan(x):
                    return True
                # CSV读取的值可能是字符串"nan"或"NaN"
                if isinstance(x, str) and x.lower() in ('nan', 'none', ''):
                    return True
                return False
            except Exception:
                return False
        
        def _to_float(x: Any) -> Optional[float]:
            if _is_nan(x):
                return None
            try:
                return float(x)
            except Exception:
                return None
        
        def _to_int(x: Any) -> Optional[int]:
            if _is_nan(x):
                return None
            try:
                return int(x)
            except Exception:
                try:
                    return int(float(x))
                except Exception:
                    return None
        
        def _parse_levels(side: str) -> List[Level]:
            # side in {"Bid", "Ask"}
            out: List[Level] = []
            for i in range(1, 6):
                px_key = side if i == 1 else f"{side}{i}"
                vol_key = f"{side}Vol" if i == 1 else f"{side}Vol{i}"
                px = _to_float(row.get(px_key))
                vol = _to_int(row.get(vol_key))
                if px is None or vol is None:
                    continue
                out.append(Level(px, vol))
            # 保险：排序（Bid 降序，Ask 升序）
            out.sort(key=lambda x: x.price, reverse=(side == "Bid"))
            return out
        
        def _parse_last_vol_split(raw_split: Any) -> List[Tuple[float, int]]:
            """解析LastVolSplit字段。
            
            CSV中该字段可能以字符串形式存储，如 "[(100.0, 10), (101.0, 20)]"
            """
            lvs: List[Tuple[float, int]] = []
            
            if raw_split is None or raw_split == '':
                return lvs
            
            # 如果已经是列表，直接处理
            if isinstance(raw_split, list):
                data = raw_split
            elif isinstance(raw_split, str):
                # 尝试解析字符串表示的列表
                try:
                    data = ast.literal_eval(raw_split)
                    if not isinstance(data, list):
                        return lvs
                except (ValueError, SyntaxError):
                    return lvs
            else:
                return lvs
            
            for it in data:
                try:
                    p, q = it
                    p2 = _to_float(p)
                    q2 = _to_int(q)
                    if p2 is not None and q2 is not None:
                        # Round price to 6 decimal places to fix floating-point precision issues
                        # e.g., 1050.199999999 -> 1050.2
                        p2 = round(p2, 6)
                        lvs.append((p2, q2))
                except Exception:
                    continue
            
            return lvs
        
        try:
            ts_exch = _to_int(row.get('ExchTick')) or 0
            ts_recv = _to_int(row.get('RecvTick'))
            
            # RecvTick is mandatory - error if missing or invalid
            if ts_recv is None:
                raise ValueError(
                    f"RecvTick field is required but missing or invalid. "
                    f"Expected integer timestamp in tick units (100ns per tick), got: {row.get('RecvTick')!r}"
                )
            
            bids = _parse_levels("Bid")
            asks = _parse_levels("Ask")
            
            last = _to_float(row.get('Last'))
            volume = _to_int(row.get('Volume'))
            turnover = _to_float(row.get('Turnover'))
            avg_px = _to_float(row.get('AveragePrice'))
            
            last_vol_split = _parse_last_vol_split(row.get('LastVolSplit'))
            
            return NormalizedSnapshot(
                ts_recv=ts_recv,  # 主时间线（必填）
                bids=bids,
                asks=asks,
                last_vol_split=last_vol_split,
                ts_exch=ts_exch,  # 可选（仅记录）
                last=last,
                volume=volume,
                turnover=turnover,
                average_price=avg_px,
            )
        except Exception as e:
            # Re-raise ValueError for mandatory field errors
            if isinstance(e, ValueError):
                raise
            return None
