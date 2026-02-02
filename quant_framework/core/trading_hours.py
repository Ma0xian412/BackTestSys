"""交易时段助手类。

本模块提供交易时段相关的通用工具类，用于：
- 解析时间字符串
- 转换tick时间戳
- 判断时间是否在交易时段内
- 查找交易时段索引

可被 SnapshotDuplicatingFeed、EventLoopRunner 等组件复用。
"""

from typing import List, Optional


class TradingHoursHelper:
    """交易时段助手类。
    
    提供交易时段相关的通用计算方法，支持：
    - 时间字符串解析（"HH:MM:SS" 格式）
    - tick时间戳与秒数的转换
    - 交易时段判断（包括跨越午夜的时段）
    - 查找时间所在的交易时段
    
    时间单位说明：
    - tick: 100纳秒（10M ticks = 1秒）
    - seconds: 一天内的秒数 (0-86399)
    
    Attributes:
        trading_hours: 交易时段列表，每个元素应有 start_time 和 end_time 属性
    """
    
    # 时间转换常量
    TICKS_PER_SECOND = 10_000_000  # 每秒tick数（1 tick = 100ns）
    SECONDS_PER_DAY = 86400  # 每天秒数
    
    def __init__(self, trading_hours: List = None):
        """初始化交易时段助手。
        
        Args:
            trading_hours: 交易时段列表，每个元素应有 start_time 和 end_time 属性。
                          如果为None或空列表，则不进行交易时段检查。
        """
        self.trading_hours = trading_hours or []
    
    def parse_time_to_seconds(self, time_str: str) -> int:
        """将时间字符串解析为一天内的秒数。
        
        Args:
            time_str: 时间字符串，格式 "HH:MM:SS"
            
        Returns:
            一天内的秒数 (0-86399)，解析失败返回0
        """
        parts = time_str.split(":")
        if len(parts) != 3:
            return 0
        try:
            hours = int(parts[0])
            minutes = int(parts[1])
            seconds = int(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except (ValueError, IndexError):
            return 0
    
    def tick_to_day_seconds(self, tick: int) -> int:
        """将tick时间戳转换为一天内的秒数。
        
        Args:
            tick: tick时间戳（每tick=100ns）
            
        Returns:
            一天内的秒数 (0-86399)
        """
        total_seconds = tick // self.TICKS_PER_SECOND
        return total_seconds % self.SECONDS_PER_DAY
    
    def seconds_to_tick_offset(self, seconds: int) -> int:
        """将秒数转换为tick偏移量。
        
        Args:
            seconds: 秒数
            
        Returns:
            tick偏移量
        """
        return seconds * self.TICKS_PER_SECOND
    
    def is_within_trading_session(self, start_seconds: int, end_seconds: int, 
                                   time_seconds: int) -> bool:
        """检查时间是否在一个交易时段内。
        
        支持跨越午夜的时段，例如 start=21:00:00, end=02:30:00。
        
        Args:
            start_seconds: 时段开始时间（一天内的秒数）
            end_seconds: 时段结束时间（一天内的秒数）
            time_seconds: 要检查的时间（一天内的秒数）
            
        Returns:
            是否在时段内
        """
        if start_seconds <= end_seconds:
            # 正常时段（如 09:00:00 - 15:00:00）
            return start_seconds <= time_seconds <= end_seconds
        else:
            # 跨越午夜的时段（如 21:00:00 - 02:30:00）
            return time_seconds >= start_seconds or time_seconds <= end_seconds
    
    def find_trading_session_index(self, time_seconds: int) -> int:
        """查找时间所在的交易时段索引。
        
        Args:
            time_seconds: 一天内的秒数 (0-86399)
            
        Returns:
            交易时段的索引（从0开始），如果不在任何时段内则返回-1
        """
        for i, th in enumerate(self.trading_hours):
            start_str = getattr(th, 'start_time', '')
            end_str = getattr(th, 'end_time', '')
            if not start_str or not end_str:
                continue
            start_sec = self.parse_time_to_seconds(start_str)
            end_sec = self.parse_time_to_seconds(end_str)
            if self.is_within_trading_session(start_sec, end_sec, time_seconds):
                return i
        return -1
    
    def is_in_any_trading_session(self, time_seconds: int) -> bool:
        """检查时间是否在任何一个交易时段内。
        
        Args:
            time_seconds: 要检查的时间（一天内的秒数）
            
        Returns:
            是否在任何交易时段内。如果未配置交易时段，默认返回True。
        """
        if not self.trading_hours:
            return True  # 未配置交易时段，默认都在交易时间内
        
        return self.find_trading_session_index(time_seconds) >= 0
    
    def get_next_trading_session_start(self, time_seconds: int) -> Optional[int]:
        """获取下一个交易时段的开始时间（秒）。
        
        搜索所有交易时段，找到时间上在time_seconds之后最近的开始时间。
        
        Args:
            time_seconds: 当前时间（一天内的秒数）
            
        Returns:
            下一个交易时段的开始时间（秒），如果没有则返回None
        """
        if not self.trading_hours:
            return None
        
        # 收集所有交易时段的开始时间
        session_starts = []
        for th in self.trading_hours:
            start_str = getattr(th, 'start_time', '')
            if start_str:
                start_sec = self.parse_time_to_seconds(start_str)
                session_starts.append(start_sec)
        
        if not session_starts:
            return None
        
        # 排序
        session_starts.sort()
        
        # 找到下一个开始时间（严格大于当前时间）
        for start in session_starts:
            if start > time_seconds:
                return start
        
        # 如果没有找到，说明当前时间已经过了所有时段的开始时间
        # 这意味着在当天最后一个时段之后，不应该有下一个时段
        return None
    
    def spans_trading_session_gap(self, t_prev: int, t_curr: int) -> bool:
        """检查两个时间戳之间是否跨越了交易时段间隔。
        
        如果两个时间点不在同一个交易时段内，或者中间存在非交易时间，
        则认为跨越了交易时段间隔。
        
        注意：此方法使用日内时间进行比较，假设回测数据在同一交易日内。
        对于跨越多个自然日的数据，应确保数据按日分割处理。
        
        Args:
            t_prev: 前一个时间戳（tick单位）
            t_curr: 当前时间戳（tick单位）
            
        Returns:
            是否跨越交易时段间隔
        """
        if not self.trading_hours:
            return False  # 未配置交易时段，不跨越间隔
        
        prev_seconds = self.tick_to_day_seconds(t_prev)
        curr_seconds = self.tick_to_day_seconds(t_curr)
        
        # 检查两个时间点是否都在交易时段内
        prev_session = self.find_trading_session_index(prev_seconds)
        curr_session = self.find_trading_session_index(curr_seconds)
        
        # 如果两者都不在交易时段内，不需要复制
        if prev_session < 0 and curr_session < 0:
            return True
        
        # 如果任一不在交易时段内，跨越了间隔
        if prev_session < 0 or curr_session < 0:
            return True
        
        # 如果在不同的交易时段，跨越了间隔
        if prev_session != curr_session:
            return True
        
        return False
    
    def is_after_last_trading_session(self, time_seconds: int) -> bool:
        """检查时间是否在最后一个交易时段结束之后。
        
        对于跨越午夜的交易时段（如21:00-02:30），此方法将：
        - 将结束时间02:30视为当天的最后时段结束
        - 时间03:00-08:59之间被视为"在最后时段之后"（假设没有其他日盘时段）
        
        注意：此方法假设交易日内的时段按照配置顺序排列。
        对于复杂的跨日场景，需要额外的逻辑处理。
        
        Args:
            time_seconds: 当前时间（一天内的秒数）
            
        Returns:
            是否在最后一个交易时段结束之后
        """
        if not self.trading_hours:
            return False  # 未配置交易时段，永远不会"在最后时段之后"
        
        # 首先，如果当前时间在任何交易时段内，则不是"在最后时段之后"
        if self.is_in_any_trading_session(time_seconds):
            return False
        
        # 收集所有交易时段的结束时间（考虑跨夜时段）
        session_ends_non_overnight = []  # 非跨夜时段的结束时间
        has_overnight_session = False
        overnight_end = 0  # 跨夜时段的结束时间
        
        for th in self.trading_hours:
            end_str = getattr(th, 'end_time', '')
            start_str = getattr(th, 'start_time', '')
            if end_str and start_str:
                end_sec = self.parse_time_to_seconds(end_str)
                start_sec = self.parse_time_to_seconds(start_str)
                
                if end_sec >= start_sec:  # 非跨夜时段
                    session_ends_non_overnight.append(end_sec)
                else:  # 跨夜时段
                    has_overnight_session = True
                    overnight_end = end_sec  # 记录跨夜时段的结束时间
        
        # 如果只有跨夜时段，没有日盘时段
        if not session_ends_non_overnight and has_overnight_session:
            # 跨夜时段的结束时间之后，到下一天的开始之前，都是"最后时段之后"
            return time_seconds > overnight_end
        
        # 如果有非跨夜时段
        if session_ends_non_overnight:
            max_end = max(session_ends_non_overnight)
            # 如果当前时间大于最大的非跨夜时段结束时间，则是"最后时段之后"
            # 但需要排除跨夜时段的夜间部分
            if has_overnight_session:
                # 有跨夜时段，需要考虑夜盘开始前的时间
                for th in self.trading_hours:
                    end_str = getattr(th, 'end_time', '')
                    start_str = getattr(th, 'start_time', '')
                    if end_str and start_str:
                        end_sec = self.parse_time_to_seconds(end_str)
                        start_sec = self.parse_time_to_seconds(start_str)
                        if end_sec < start_sec:  # 跨夜时段
                            # 如果当前时间在日盘结束后、夜盘开始前
                            if max_end < time_seconds < start_sec:
                                return False  # 这是"两个时段之间"
                            break
            
            return time_seconds > max_end
        
        return False
