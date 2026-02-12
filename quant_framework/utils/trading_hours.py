"""交易时段助手类。"""

from typing import List, Optional


class TradingHoursHelper:
    """交易时段助手类。"""

    TICKS_PER_SECOND = 10_000_000
    SECONDS_PER_DAY = 86400

    def __init__(self, trading_hours: List = None):
        self.trading_hours = trading_hours or []

    def parse_time_to_seconds(self, time_str: str) -> int:
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
        total_seconds = tick // self.TICKS_PER_SECOND
        return total_seconds % self.SECONDS_PER_DAY

    def seconds_to_tick_offset(self, seconds: int) -> int:
        return seconds * self.TICKS_PER_SECOND

    def is_within_trading_session(self, start_seconds: int, end_seconds: int, time_seconds: int) -> bool:
        if start_seconds <= end_seconds:
            return start_seconds <= time_seconds <= end_seconds
        return time_seconds >= start_seconds or time_seconds <= end_seconds

    def find_trading_session_index(self, time_seconds: int) -> int:
        for i, th in enumerate(self.trading_hours):
            start_str = getattr(th, "start_time", "")
            end_str = getattr(th, "end_time", "")
            if not start_str or not end_str:
                continue
            start_sec = self.parse_time_to_seconds(start_str)
            end_sec = self.parse_time_to_seconds(end_str)
            if self.is_within_trading_session(start_sec, end_sec, time_seconds):
                return i
        return -1

    def is_in_any_trading_session(self, time_seconds: int) -> bool:
        if not self.trading_hours:
            return True
        return self.find_trading_session_index(time_seconds) >= 0

    def get_next_trading_session_start(self, time_seconds: int) -> Optional[int]:
        if not self.trading_hours:
            return None
        session_starts = []
        for th in self.trading_hours:
            start_str = getattr(th, "start_time", "")
            if start_str:
                start_sec = self.parse_time_to_seconds(start_str)
                session_starts.append(start_sec)
        if not session_starts:
            return None
        session_starts.sort()
        for start in session_starts:
            if start > time_seconds:
                return start
        return None

    def spans_trading_session_gap(self, t_prev: int, t_curr: int) -> bool:
        if not self.trading_hours:
            return False
        prev_seconds = self.tick_to_day_seconds(t_prev)
        curr_seconds = self.tick_to_day_seconds(t_curr)
        prev_session = self.find_trading_session_index(prev_seconds)
        curr_session = self.find_trading_session_index(curr_seconds)
        if prev_session < 0 and curr_session < 0:
            return True
        if prev_session < 0 or curr_session < 0:
            return True
        if prev_session != curr_session:
            return True
        return False

    def is_after_last_trading_session(self, time_seconds: int) -> bool:
        if not self.trading_hours:
            return False
        if self.is_in_any_trading_session(time_seconds):
            return False
        session_ends_non_overnight = []
        has_overnight_session = False
        overnight_end = 0
        for th in self.trading_hours:
            end_str = getattr(th, "end_time", "")
            start_str = getattr(th, "start_time", "")
            if end_str and start_str:
                end_sec = self.parse_time_to_seconds(end_str)
                start_sec = self.parse_time_to_seconds(start_str)
                if end_sec >= start_sec:
                    session_ends_non_overnight.append(end_sec)
                else:
                    has_overnight_session = True
                    overnight_end = end_sec
        if not session_ends_non_overnight and has_overnight_session:
            return time_seconds > overnight_end
        if session_ends_non_overnight:
            max_end = max(session_ends_non_overnight)
            if has_overnight_session:
                for th in self.trading_hours:
                    end_str = getattr(th, "end_time", "")
                    start_str = getattr(th, "start_time", "")
                    if end_str and start_str:
                        end_sec = self.parse_time_to_seconds(end_str)
                        start_sec = self.parse_time_to_seconds(start_str)
                        if end_sec < start_sec:
                            if max_end < time_seconds < start_sec:
                                return False
                            break
            return time_seconds > max_end
        return False
