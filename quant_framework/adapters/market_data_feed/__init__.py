"""MarketDataFeed 端口适配器。"""

from .impl import CsvMarketDataFeed, PickleMarketDataFeed, SnapshotDuplicatingFeed

__all__ = ["CsvMarketDataFeed", "PickleMarketDataFeed", "SnapshotDuplicatingFeed"]
