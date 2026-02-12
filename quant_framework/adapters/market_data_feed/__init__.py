"""MarketDataFeed 端口适配器。"""

from .MarketDataFeed import CsvMarketDataFeed, PickleMarketDataFeed, SnapshotDuplicatingFeed

__all__ = ["CsvMarketDataFeed", "PickleMarketDataFeed", "SnapshotDuplicatingFeed"]
