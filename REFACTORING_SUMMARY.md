# Framework Refactoring Summary

## Overview

This refactoring simplifies the backtest framework by implementing a unified EventLoop-based architecture, replacing the previous dual-path (classic vs event loop) system.

## Key Changes

### 1. Single Running Path
- **Before**: Multiple execution paths (classic, event loop, Monte Carlo)
- **After**: Only EventLoop mode with clean event-driven coordination

### 2. Clear Component Responsibilities

| Component | Responsibility | Does NOT Do |
|-----------|---------------|-------------|
| `ITapeBuilder` | Build event tape from A/B snapshots | Matching logic |
| `IExchangeSimulator` | Maintain queue state + execute FIFO matching | Generate tape |
| `IStrategyNew` | Generate orders based on market state/receipts | Matching logic |
| `EventLoopRunner` | Drive time progression + coordinate components | Specific calculations |

### 3. New Architecture Components

#### Created Files
- `quant_framework/tape/builder.py` - UnifiedTapeBuilder
  - Constructs event tape from snapshot pairs
  - Implements optimal price path building
  - Iterative volume allocation
  - Cancellation derivation

- `quant_framework/exchange/simulator.py` - FIFOExchangeSimulator
  - Per-price-level state management (ahead + shadow queue)
  - FIFO order matching
  - No-impact assumption enforcement
  - Boundary alignment

- `quant_framework/runner/event_loop.py` - EventLoopRunner
  - Event priority queue (heapq)
  - Event types: SEGMENT_END, ORDER_ARRIVAL, CANCEL_ARRIVAL, RECEIPT_TO_STRATEGY, INTERVAL_END
  - Delay handling (delay_out, delay_in)
  - Component coordination

#### Updated Files
- `quant_framework/core/interfaces.py`
  - Added `ITapeBuilder`, `IExchangeSimulator`, `IStrategyNew`, `IOrderManager`
  - Mandatory abstract methods (no default empty implementations)

- `quant_framework/core/types.py`
  - Added `TapeSegment`, `OrderReceipt`, `ReceiptType`
  - Documentation for enum usage

- `quant_framework/trading/oms.py`
  - Now implements `IOrderManager` interface
  - Backward compatible with legacy interface
  - Receipt-based order state updates

- `quant_framework/trading/strategy.py`
  - Added `SimpleNewStrategy` implementing `IStrategyNew`
  - Required `on_snapshot` and `on_receipt` methods

- `main.py`
  - Factory functions for all components
  - Clean configuration structure
  - EventLoopRunner integration

#### Deleted Files
- `quant_framework/execution/` (entire directory)
  - `engine.py`, `policies.py`, `queue_models.py`
  
- `quant_framework/simulation/` (entire directory)
  - `simple.py`, `unified_bridge.py`, `unified_tape_model.py`
  - `context.py`, `ports.py`

## New Interface Contract

### IStrategyNew
```python
@abstractmethod
def on_snapshot(self, snapshot: NormalizedSnapshot, oms: IOrderManager) -> List[Order]:
    """Called when a new snapshot arrives."""
    pass

@abstractmethod
def on_receipt(self, receipt: OrderReceipt, snapshot: NormalizedSnapshot, oms: IOrderManager) -> List[Order]:
    """Called when an order receipt is received (MANDATORY)."""
    pass
```

### IExchangeSimulator
- `reset()` - Reset state for new interval
- `on_order_arrival()` - Handle order arrival with market qty
- `on_cancel_arrival()` - Handle cancel request
- `advance()` - Process time slice with tape segment
- `align_at_boundary()` - Align state at interval boundary
- `get_queue_depth()` - Query queue depth

## Testing

All core components tested:
- ✓ Tape builder generates valid segments
- ✓ Exchange simulator handles orders correctly
- ✓ Order manager implements interface properly
- ✓ Strategy callbacks work as expected
- ✓ Integration test passes

## Security

- CodeQL scan: **0 vulnerabilities found**
- Code review: All issues addressed

## Migration Guide

### For Strategy Developers

**Old Interface:**
```python
class MyStrategy(IStrategy):
    def on_market_tick(self, book, oms):
        return []
```

**New Interface:**
```python
class MyStrategy(IStrategyNew):
    def on_snapshot(self, snapshot, oms):
        # React to market snapshots
        return []
    
    def on_receipt(self, receipt, snapshot, oms):
        # React to order fills/cancels (REQUIRED)
        return []
```

### For Backtest Users

**Old:**
```python
from quant_framework.runner.system import UnifiedRunner
runner = UnifiedRunner(...)
```

**New:**
```python
from quant_framework.runner.event_loop import EventLoopRunner
runner = EventLoopRunner(
    feed=...,
    tape_builder=...,
    exchange=...,
    strategy=...,
    oms=...,
)
```

## Benefits

1. **Simpler Architecture**: One execution path instead of multiple
2. **Clear Separation**: Each component has single responsibility
3. **Better Testability**: Components can be tested in isolation
4. **Explicit Contracts**: No default implementations hiding bugs
5. **Event-Driven**: Natural model for financial systems
6. **Extensible**: Easy to add new event types or components

## Performance

- No performance regression expected
- Event queue overhead is minimal (heapq is O(log n))
- Tape building is still O(n) where n is number of segments
- Exchange matching is still O(m) where m is number of orders

## Future Enhancements

1. Add async/await support for real-time systems
2. Extend event types for more complex strategies
3. Add risk management events
4. Support for multiple instruments
5. Distributed backtesting support
