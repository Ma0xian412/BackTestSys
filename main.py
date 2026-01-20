"""Main entry point for the unified EventLoop-based backtest framework.

This demonstrates the new architecture with:
- EventLoopRunner for coordinating all components
- UnifiedTapeBuilder for constructing event tapes
- FIFOExchangeSimulator for exchange matching
- SimpleNewStrategy for strategy logic
"""

from quant_framework.core.data_loader import PickleMarketDataFeed
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig
from quant_framework.trading.strategy import SimpleNewStrategy
from quant_framework.trading.oms import OrderManager, Portfolio


# Configuration
DATA_PATH = "data/sample.pkl"


def create_feed():
    """Create market data feed."""
    return PickleMarketDataFeed(DATA_PATH)


def create_tape_builder():
    """Create tape builder with configuration."""
    config = TapeConfig(
        ghost_rule="symmetric",      # Two-sided allocation of lastvolsplit
        epsilon=1.0,                 # Baseline weight for segment duration
        segment_iterations=2,        # Two-round iteration
        cancel_front_ratio=0.5,      # Neutral cancellation assumption
        crossing_order_policy="passive",  # Treat crossing orders as passive
    )
    return UnifiedTapeBuilder(config=config, tick_size=1.0)


def create_exchange():
    """Create exchange simulator."""
    return FIFOExchangeSimulator(cancel_front_ratio=0.5)


def create_strategy():
    """Create strategy."""
    return SimpleNewStrategy(name="SimpleStrategy")


def create_oms():
    """Create order manager."""
    portfolio = Portfolio(cash=100000.0)
    return OrderManager(portfolio=portfolio)


def create_runner_config():
    """Create runner configuration."""
    return RunnerConfig(
        delay_out=0,  # Strategy -> Exchange delay
        delay_in=0,   # Exchange -> Strategy delay
    )


def run_backtest():
    """Run the backtest using the new EventLoop architecture."""
    print("\n" + "="*60)
    print("EventLoop-Based Unified Backtest Framework")
    print("="*60 + "\n")
    
    # Create components
    feed = create_feed()
    tape_builder = create_tape_builder()
    exchange = create_exchange()
    strategy = create_strategy()
    oms = create_oms()
    config = create_runner_config()
    
    # Create runner
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=exchange,
        strategy=strategy,
        oms=oms,
        config=config,
    )
    
    # Run backtest
    print("Starting backtest...")
    try:
        results = runner.run()
        print("\nBacktest completed successfully!")
        print(f"Results: {results}")
        
        # Print portfolio summary
        print(f"\nPortfolio Summary:")
        print(f"  Cash: {oms.portfolio.cash:.2f}")
        print(f"  Position: {oms.portfolio.position}")
        print(f"  Realized PnL: {oms.portfolio.realized_pnl:.2f}")
        
        # Print order summary
        active_orders = oms.get_active_orders()
        all_orders = list(oms.orders.values())
        print(f"\nOrder Summary:")
        print(f"  Total orders: {len(all_orders)}")
        print(f"  Active orders: {len(active_orders)}")
        print(f"  Filled orders: {sum(1 for o in all_orders if o.status.value == 'FILLED')}")
        
    except FileNotFoundError:
        print(f"\nError: Data file not found at {DATA_PATH}")
        print("Please ensure the data file exists before running the backtest.")
        print("\nNote: This is expected if you don't have sample data yet.")
        print("The framework is ready to use once you provide data.")
    except Exception as e:
        print(f"\nError during backtest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_backtest()
