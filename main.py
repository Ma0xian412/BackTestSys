"""Main entry point for the unified EventLoop-based backtest framework.

This demonstrates the new architecture with:
- EventLoopRunner for coordinating all components
- UnifiedTapeBuilder for constructing event tapes
- FIFOExchangeSimulator for exchange matching
- SimpleNewStrategy for strategy logic

Observability features:
- Progress bar for tracking backtest progress (requires tqdm)
- Receipt logging for tracking all order receipts
- Debug logging for exchange simulator events
"""

import argparse
import logging
import sys

from quant_framework.core.data_loader import PickleMarketDataFeed
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig
from quant_framework.trading.strategy import SimpleStrategy
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.receipt_logger import ReceiptLogger


# Configuration
DATA_PATH = "data/sample.pkl"


def setup_logging(debug: bool = False, log_file: str = None):
    """Setup logging configuration.
    
    Args:
        debug: If True, enable DEBUG level logging for exchange simulator
        log_file: Optional file path to write logs to
    """
    # Configure root logger
    log_level = logging.DEBUG if debug else logging.INFO
    
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    handlers.append(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        handlers.append(file_handler)
    
    # Configure loggers
    logging.basicConfig(level=log_level, handlers=handlers)
    
    # Set specific module log levels
    if debug:
        logging.getLogger('quant_framework.exchange.simulator').setLevel(logging.DEBUG)
        logging.getLogger('quant_framework.runner.event_loop').setLevel(logging.DEBUG)
        logging.getLogger('quant_framework.trading.receipt_logger').setLevel(logging.DEBUG)
    else:
        logging.getLogger('quant_framework.exchange.simulator').setLevel(logging.WARNING)
        logging.getLogger('quant_framework.runner.event_loop').setLevel(logging.WARNING)


def create_feed(data_path: str = DATA_PATH):
    """Create market data feed."""
    return PickleMarketDataFeed(data_path)


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
    return SimpleStrategy(name="SimpleStrategy")


def create_oms():
    """Create order manager."""
    portfolio = Portfolio(cash=100000.0)
    return OrderManager(portfolio=portfolio)


def create_runner_config(show_progress: bool = False):
    """Create runner configuration.
    
    Args:
        show_progress: If True, show progress bar during backtest
    """
    return RunnerConfig(
        delay_out=0,  # Strategy -> Exchange delay
        delay_in=0,   # Exchange -> Strategy delay
        show_progress=show_progress,
    )


def run_backtest(
    data_path: str = DATA_PATH,
    show_progress: bool = False,
    verbose_receipts: bool = False,
    save_receipts: str = None,
    debug: bool = False,
    log_file: str = None,
):
    """Run the backtest using the new EventLoop architecture.
    
    Args:
        data_path: Path to market data file
        show_progress: If True, show progress bar during backtest
        verbose_receipts: If True, print receipts in real-time
        save_receipts: If provided, save receipts to this CSV file
        debug: If True, enable debug logging for exchange simulator
        log_file: If provided, save logs to this file
    """
    # Setup logging
    setup_logging(debug=debug, log_file=log_file)
    
    print("\n" + "="*60)
    print("EventLoop-Based Unified Backtest Framework")
    print("="*60 + "\n")
    
    if debug:
        print("DEBUG mode enabled - detailed exchange logs will be shown")
    if show_progress:
        print("Progress bar enabled")
    if verbose_receipts:
        print("Verbose receipts enabled - will print receipts in real-time")
    if save_receipts:
        print(f"Receipts will be saved to: {save_receipts}")
    print()
    
    # Create components
    feed = create_feed(data_path)
    tape_builder = create_tape_builder()
    exchange = create_exchange()
    strategy = create_strategy()
    oms = create_oms()
    config = create_runner_config(show_progress=show_progress)
    
    # Create receipt logger for observability
    receipt_logger = ReceiptLogger(
        output_file=save_receipts,
        verbose=verbose_receipts,
    )
    
    # Create runner with receipt logger
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=exchange,
        strategy=strategy,
        oms=oms,
        config=config,
        receipt_logger=receipt_logger,
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
        
        # Print receipt summary
        receipt_logger.print_summary()
        
        # Save receipts to file if specified
        if save_receipts:
            receipt_logger.save_to_file()
            print(f"\nReceipts saved to: {save_receipts}")
        
        # Print all receipts if verbose mode is not already enabled
        if not verbose_receipts and receipt_logger.records:
            print("\nTo see all receipts, run with --verbose-receipts flag")
            print("Or use receipt_logger.print_all_receipts() programmatically")
        
    except FileNotFoundError:
        print(f"\nError: Data file not found at {data_path}")
        print("Please ensure the data file exists before running the backtest.")
        print("\nNote: This is expected if you don't have sample data yet.")
        print("The framework is ready to use once you provide data.")
    except Exception as e:
        print(f"\nError during backtest: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="EventLoop-Based Unified Backtest Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  python main.py
  
  # With progress bar
  python main.py --progress
  
  # With verbose receipts (print in real-time)
  python main.py --verbose-receipts
  
  # Save receipts to CSV
  python main.py --save-receipts output/receipts.csv
  
  # Enable debug logging for exchange
  python main.py --debug
  
  # Save debug logs to file
  python main.py --debug --log-file output/debug.log
  
  # Full observability mode
  python main.py --progress --verbose-receipts --debug --save-receipts output/receipts.csv
"""
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=DATA_PATH,
        help=f'Path to market data file (default: {DATA_PATH})'
    )
    
    parser.add_argument(
        '--progress', '-p',
        action='store_true',
        help='Show progress bar during backtest (requires tqdm)'
    )
    
    parser.add_argument(
        '--verbose-receipts', '-v',
        action='store_true',
        help='Print receipts in real-time during backtest'
    )
    
    parser.add_argument(
        '--save-receipts', '-s',
        type=str,
        default=None,
        help='Save receipts to CSV file'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging for exchange simulator'
    )
    
    parser.add_argument(
        '--log-file', '-l',
        type=str,
        default=None,
        help='Save logs to file'
    )
    
    args = parser.parse_args()
    
    run_backtest(
        data_path=args.data,
        show_progress=args.progress,
        verbose_receipts=args.verbose_receipts,
        save_receipts=args.save_receipts,
        debug=args.debug,
        log_file=args.log_file,
    )


if __name__ == "__main__":
    main()
