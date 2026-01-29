"""Main entry point for the unified EventLoop-based backtest framework.

This demonstrates the new architecture with:
- EventLoopRunner for coordinating all components
- UnifiedTapeBuilder for constructing event tapes
- FIFOExchangeSimulator for exchange matching
- SimpleStrategy for strategy logic

Observability features:
- Progress bar for tracking backtest progress (requires tqdm)
- Receipt logging for tracking all order receipts
- Debug logging for exchange simulator events

Configuration:
- Supports external YAML/JSON configuration files
- Default configuration file: config.yaml
- Use --config to specify a custom configuration file
"""

import argparse
import logging
import sys

from quant_framework.core.data_loader import PickleMarketDataFeed
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig as FrameworkTapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig as FrameworkRunnerConfig
from quant_framework.trading.strategy import SimpleStrategy
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.receipt_logger import ReceiptLogger
from quant_framework.config import load_config, print_config, BacktestConfig


# Default configuration path
DEFAULT_CONFIG_PATH = "config.yaml"


def setup_logging(config: BacktestConfig):
    """Setup logging configuration from config.
    
    Args:
        config: Backtest configuration object
    """
    debug = config.logging.debug
    log_file = config.logging.log_file or None
    
    # Configure root logger
    log_level = logging.DEBUG if debug else getattr(logging, config.logging.level, logging.INFO)
    
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
        logging.getLogger('quant_framework.trading.receipt_logger').setLevel(logging.WARNING)


def create_feed(config: BacktestConfig):
    """Create market data feed from config."""
    return PickleMarketDataFeed(config.data.path)


def create_tape_builder(config: BacktestConfig):
    """Create tape builder from config."""
    tape_config = FrameworkTapeConfig(
        ghost_rule=config.tape.ghost_rule,
        ghost_alpha=config.tape.ghost_alpha,
        epsilon=config.tape.epsilon,
        segment_iterations=config.tape.segment_iterations,
        time_scale_lambda=config.tape.time_scale_lambda,
        cancel_front_ratio=config.tape.cancel_front_ratio,
        crossing_order_policy=config.tape.crossing_order_policy,
        top_k=config.tape.top_k,
    )
    return UnifiedTapeBuilder(config=tape_config, tick_size=config.tape.tick_size)


def create_exchange(config: BacktestConfig):
    """Create exchange simulator from config."""
    return FIFOExchangeSimulator(cancel_front_ratio=config.exchange.cancel_front_ratio)


def create_strategy(config: BacktestConfig):
    """Create strategy from config."""
    return SimpleStrategy(name=config.strategy.name)


def create_oms(config: BacktestConfig):
    """Create order manager from config."""
    portfolio = Portfolio(cash=config.portfolio.initial_cash)
    return OrderManager(portfolio=portfolio)


def create_runner_config(config: BacktestConfig):
    """Create runner configuration from config."""
    return FrameworkRunnerConfig(
        delay_out=config.runner.delay_out,
        delay_in=config.runner.delay_in,
        show_progress=config.runner.show_progress,
    )


def run_backtest(config: BacktestConfig, show_config: bool = False):
    """Run the backtest using the new EventLoop architecture.
    
    Args:
        config: Backtest configuration object
        show_config: If True, print configuration before running
    """
    # Setup logging
    setup_logging(config)
    
    print("\n" + "="*60)
    print("EventLoop-Based Unified Backtest Framework")
    print("="*60 + "\n")
    
    if show_config:
        print_config(config)
    
    if config.logging.debug:
        print("DEBUG mode enabled - detailed exchange logs will be shown")
    if config.runner.show_progress:
        print("Progress bar enabled")
    if config.receipt_logger.verbose:
        print("Verbose receipts enabled - will print receipts in real-time")
    if config.receipt_logger.output_file:
        print(f"Receipts will be saved to: {config.receipt_logger.output_file}")
    print()
    
    # Create components using config
    feed = create_feed(config)
    tape_builder = create_tape_builder(config)
    exchange = create_exchange(config)
    strategy = create_strategy(config)
    oms = create_oms(config)
    runner_config = create_runner_config(config)
    
    # Create receipt logger for observability
    receipt_logger = ReceiptLogger(
        output_file=config.receipt_logger.output_file or None,
        verbose=config.receipt_logger.verbose,
    )
    
    # Create runner with receipt logger
    runner = EventLoopRunner(
        feed=feed,
        tape_builder=tape_builder,
        exchange=exchange,
        strategy=strategy,
        oms=oms,
        config=runner_config,
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
        if config.receipt_logger.output_file:
            receipt_logger.save_to_file()
            print(f"\nReceipts saved to: {config.receipt_logger.output_file}")
        
        # Print all receipts if verbose mode is not already enabled
        if not config.receipt_logger.verbose and receipt_logger.records:
            print("\nTo see all receipts, set receipt_logger.verbose: true in config")
            print("Or use receipt_logger.print_all_receipts() programmatically")
        
    except FileNotFoundError:
        print(f"\nError: Data file not found at {config.data.path}")
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
  # Run with default configuration (config.yaml)
  python main.py
  
  # Run with custom configuration file
  python main.py --config my_config.yaml
  
  # Show configuration before running
  python main.py --show-config
  
  # Override specific settings via command line
  python main.py --data data/custom.pkl --progress --debug
  
  # Full observability mode
  python main.py --progress --verbose-receipts --debug --save-receipts output/receipts.csv

Configuration File:
  The system loads configuration from config.yaml by default.
  See CONFIG.md for detailed documentation on all available parameters.
"""
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help=f'Path to configuration file (default: {DEFAULT_CONFIG_PATH})'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Print configuration before running backtest'
    )
    
    # Command-line overrides (optional)
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='Override data.path from configuration'
    )
    
    parser.add_argument(
        '--progress', '-p',
        action='store_true',
        help='Override runner.show_progress to enable progress bar'
    )
    
    parser.add_argument(
        '--verbose-receipts', '-v',
        action='store_true',
        help='Override receipt_logger.verbose to print receipts in real-time'
    )
    
    parser.add_argument(
        '--save-receipts', '-s',
        type=str,
        default=None,
        help='Override receipt_logger.output_file to save receipts to CSV'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Override logging.debug to enable debug logging'
    )
    
    parser.add_argument(
        '--log-file', '-l',
        type=str,
        default=None,
        help='Override logging.log_file to save logs to file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please create a configuration file or specify a valid path with --config")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Apply command-line overrides
    if args.data is not None:
        config.data.path = args.data
    if args.progress:
        config.runner.show_progress = True
    if args.verbose_receipts:
        config.receipt_logger.verbose = True
    if args.save_receipts is not None:
        config.receipt_logger.output_file = args.save_receipts
    if args.debug:
        config.logging.debug = True
    if args.log_file is not None:
        config.logging.log_file = args.log_file
    
    run_backtest(config, show_config=args.show_config)


if __name__ == "__main__":
    main()
