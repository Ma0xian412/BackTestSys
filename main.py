"""Main entry point for the kernel/ports backtest architecture.

This demonstrates:
- BacktestApp + CompositionRoot for runtime wiring
- EventLoopKernel for event-driven execution
- ExecutionVenueImpl over FIFOExchangeSimulator
- Single-entry strategy interface (on_event)

Observability features:
- Progress bar for tracking backtest progress (requires tqdm)
- Receipt logging for tracking all order receipts
- Debug logging for exchange simulator events

Configuration:
- Supports external XML configuration files (default)
- Also supports YAML/JSON for backward compatibility
- Default configuration file: config.xml
- Use --config to specify a custom configuration file
"""

import argparse
import logging
import os
import sys
from datetime import datetime

from quant_framework.core import BacktestApp
from quant_framework.config import load_config, print_config, BacktestConfig


# Default configuration path
DEFAULT_CONFIG_PATH = "config.xml"


def resolve_output_path(path: str, default_filename: str) -> str:
    """Resolve a folder path to a complete file path.
    
    If the provided path is a folder, auto-generate a timestamped filename.
    If the provided path is a complete file path, return it as-is.
    
    Args:
        path: User-provided path (can be a folder or complete file path)
        default_filename: Default filename prefix (without timestamp and extension)
        
    Returns:
        Complete file path, or empty string if path is empty
    """
    if not path:
        return ""
    
    # Strip trailing slashes for consistent path handling
    normalized_path = path.rstrip(os.sep).rstrip('/')
    
    # Check if path is a directory (existing directory, ends with separator, or has no extension)
    is_directory = os.path.isdir(normalized_path) or path.endswith(os.sep) or path.endswith('/')
    
    # If path has no extension and is not an existing file, treat it as a folder
    if not is_directory and not os.path.isfile(normalized_path):
        _, file_ext = os.path.splitext(normalized_path)
        if not file_ext:
            is_directory = True
    
    if is_directory:
        # Ensure directory exists
        os.makedirs(normalized_path, exist_ok=True)
        
        # Generate timestamped filename with milliseconds for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # Determine extension based on default_filename
        if 'log' in default_filename.lower():
            output_ext = '.log'
        elif 'receipt' in default_filename.lower():
            output_ext = '.csv'
        else:
            output_ext = '.txt'
        
        filename = f"{default_filename}_{timestamp}{output_ext}"
        return os.path.join(normalized_path, filename)
    else:
        # For complete file paths, ensure parent directory exists
        dir_path = os.path.dirname(normalized_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        return normalized_path


def setup_logging(config: BacktestConfig) -> str:
    """Setup logging configuration from config.
    
    Args:
        config: Backtest configuration object
        
    Returns:
        Actual log file path if configured, otherwise empty string
    """
    debug = config.logging.debug
    log_file_config = config.logging.log_file or None
    
    # Resolve log file path if configured
    log_file = None
    if log_file_config:
        log_file = resolve_output_path(log_file_config, "backtest_log")
    
    # Configure root logger
    log_level = logging.DEBUG if debug else getattr(logging, config.logging.level, logging.INFO)
    
    handlers = []
    
    # Console handler (optional, controlled by config.logging.console)
    if config.logging.console:
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
        logging.getLogger('quant_framework.adapters.execution_venue.simulator').setLevel(logging.DEBUG)
        logging.getLogger('quant_framework.core.kernel').setLevel(logging.DEBUG)
        logging.getLogger('quant_framework.core.handlers').setLevel(logging.DEBUG)
        logging.getLogger('quant_framework.adapters.trading.receipt_logger').setLevel(logging.DEBUG)
    else:
        logging.getLogger('quant_framework.adapters.execution_venue.simulator').setLevel(logging.WARNING)
        logging.getLogger('quant_framework.core.kernel').setLevel(logging.WARNING)
        logging.getLogger('quant_framework.core.handlers').setLevel(logging.WARNING)
        logging.getLogger('quant_framework.adapters.trading.receipt_logger').setLevel(logging.WARNING)
    
    return log_file or ""


def run_backtest(config: BacktestConfig, show_config: bool = False):
    """Run the backtest using kernel + ports architecture.
    
    Args:
        config: Backtest configuration object
        show_config: If True, print configuration before running
    """
    # Setup logging (returns actual log file path if configured)
    actual_log_file = setup_logging(config)
    
    # Resolve receipt output file path (folder -> auto-generated filename)
    receipt_output_file = None
    if config.receipt_logger.output_file:
        receipt_output_file = resolve_output_path(
            config.receipt_logger.output_file, 
            "receipts"
        )
    
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
    if actual_log_file:
        print(f"Logs will be saved to: {actual_log_file}")
    if receipt_output_file:
        print(f"Receipts will be saved to: {receipt_output_file}")
    print()
    
    # main 只处理输入/config；组件构造交给 BacktestApp + CompositionRoot
    if receipt_output_file:
        config.receipt_logger.output_file = receipt_output_file

    app = BacktestApp(config)
    
    # Run backtest
    print("Starting backtest...")
    try:
        results = app.run()
        ctx = app.last_context
        oms = ctx.oms if ctx is not None else None
        receipt_logger = getattr(getattr(ctx, "obs", None), "receipt_logger", None)
        print("\nBacktest completed successfully!")
        print("=" * 60)
        print("回测运行结果 (Backtest Execution Results)")
        print("=" * 60)
        print(f"  intervals (处理的快照区间数): {results.get('intervals', 0)}")
        print(f"  final_time (最终时间戳 tick): {results.get('final_time', 0)}")
        
        diagnostics = results.get('diagnostics', {})
        print(f"\n诊断信息 (Diagnostics):")
        print(f"  intervals_processed (已处理区间数): {diagnostics.get('intervals_processed', 0)}")
        print(f"  orders_submitted (提交的订单数): {diagnostics.get('orders_submitted', 0)}")
        print(f"  orders_filled (成交的订单数): {diagnostics.get('orders_filled', 0)}")
        print(f"  receipts_generated (产生的回执数): {diagnostics.get('receipts_generated', 0)}")
        print(f"  cancels_submitted (提交的撤单数): {diagnostics.get('cancels_submitted', 0)}")
        
        # Print portfolio summary
        print(f"\n投资组合摘要 (Portfolio Summary):")
        if oms is not None:
            print(f"  Cash (现金余额): {oms.portfolio.cash:.2f}")
            print(f"  Position (持仓数量): {oms.portfolio.position}")
            print(f"  Realized PnL (已实现盈亏): {oms.portfolio.realized_pnl:.2f}")
        
        # Print order summary
        active_orders = oms.get_active_orders() if oms is not None else []
        all_orders = list(oms.orders.values()) if oms is not None else []
        print(f"\n订单摘要 (Order Summary):")
        print(f"  Total orders (总订单数): {len(all_orders)}")
        print(f"  Active orders (活跃订单数): {len(active_orders)}")
        print(f"  Filled orders (已成交订单数): {sum(1 for o in all_orders if o.status.value == 'FILLED')}")
        
        # Print receipt summary
        if receipt_logger is not None:
            receipt_logger.print_summary()
        
        # Save receipts to file if specified
        if receipt_output_file and receipt_logger is not None:
            receipt_logger.save_to_file()
            print(f"\nReceipts saved to: {receipt_output_file}")
        
        # Print all receipts if verbose mode is not already enabled
        if receipt_logger is not None and (not config.receipt_logger.verbose) and receipt_logger.records:
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
  # Run with default configuration (config.xml)
  python main.py
  
  # Run with custom configuration file
  python main.py --config my_config.xml
  
  # Show configuration before running
  python main.py --show-config
  
  # Override specific settings via command line
  python main.py --data data/custom.pkl --progress --debug
  
  # Full observability mode
  python main.py --progress --verbose-receipts --debug --save-receipts output/receipts.csv

Configuration File:
  The system loads configuration from config.xml by default.
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
