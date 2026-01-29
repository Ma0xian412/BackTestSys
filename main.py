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

from quant_framework.core.data_loader import PickleMarketDataFeed, CsvMarketDataFeed, SnapshotDuplicatingFeed
from quant_framework.tape.builder import UnifiedTapeBuilder, TapeConfig as FrameworkTapeConfig
from quant_framework.exchange.simulator import FIFOExchangeSimulator
from quant_framework.runner.event_loop import EventLoopRunner, RunnerConfig as FrameworkRunnerConfig
from quant_framework.trading.strategy import SimpleStrategy
from quant_framework.trading.oms import OrderManager, Portfolio
from quant_framework.trading.receipt_logger import ReceiptLogger
from quant_framework.config import load_config, print_config, BacktestConfig


# Default configuration path
DEFAULT_CONFIG_PATH = "config.xml"


def resolve_output_path(path: str, default_filename: str) -> str:
    """将文件夹路径解析为完整的文件路径。
    
    如果提供的路径是文件夹，则自动生成带时间戳的文件名。
    如果提供的路径是完整文件路径，则直接返回。
    
    Args:
        path: 用户提供的路径（可以是文件夹或完整文件路径）
        default_filename: 默认文件名前缀（不含时间戳和扩展名）
        
    Returns:
        完整的文件路径
    """
    if not path:
        return path
    
    # 检查是否为文件夹路径（通过判断是否已存在的目录，或路径以/结尾，或没有扩展名）
    is_directory = os.path.isdir(path) or path.endswith(os.sep) or path.endswith('/')
    
    # 如果路径没有扩展名且不是已存在的文件，也认为是文件夹
    if not is_directory and not os.path.isfile(path):
        _, ext = os.path.splitext(path)
        if not ext:  # 没有扩展名，认为是文件夹
            is_directory = True
    
    if is_directory:
        # 确保目录存在
        os.makedirs(path, exist_ok=True)
        
        # 生成带时间戳的文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 根据默认文件名确定扩展名
        if 'log' in default_filename.lower():
            ext = '.log'
        elif 'receipt' in default_filename.lower():
            ext = '.csv'
        else:
            ext = '.txt'
        
        filename = f"{default_filename}_{timestamp}{ext}"
        return os.path.join(path, filename)
    else:
        # 如果是完整文件路径，确保其父目录存在
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        return path


def setup_logging(config: BacktestConfig) -> str:
    """Setup logging configuration from config.
    
    Args:
        config: Backtest configuration object
        
    Returns:
        实际使用的日志文件路径（如果配置了日志文件），否则返回空字符串
    """
    debug = config.logging.debug
    log_file_config = config.logging.log_file or None
    
    # 如果配置了日志文件路径，解析为完整路径
    log_file = None
    if log_file_config:
        log_file = resolve_output_path(log_file_config, "backtest_log")
    
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
    
    return log_file or ""


def create_feed(config: BacktestConfig):
    """Create market data feed from config.
    
    根据配置创建市场数据源：
    1. 根据 data.format 选择 CsvMarketDataFeed 或 PickleMarketDataFeed 作为 inner feed
    2. 使用 SnapshotDuplicatingFeed 包装 inner feed，实现快照复制逻辑
    
    Args:
        config: BacktestConfig 配置对象
        
    Returns:
        IMarketDataFeed: 包装后的数据源
    """
    # 根据格式选择内部 feed
    if config.data.format == "csv":
        inner_feed = CsvMarketDataFeed(config.data.path)
    else:
        inner_feed = PickleMarketDataFeed(config.data.path)
    
    # 使用 SnapshotDuplicatingFeed 包装，传入配置的 tolerance_tick
    return SnapshotDuplicatingFeed(
        inner_feed=inner_feed,
        tolerance_tick=config.snapshot.tolerance_tick
    )


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
    
    # Create components using config
    feed = create_feed(config)
    tape_builder = create_tape_builder(config)
    exchange = create_exchange(config)
    strategy = create_strategy(config)
    oms = create_oms(config)
    runner_config = create_runner_config(config)
    
    # Create receipt logger for observability
    receipt_logger = ReceiptLogger(
        output_file=receipt_output_file,
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
        if receipt_output_file:
            receipt_logger.save_to_file()
            print(f"\nReceipts saved to: {receipt_output_file}")
        
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
