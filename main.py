"""内核/端口回测架构的主入口。

本模块演示：
- BacktestApp + CompositionRoot 运行时装配
- EventLoopKernel 事件驱动执行
- ExecutionVenue_Impl 基于 Simulator + SegmentBaseAlgorithm
- 单入口策略接口 (on_event)

可观测性：
- 进度条跟踪回测进度（需 tqdm）
- 回执日志记录所有订单回执
- 调试日志记录交易所模拟器事件

配置：
- 支持外部 XML 配置文件（默认）
- 支持 YAML/JSON 向后兼容
- 默认配置文件：config.xml
- 使用 --config 指定自定义配置文件
"""

import argparse
import csv
import logging
import os
import sys
from dataclasses import asdict, fields
from datetime import datetime

from quant_framework.core import BacktestApp
from quant_framework.core.run_result import (
    BacktestRunResult,
    CancelRequestRecord,
    DoneInfo,
    ExecutionDetail,
    OrderInfo,
)
from quant_framework.adapters.factory import BacktestConfigFactory
from quant_framework.config import load_config, print_config, BacktestConfig


# 默认配置文件路径
DEFAULT_CONFIG_PATH = "config.xml"


def resolve_output_path(path: str, default_filename: str) -> str:
    """将文件夹路径解析为完整文件路径。
    
    若提供的路径是文件夹，则自动生成带时间戳的文件名。
    若提供的路径已是完整文件路径，则原样返回。
    
    Args:
        path: 用户提供的路径（可为文件夹或完整文件路径）
        default_filename: 默认文件名前缀（不含时间戳和扩展名）
        
    Returns:
        完整文件路径，若路径为空则返回空字符串
    """
    if not path:
        return ""
    
    # 去掉尾部斜杠以统一路径处理
    normalized_path = path.rstrip(os.sep).rstrip('/')
    
    # 判断路径是否为目录（已存在的目录、以分隔符结尾、或无扩展名）
    is_directory = os.path.isdir(normalized_path) or path.endswith(os.sep) or path.endswith('/')
    
    # 若路径无扩展名且非已存在文件，则视为文件夹
    if not is_directory and not os.path.isfile(normalized_path):
        _, file_ext = os.path.splitext(normalized_path)
        if not file_ext:
            is_directory = True
    
    if is_directory:
        # 确保目录存在
        os.makedirs(normalized_path, exist_ok=True)
        
        # 生成带毫秒的时间戳文件名以保证唯一性
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        # 根据 default_filename 确定扩展名
        if 'log' in default_filename.lower():
            output_ext = '.log'
        elif 'receipt' in default_filename.lower():
            output_ext = '.csv'
        else:
            output_ext = '.txt'
        
        filename = f"{default_filename}_{timestamp}{output_ext}"
        return os.path.join(normalized_path, filename)
    else:
        # 对于完整文件路径，确保父目录存在
        dir_path = os.path.dirname(normalized_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        return normalized_path


def _contract_name_suffix(config: BacktestConfig) -> str:
    """从 config.contract 提取合约标识，用于文件名后缀。"""
    c = config.contract
    if c.contract_id and c.contract_id.strip():
        return c.contract_id.strip()
    if c.contract_info and c.contract_info.contract_id:
        return str(c.contract_info.contract_id)
    return ""


def resolve_result_output_base(path: str, config: BacktestConfig) -> tuple[str, bool] | None:
    """解析回测结果落盘路径。

    若路径为空返回 None。
    若为目录，创建 run_result_合约_时间戳 子目录并返回 (目录路径, True)。
    若为文件路径，返回 (去掉扩展名的基础路径_合约, False)。
    合约信息来自 config.contract，无则省略。

    Args:
        path: 配置中的 output_file
        config: 回测配置，用于获取合约信息

    Returns:
        (base_path, is_directory) 或 None
    """
    if not path or not path.strip():
        return None
    contract_suffix = _contract_name_suffix(config)
    contract_part = f"_{contract_suffix}" if contract_suffix else ""
    normalized = path.strip().rstrip(os.sep).rstrip("/")
    is_directory = (
        os.path.isdir(normalized)
        or path.endswith(os.sep)
        or path.endswith("/")
    )
    if not is_directory and not os.path.isfile(normalized):
        _, ext = os.path.splitext(normalized)
        if not ext:
            is_directory = True
    if is_directory:
        os.makedirs(normalized, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        subdir = os.path.join(normalized, f"run_result{contract_part}_{timestamp}")
        os.makedirs(subdir, exist_ok=True)
        return (subdir, True)
    parent = os.path.dirname(normalized)
    if parent:
        os.makedirs(parent, exist_ok=True)
    base, _ = os.path.splitext(normalized)
    return (f"{base}{contract_part}", False)


_RESULT_TABLE_FIELDS = {
    "DoneInfo": [f.name for f in fields(DoneInfo)],
    "ExecutionDetail": [f.name for f in fields(ExecutionDetail)],
    "OrderInfo": [f.name for f in fields(OrderInfo)],
    "CancelRequest": [f.name for f in fields(CancelRequestRecord)],
}


def _save_contract_info(config: BacktestConfig, base_path: str, is_directory: bool) -> str | None:
    """落盘 config.contract 信息为 contract_info.csv。

    Returns:
        写入的文件路径，若无合约信息则返回 None
    """
    c = config.contract
    rows = [
        ("contract_id", c.contract_id or ""),
        ("contract_dictionary_path", c.contract_dictionary_path or ""),
    ]
    ci = c.contract_info
    if ci:
        rows.extend([
            ("contract_info.contract_id", str(ci.contract_id)),
            ("contract_info.partition_day", str(ci.partition_day)),
            ("contract_info.tick_size", str(ci.tick_size)),
            ("contract_info.exchange_code", ci.exchange_code or ""),
            ("contract_info.machine_name", ci.machine_name or ""),
        ])
    has_any = any(v for _, v in rows if v)
    if not has_any:
        return None
    if is_directory:
        filepath = os.path.join(base_path, "contract_info.csv")
    else:
        filepath = f"{base_path}_contract_info.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("key", "value"))
        writer.writerows(rows)
    return filepath


def save_run_result_to_csv(
    result: BacktestRunResult,
    base_path: str,
    is_directory: bool,
    config: BacktestConfig,
) -> list[str]:
    """将 BacktestRunResult 落盘为 4 个 CSV 文件，并落盘 contract 信息。

    Args:
        result: 回测结果
        base_path: 基础路径（目录或文件前缀）
        is_directory: True 则在 base_path 下生成 DoneInfo.csv 等；False 则生成 base_path_DoneInfo.csv 等
        config: 回测配置，用于落盘 contract 信息

    Returns:
        生成的文件路径列表（含 contract_info.csv，若有）
    """
    tables = [
        ("DoneInfo", result.DoneInfo),
        ("ExecutionDetail", result.ExecutionDetail),
        ("OrderInfo", result.OrderInfo),
        ("CancelRequest", result.CancelRequest),
    ]
    written = []
    for name, rows in tables:
        if is_directory:
            filepath = os.path.join(base_path, f"{name}.csv")
        else:
            filepath = f"{base_path}_{name}.csv"
        field_names = _RESULT_TABLE_FIELDS[name]
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))
        written.append(filepath)
    contract_path = _save_contract_info(config, base_path, is_directory)
    if contract_path:
        written.append(contract_path)
    return written


def setup_logging(config: BacktestConfig) -> str:
    """根据配置设置日志。
    
    Args:
        config: 回测配置对象
        
    Returns:
        若已配置日志文件则返回实际路径，否则返回空字符串
    """
    debug = config.logging.debug
    log_file_config = config.logging.log_file or None
    
    # 若配置了日志文件则解析其路径
    log_file = None
    if log_file_config:
        log_file = resolve_output_path(log_file_config, "backtest_log")
    
    # 配置根 logger
    log_level = logging.DEBUG if debug else getattr(logging, config.logging.level, logging.INFO)
    
    handlers = []
    
    # 控制台 handler（可选，由 config.logging.console 控制）
    if config.logging.console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        handlers.append(console_handler)
    
    # 文件 handler（可选）
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        handlers.append(file_handler)
    
    # 配置各 logger
    logging.basicConfig(level=log_level, handlers=handlers)
    
    # 设置各模块的日志级别
    if debug:
        logging.getLogger('quant_framework.adapters.execution_venue.simulator').setLevel(logging.DEBUG)
        logging.getLogger('quant_framework.adapters.execution_venue.match_algorithm').setLevel(logging.DEBUG)
        logging.getLogger('quant_framework.core.kernel').setLevel(logging.DEBUG)
        logging.getLogger('quant_framework.core.handlers').setLevel(logging.DEBUG)
        logging.getLogger('quant_framework.adapters.observability.Observability_Impl').setLevel(logging.DEBUG)
    else:
        logging.getLogger('quant_framework.adapters.execution_venue.simulator').setLevel(logging.WARNING)
        logging.getLogger('quant_framework.adapters.execution_venue.match_algorithm').setLevel(logging.WARNING)
        logging.getLogger('quant_framework.core.kernel').setLevel(logging.WARNING)
        logging.getLogger('quant_framework.core.handlers').setLevel(logging.WARNING)
        logging.getLogger('quant_framework.adapters.observability.Observability_Impl').setLevel(logging.WARNING)
    
    return log_file or ""


def run_backtest(config: BacktestConfig, show_config: bool = False):
    """使用内核 + 端口架构运行回测。
    
    Args:
        config: 回测配置对象
        show_config: 为 True 时在运行前打印配置
    """
    # 设置日志（若已配置则返回实际日志文件路径）
    actual_log_file = setup_logging(config)
    
    # 解析回执输出文件路径（文件夹 -> 自动生成文件名）
    receipt_output_file = None
    if config.receipt_logger.output_file:
        receipt_output_file = resolve_output_path(
            config.receipt_logger.output_file,
            "receipts"
        )

    # 解析回测结果落盘路径（含合约信息）
    result_output_base = resolve_result_output_base(config.run_result.output_file, config)
    
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
    if result_output_base:
        base_path, is_dir = result_output_base
        if is_dir:
            print(f"Run result will be saved to: {base_path}/")
        else:
            print(f"Run result will be saved to: {base_path}_*.csv")
    print()
    
    # main 只处理输入/config；组件构造交给 BacktestApp + CompositionRoot
    if receipt_output_file:
        config.receipt_logger.output_file = receipt_output_file

    runtime_cfg = BacktestConfigFactory().create(config)
    app = BacktestApp(runtime_cfg)
    
    # 运行回测
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
        print(f"  DoneInfo 数量: {len(results.DoneInfo)}")
        print(f"  ExecutionDetail 数量: {len(results.ExecutionDetail)}")
        print(f"  OrderInfo 数量: {len(results.OrderInfo)}")
        print(f"  CancelRequest 数量: {len(results.CancelRequest)}")
        
        # 打印投资组合摘要
        print(f"\n投资组合摘要 (Portfolio Summary):")
        if oms is not None:
            print(f"  Cash (现金余额): {oms.portfolio.cash:.2f}")
            print(f"  Position (持仓数量): {oms.portfolio.position}")
            print(f"  Realized PnL (已实现盈亏): {oms.portfolio.realized_pnl:.2f}")
        
        # 打印订单摘要
        active_orders = oms.get_active_orders() if oms is not None else []
        all_orders = list(oms.orders.values()) if oms is not None else []
        print(f"\n订单摘要 (Order Summary):")
        print(f"  Total orders (总订单数): {len(all_orders)}")
        print(f"  Active orders (活跃订单数): {len(active_orders)}")
        print(f"  Filled orders (已成交订单数): {sum(1 for o in all_orders if o.status.value == 'FILLED')}")
        
        # 打印回执摘要
        if receipt_logger is not None:
            receipt_logger.print_summary()
        
        # 若已指定则保存回执到文件
        if receipt_output_file and receipt_logger is not None:
            receipt_logger.save_to_file()
            print(f"\nReceipts saved to: {receipt_output_file}")

        # 若已指定则落盘回测结果（4 个 CSV）
        if result_output_base:
            base_path, is_dir = result_output_base
            written = save_run_result_to_csv(results, base_path, is_dir, config)
            print("\nRun result saved to:")
            for p in written:
                print(f"  {p}")
        
        # 若未启用 verbose 模式则提示可查看所有回执
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
    """主入口，负责命令行参数解析。"""
    parser = argparse.ArgumentParser(
        description="EventLoop-Based Unified Backtest Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 使用默认配置（config.xml）运行
  python main.py
  
  # 使用自定义配置文件运行
  python main.py --config my_config.xml
  
  # 运行前显示配置
  python main.py --show-config
  
  # 通过命令行覆盖特定设置
  python main.py --data data/custom.pkl --progress --debug
  
  # 完整可观测模式
  python main.py --progress --verbose-receipts --debug --save-receipts output/receipts.csv

配置文件：
  系统默认从 config.xml 加载配置。
  详见 CONFIG.md 获取所有可用参数的文档。
"""
    )
    
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help=f'配置文件路径（默认：{DEFAULT_CONFIG_PATH}）'
    )
    
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='运行回测前打印配置'
    )
    
    # 命令行覆盖（可选）
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=None,
        help='覆盖配置中的 data.path'
    )
    
    parser.add_argument(
        '--progress', '-p',
        action='store_true',
        help='覆盖 runner.show_progress 以启用进度条'
    )
    
    parser.add_argument(
        '--verbose-receipts', '-v',
        action='store_true',
        help='覆盖 receipt_logger.verbose 以实时打印回执'
    )
    
    parser.add_argument(
        '--save-receipts', '-s',
        type=str,
        default=None,
        help='覆盖 receipt_logger.output_file 以将回执保存为 CSV'
    )

    parser.add_argument(
        '--save-result',
        type=str,
        default=None,
        help='覆盖 run_result.output_file 以将回测结果落盘为 4 个 CSV'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='覆盖 logging.debug 以启用调试日志'
    )
    
    parser.add_argument(
        '--log-file', '-l',
        type=str,
        default=None,
        help='覆盖 logging.log_file 以将日志保存到文件'
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please create a configuration file or specify a valid path with --config")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # 应用命令行覆盖
    if args.data is not None:
        config.data.path = args.data
    if args.progress:
        config.runner.show_progress = True
    if args.verbose_receipts:
        config.receipt_logger.verbose = True
    if args.save_receipts is not None:
        config.receipt_logger.output_file = args.save_receipts
    if args.save_result is not None:
        config.run_result.output_file = args.save_result
    if args.debug:
        config.logging.debug = True
    if args.log_file is not None:
        config.logging.log_file = args.log_file
    
    run_backtest(config, show_config=args.show_config)


if __name__ == "__main__":
    main()
