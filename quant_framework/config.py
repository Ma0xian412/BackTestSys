"""配置加载模块。

本模块实现从外部XML配置文件加载系统配置：
- 支持XML格式（主要）
- 支持YAML和JSON格式（向后兼容）
- 提供配置数据类，确保类型安全
- 支持默认值和配置验证
- 支持环境变量覆盖
- 支持从外部XML文件加载合约字典配置
"""

import os
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from pathlib import Path


# 尝试导入yaml，如果不可用则使用JSON
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False


@dataclass
class TradingHour:
    """交易时段配置。
    
    表示一个交易时段的开始和结束时间。
    时间格式为 "HH:MM:SS"，例如 "21:00:00" 或 "02:30:00"。
    
    注意：如果结束时间小于开始时间，表示跨越午夜。
    例如 StartTime="21:00:00", EndTime="02:30:00" 表示从晚上9点到凌晨2点半。
    
    Attributes:
        start_time: 开始时间，格式 "HH:MM:SS"
        end_time: 结束时间，格式 "HH:MM:SS"
    """
    start_time: str = ""
    end_time: str = ""


@dataclass
class ContractInfo:
    """合约信息配置。
    
    存储从合约字典XML文件中读取的合约信息。
    
    Attributes:
        contract_id: 合约ID
        tick_size: 最小价格变动单位
        exchange_code: 交易所代码
        trading_hours: 交易时段列表
    """
    contract_id: str = ""
    tick_size: float = 1.0
    exchange_code: str = ""
    trading_hours: List[TradingHour] = field(default_factory=list)


@dataclass
class ContractConfig:
    """合约配置。
    
    用户在主配置文件中指定合约ID和合约字典XML文件路径，
    系统会自动从合约字典中读取对应合约的详细信息。
    
    Attributes:
        contract_id: 用户指定的合约ID
        contract_dictionary_path: 合约字典XML文件路径
        contract_info: 从合约字典中读取的合约详细信息（自动填充）
    """
    contract_id: str = ""
    contract_dictionary_path: str = ""
    contract_info: Optional[ContractInfo] = None


@dataclass
class DataConfig:
    """数据配置。"""
    path: str = "data/sample.pkl"
    format: str = "pkl"  # 数据格式：pkl 或 csv
    
    def __post_init__(self):
        """Validate configuration values."""
        valid_formats = {"pkl", "csv"}
        if self.format not in valid_formats:
            raise ValueError(
                f"Invalid data format '{self.format}'. "
                f"Must be one of: {', '.join(valid_formats)}"
            )


@dataclass
class TapeConfig:
    """Tape构建器配置。"""
    ghost_rule: str = "symmetric"
    ghost_alpha: float = 0.5
    epsilon: float = 1.0
    segment_iterations: int = 2
    time_scale_lambda: float = 0.0
    cancel_front_ratio: float = 0.5
    crossing_order_policy: str = "passive"
    top_k: int = 5
    tick_size: float = 1.0
    
    def __post_init__(self):
        """Validate configuration values."""
        valid_ghost_rules = {"symmetric", "proportion", "single_bid", "single_ask"}
        if self.ghost_rule not in valid_ghost_rules:
            raise ValueError(
                f"Invalid ghost_rule '{self.ghost_rule}'. "
                f"Must be one of: {', '.join(valid_ghost_rules)}"
            )
        
        valid_crossing_policies = {"reject", "adjust", "passive"}
        if self.crossing_order_policy not in valid_crossing_policies:
            raise ValueError(
                f"Invalid crossing_order_policy '{self.crossing_order_policy}'. "
                f"Must be one of: {', '.join(valid_crossing_policies)}"
            )


@dataclass
class ExchangeConfig:
    """交易所模拟器配置。"""
    cancel_front_ratio: float = 0.5


@dataclass
class RunnerConfig:
    """运行器配置。"""
    delay_out: int = 0
    delay_in: int = 0
    show_progress: bool = False


@dataclass
class PortfolioConfig:
    """投资组合配置。"""
    initial_cash: float = 100000.0


@dataclass
class StrategyParams:
    """策略参数。"""
    order_interval: int = 10
    max_active_orders: int = 5


@dataclass
class StrategyConfig:
    """策略配置。"""
    name: str = "SimpleStrategy"
    params: StrategyParams = field(default_factory=StrategyParams)


@dataclass
class ReceiptLoggerConfig:
    """回执记录器配置。"""
    verbose: bool = False
    output_file: str = ""


@dataclass
class LoggingConfig:
    """日志配置。"""
    debug: bool = False
    log_file: str = ""
    level: str = "INFO"
    console: bool = True  # 是否在终端打印日志
    
    def __post_init__(self):
        """Validate configuration values."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.level.upper() not in valid_levels:
            raise ValueError(
                f"Invalid log level '{self.level}'. "
                f"Must be one of: {', '.join(valid_levels)}"
            )
        # Normalize to uppercase
        self.level = self.level.upper()


@dataclass
class SnapshotConfig:
    """快照处理配置。"""
    min_interval_tick: int = 5000000  # 500ms
    tolerance_tick: int = 100000  # 10ms


@dataclass
class BacktestConfig:
    """完整的回测配置。"""
    data: DataConfig = field(default_factory=DataConfig)
    tape: TapeConfig = field(default_factory=TapeConfig)
    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    receipt_logger: ReceiptLoggerConfig = field(default_factory=ReceiptLoggerConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)
    contract: ContractConfig = field(default_factory=ContractConfig)


def _dict_to_dataclass(data_class, data: Dict[str, Any]):
    """将字典转换为数据类实例。
    
    Args:
        data_class: 数据类类型
        data: 字典数据
        
    Returns:
        数据类实例
    """
    if data is None:
        return data_class()
    
    # 获取数据类字段
    field_names = {f.name for f in data_class.__dataclass_fields__.values()}
    
    # 过滤出有效的字段
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    
    return data_class(**filtered_data)


def load_config(config_path: str = None) -> BacktestConfig:
    """加载配置文件。
    
    支持XML、YAML和JSON格式。如果配置文件不存在，返回默认配置。
    
    Args:
        config_path: 配置文件路径。如果为None，按以下顺序查找：
                    1. 环境变量 BACKTEST_CONFIG_PATH
                    2. ./config.xml
                    3. ./config.yaml
                    4. ./config.json
                    5. 默认配置
                    
    Returns:
        BacktestConfig实例
        
    Raises:
        ValueError: 配置文件格式不支持
        FileNotFoundError: 指定的配置文件不存在
    """
    # 确定配置文件路径
    if config_path is None:
        config_path = os.environ.get("BACKTEST_CONFIG_PATH")
        
    if config_path is None:
        # 按优先级查找配置文件（XML优先）
        for path in ["config.xml", "config.yaml", "config.yml", "config.json"]:
            if os.path.exists(path):
                config_path = path
                break
    
    # 如果没有找到配置文件，返回默认配置
    if config_path is None or not os.path.exists(config_path):
        if config_path is not None:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        return BacktestConfig()
    
    # 读取配置文件
    file_ext = Path(config_path).suffix.lower()
    
    if file_ext == ".xml":
        raw_config = _load_xml_config(config_path)
    elif file_ext in [".yaml", ".yml"]:
        if not YAML_AVAILABLE:
            raise ValueError(
                "YAML configuration requires PyYAML. "
                "Install with: pip install pyyaml"
            )
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    elif file_ext == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration format: {file_ext}")
    
    if raw_config is None:
        return BacktestConfig()
    
    # 构建配置对象
    return _parse_config(raw_config)


def _load_xml_config(config_path: str) -> Dict[str, Any]:
    """从XML文件加载配置。
    
    注意：配置文件应该来自可信源。XML解析可能受到实体扩展攻击的影响。
    
    Args:
        config_path: XML配置文件路径
        
    Returns:
        配置字典
        
    Raises:
        ValueError: XML解析错误
    """
    try:
        tree = ET.parse(config_path)
        root = tree.getroot()
        return _xml_element_to_dict(root)
    except ET.ParseError as e:
        raise ValueError(f"XML parsing error in {config_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading XML config {config_path}: {e}")


def _xml_element_to_dict(element: ET.Element) -> Dict[str, Any]:
    """将XML元素转换为字典。
    
    Args:
        element: XML元素
        
    Returns:
        字典
    """
    result = {}
    
    for child in element:
        # 跳过注释和处理指令（它们的tag是可调用的）
        if callable(child.tag):
            continue
            
        if len(child) > 0:
            # 有子元素，递归处理
            result[child.tag] = _xml_element_to_dict(child)
        else:
            # 叶子节点，获取文本值并转换类型
            text = child.text.strip() if child.text else ""
            result[child.tag] = _convert_xml_value(text)
    
    return result


def _is_integer_string(s: str) -> bool:
    """检查字符串是否是有效的整数格式。
    
    有效格式: 可选的正负号 + 一个或多个数字
    例如: "123", "-456", "+789"
    
    Args:
        s: 要检查的字符串
        
    Returns:
        如果是有效的整数格式返回True
    """
    if not s:
        return False
    
    # 去掉可选的正负号
    if s[0] in '+-':
        s = s[1:]
    
    # 剩余部分必须是非空的纯数字
    return len(s) > 0 and s.isdigit()


def _is_float_string(s: str) -> bool:
    """检查字符串是否是有效的浮点数格式。
    
    有效格式: 
    - 可选正负号
    - 数字（至少一个）
    - 可选的小数点和小数部分
    - 可选的科学计数法 (e/E 后跟可选正负号和数字)
    
    例如: "3.14", "-1.5", "1e10", "1.5e-3"
    
    Args:
        s: 要检查的字符串
        
    Returns:
        如果是有效的浮点数格式返回True
    """
    if not s:
        return False
    
    # 使用状态机方式验证
    i = 0
    n = len(s)
    
    # 可选的正负号
    if i < n and s[i] in '+-':
        i += 1
    
    # 必须有数字（小数点前或小数点后）
    has_digits = False
    
    # 整数部分的数字
    while i < n and s[i].isdigit():
        has_digits = True
        i += 1
    
    # 可选的小数点和小数部分
    if i < n and s[i] == '.':
        i += 1
        while i < n and s[i].isdigit():
            has_digits = True
            i += 1
    
    # 如果没有任何数字，不是有效浮点数
    if not has_digits:
        return False
    
    # 可选的科学计数法部分
    if i < n and s[i] in 'eE':
        i += 1
        # e/E 后可以有可选的正负号
        if i < n and s[i] in '+-':
            i += 1
        # e/E 后必须有数字
        if i >= n or not s[i].isdigit():
            return False
        while i < n and s[i].isdigit():
            i += 1
    
    # 必须消耗完所有字符
    return i == n


def _convert_xml_value(value: str) -> Any:
    """将XML文本值转换为适当的Python类型。
    
    转换规则（按优先级）：
    1. 空字符串 -> 空字符串
    2. "true"/"false"（不区分大小写）-> bool
    3. 纯整数字符串（可带正负号）-> int
    4. 纯浮点数字符串（可带正负号和科学计数法）-> float
    5. 其他 -> 保持为字符串（包括路径、日期等）
    
    Args:
        value: XML文本值
        
    Returns:
        转换后的值
    """
    if value == "":
        return ""
    
    # 尝试转换为布尔值
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    
    stripped = value.strip()
    
    # 尝试转换为整数
    if _is_integer_string(stripped):
        return int(stripped)
    
    # 尝试转换为浮点数
    if _is_float_string(stripped):
        return float(stripped)
    
    # 返回字符串（包括路径、日期、时间等）
    return value


def _load_contract_dictionary(dictionary_path: str, contract_id: str) -> Optional[ContractInfo]:
    """从合约字典XML文件中加载指定合约的信息。
    
    合约字典XML格式：
    <ContractDictionaryConfig>
        <Contract>
            <ContractId>...</ContractId>
            <TickSize>...</TickSize>
            <ExchangeCode>...</ExchangeCode>
            <TradingHours>
                <TradingHour>
                    <StartTime>21:00:00</StartTime>
                    <EndTime>02:30:00</EndTime>
                </TradingHour>
                ...
            </TradingHours>
        </Contract>
        ...
    </ContractDictionaryConfig>
    
    Args:
        dictionary_path: 合约字典XML文件路径
        contract_id: 要查找的合约ID
        
    Returns:
        找到的合约信息，如果未找到则返回None
        
    Raises:
        FileNotFoundError: 合约字典文件不存在
        ValueError: XML解析错误
    """
    if not dictionary_path or not contract_id:
        return None
    
    if not os.path.exists(dictionary_path):
        raise FileNotFoundError(f"Contract dictionary file not found: {dictionary_path}")
    
    try:
        tree = ET.parse(dictionary_path)
        root = tree.getroot()
        
        # 查找匹配的合约
        for contract_elem in root.findall("Contract"):
            cid_elem = contract_elem.find("ContractId")
            if cid_elem is None or cid_elem.text is None:
                continue
            
            if cid_elem.text.strip() == str(contract_id):
                # 找到匹配的合约，解析信息
                tick_size = 1.0
                exchange_code = ""
                trading_hours: List[TradingHour] = []
                
                # 解析 TickSize
                tick_size_elem = contract_elem.find("TickSize")
                if tick_size_elem is not None and tick_size_elem.text:
                    try:
                        tick_size = float(tick_size_elem.text.strip())
                    except ValueError:
                        pass
                
                # 解析 ExchangeCode
                exchange_code_elem = contract_elem.find("ExchangeCode")
                if exchange_code_elem is not None and exchange_code_elem.text:
                    exchange_code = exchange_code_elem.text.strip()
                
                # 解析 TradingHours
                trading_hours_elem = contract_elem.find("TradingHours")
                if trading_hours_elem is not None:
                    for th_elem in trading_hours_elem.findall("TradingHour"):
                        start_time_elem = th_elem.find("StartTime")
                        end_time_elem = th_elem.find("EndTime")
                        
                        start_time = ""
                        end_time = ""
                        
                        if start_time_elem is not None and start_time_elem.text:
                            start_time = start_time_elem.text.strip()
                        if end_time_elem is not None and end_time_elem.text:
                            end_time = end_time_elem.text.strip()
                        
                        if start_time and end_time:
                            trading_hours.append(TradingHour(
                                start_time=start_time,
                                end_time=end_time
                            ))
                
                return ContractInfo(
                    contract_id=contract_id,
                    tick_size=tick_size,
                    exchange_code=exchange_code,
                    trading_hours=trading_hours
                )
        
        # 未找到匹配的合约
        return None
        
    except ET.ParseError as e:
        raise ValueError(f"XML parsing error in {dictionary_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error reading contract dictionary {dictionary_path}: {e}")


def _parse_config(raw_config: Dict[str, Any]) -> BacktestConfig:
    """解析原始配置字典。
    
    Args:
        raw_config: 原始配置字典
        
    Returns:
        BacktestConfig实例
    """
    # 解析各个子配置
    data_config = _dict_to_dataclass(DataConfig, raw_config.get("data"))
    tape_config = _dict_to_dataclass(TapeConfig, raw_config.get("tape"))
    exchange_config = _dict_to_dataclass(ExchangeConfig, raw_config.get("exchange"))
    runner_config = _dict_to_dataclass(RunnerConfig, raw_config.get("runner"))
    portfolio_config = _dict_to_dataclass(PortfolioConfig, raw_config.get("portfolio"))
    receipt_logger_config = _dict_to_dataclass(ReceiptLoggerConfig, raw_config.get("receipt_logger"))
    logging_config = _dict_to_dataclass(LoggingConfig, raw_config.get("logging"))
    snapshot_config = _dict_to_dataclass(SnapshotConfig, raw_config.get("snapshot"))
    
    # 解析策略配置（包含嵌套的params）
    strategy_raw = raw_config.get("strategy", {})
    strategy_params = _dict_to_dataclass(
        StrategyParams, 
        strategy_raw.get("params") if strategy_raw else None
    )
    strategy_config = StrategyConfig(
        name=strategy_raw.get("name", "SimpleStrategy") if strategy_raw else "SimpleStrategy",
        params=strategy_params
    )
    
    # 解析合约配置
    contract_raw = raw_config.get("contract", {})
    contract_id = contract_raw.get("contract_id", "") if contract_raw else ""
    contract_dictionary_path = contract_raw.get("contract_dictionary_path", "") if contract_raw else ""
    
    # 如果提供了合约ID和字典路径，则从字典中加载合约信息
    contract_info = None
    if contract_id and contract_dictionary_path:
        contract_info = _load_contract_dictionary(contract_dictionary_path, contract_id)
    
    contract_config = ContractConfig(
        contract_id=contract_id,
        contract_dictionary_path=contract_dictionary_path,
        contract_info=contract_info
    )
    
    return BacktestConfig(
        data=data_config,
        tape=tape_config,
        exchange=exchange_config,
        runner=runner_config,
        portfolio=portfolio_config,
        strategy=strategy_config,
        receipt_logger=receipt_logger_config,
        logging=logging_config,
        snapshot=snapshot_config,
        contract=contract_config,
    )


def save_config(config: BacktestConfig, config_path: str) -> None:
    """保存配置到文件。
    
    Args:
        config: 配置对象
        config_path: 配置文件路径
        
    Raises:
        ValueError: 配置文件格式不支持
    """
    file_ext = Path(config_path).suffix.lower()
    
    # 将配置转换为字典
    config_dict = _config_to_dict(config)
    
    if file_ext == ".xml":
        _save_xml_config(config_dict, config_path)
    elif file_ext in [".yaml", ".yml"]:
        if not YAML_AVAILABLE:
            raise ValueError(
                "YAML configuration requires PyYAML. "
                "Install with: pip install pyyaml"
            )
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
    elif file_ext == ".json":
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unsupported configuration format: {file_ext}")


def _save_xml_config(config_dict: Dict[str, Any], config_path: str) -> None:
    """保存配置到XML文件。
    
    Args:
        config_dict: 配置字典
        config_path: XML配置文件路径
    """
    root = ET.Element("config")
    _dict_to_xml_element(config_dict, root)
    
    # 格式化输出
    _indent_xml(root)
    
    tree = ET.ElementTree(root)
    with open(config_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)


def _dict_to_xml_element(data: Dict[str, Any], parent: ET.Element) -> None:
    """将字典转换为XML元素。
    
    Args:
        data: 字典数据
        parent: 父XML元素
    """
    for key, value in data.items():
        child = ET.SubElement(parent, key)
        if isinstance(value, dict):
            _dict_to_xml_element(value, child)
        else:
            # 转换为字符串
            if isinstance(value, bool):
                child.text = "true" if value else "false"
            else:
                child.text = str(value)


def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """格式化XML输出，添加缩进。
    
    Args:
        elem: XML元素
        level: 缩进级别
    """
    indent = "\n" + "    " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "    "
        if not elem.tail or not elem.tail.strip():
            elem.tail = indent
        for child in elem:
            _indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = indent


def _config_to_dict(config: BacktestConfig) -> Dict[str, Any]:
    """将配置对象转换为字典。
    
    Args:
        config: 配置对象
        
    Returns:
        配置字典
    """
    result = asdict(config)
    return result


def get_default_config() -> BacktestConfig:
    """获取默认配置。
    
    Returns:
        默认的BacktestConfig实例
    """
    return BacktestConfig()


def print_config(config: BacktestConfig) -> None:
    """打印配置信息。
    
    Args:
        config: 配置对象
    """
    print("\n" + "=" * 60)
    print("Backtest Configuration")
    print("=" * 60)
    
    print("\n[Data]")
    print(f"  path: {config.data.path}")
    
    print("\n[Tape Builder]")
    print(f"  ghost_rule: {config.tape.ghost_rule}")
    print(f"  ghost_alpha: {config.tape.ghost_alpha}")
    print(f"  epsilon: {config.tape.epsilon}")
    print(f"  segment_iterations: {config.tape.segment_iterations}")
    print(f"  time_scale_lambda: {config.tape.time_scale_lambda}")
    print(f"  cancel_front_ratio: {config.tape.cancel_front_ratio}")
    print(f"  crossing_order_policy: {config.tape.crossing_order_policy}")
    print(f"  top_k: {config.tape.top_k}")
    print(f"  tick_size: {config.tape.tick_size}")
    
    print("\n[Exchange]")
    print(f"  cancel_front_ratio: {config.exchange.cancel_front_ratio}")
    
    print("\n[Runner]")
    print(f"  delay_out: {config.runner.delay_out}")
    print(f"  delay_in: {config.runner.delay_in}")
    print(f"  show_progress: {config.runner.show_progress}")
    
    print("\n[Portfolio]")
    print(f"  initial_cash: {config.portfolio.initial_cash}")
    
    print("\n[Strategy]")
    print(f"  name: {config.strategy.name}")
    print(f"  params.order_interval: {config.strategy.params.order_interval}")
    print(f"  params.max_active_orders: {config.strategy.params.max_active_orders}")
    
    print("\n[Receipt Logger]")
    print(f"  verbose: {config.receipt_logger.verbose}")
    print(f"  output_file: {config.receipt_logger.output_file or '(not set)'}")
    
    print("\n[Logging]")
    print(f"  debug: {config.logging.debug}")
    print(f"  log_file: {config.logging.log_file or '(not set)'}")
    print(f"  level: {config.logging.level}")
    print(f"  console: {config.logging.console}")
    
    print("\n[Snapshot]")
    print(f"  min_interval_tick: {config.snapshot.min_interval_tick}")
    print(f"  tolerance_tick: {config.snapshot.tolerance_tick}")
    
    print("\n[Contract]")
    print(f"  contract_id: {config.contract.contract_id or '(not set)'}")
    print(f"  contract_dictionary_path: {config.contract.contract_dictionary_path or '(not set)'}")
    if config.contract.contract_info:
        info = config.contract.contract_info
        print(f"  Contract Info (loaded from dictionary):")
        print(f"    tick_size: {info.tick_size}")
        print(f"    exchange_code: {info.exchange_code or '(not set)'}")
        print(f"    trading_hours: {len(info.trading_hours)} session(s)")
        for i, th in enumerate(info.trading_hours, 1):
            print(f"      Session {i}: {th.start_time} - {th.end_time}")
    else:
        print(f"  contract_info: (not loaded)")
    
    print("=" * 60)
