# BackTestSys 配置文档

本文档详细说明了回测系统所有可配置参数及其使用方法。

## 目录

- [快速开始](#快速开始)
- [配置文件格式](#配置文件格式)
- [配置参数详解](#配置参数详解)
  - [数据配置 (data)](#数据配置-data)
  - [Tape构建器配置 (tape)](#tape构建器配置-tape)
  - [交易所模拟器配置 (exchange)](#交易所模拟器配置-exchange)
  - [运行器配置 (runner)](#运行器配置-runner)
  - [投资组合配置 (portfolio)](#投资组合配置-portfolio)
  - [策略配置 (strategy)](#策略配置-strategy)
  - [回执记录器配置 (receipt_logger)](#回执记录器配置-receipt_logger)
  - [日志配置 (logging)](#日志配置-logging)
  - [快照处理配置 (snapshot)](#快照处理配置-snapshot)
- [命令行覆盖](#命令行覆盖)
- [环境变量](#环境变量)
- [配置示例](#配置示例)

## 快速开始

1. 复制默认配置文件：
```bash
cp config.xml my_config.xml
```

2. 编辑配置文件，修改所需参数

3. 运行回测：
```bash
# 使用默认配置文件 (config.xml)
python main.py

# 使用自定义配置文件
python main.py --config my_config.xml

# 显示配置信息
python main.py --show-config
```

## 配置文件格式

系统支持 XML 格式配置文件（推荐），同时向后兼容 YAML 和 JSON 格式。

### XML 格式 (推荐)
```xml
<?xml version="1.0" encoding="UTF-8"?>
<config>
    <data>
        <path>data/sample.pkl</path>
    </data>
    
    <tape>
        <ghost_rule>symmetric</ghost_rule>
        <epsilon>1.0</epsilon>
    </tape>
</config>
```

### YAML 格式 (向后兼容)
```yaml
data:
  path: "data/sample.pkl"

tape:
  ghost_rule: "symmetric"
  epsilon: 1.0
```

### JSON 格式 (向后兼容)
```json
{
  "data": {
    "path": "data/sample.pkl"
  },
  "tape": {
    "ghost_rule": "symmetric",
    "epsilon": 1.0
  }
}
```

## 配置参数详解

### 数据配置 (data)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `path` | string | `"data/sample.pkl"` | 市场数据文件路径，支持pickle格式 |

**示例：**
```xml
<data>
    <path>data/my_market_data.pkl</path>
</data>
```

---

### Tape构建器配置 (tape)

Tape构建器负责从快照对构建事件Tape，是回测引擎的核心组件。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `epsilon` | float | `1.0` | 时间分配的基准权重，防止零长度段 |
| `time_scale_lambda` | float | `0.0` | 时间缩放参数，控制事件早期/晚期分布 |
| `top_k` | int | `5` | 跟踪的价格档位数 |
| `tick_size` | float | `1.0` | 最小价格变动单位 |

**示例：**
```xml
<tape>
    <epsilon>1.0</epsilon>
    <top_k>5</top_k>
    <tick_size>0.01</tick_size>  <!-- 适用于股票等市场 -->
</tape>
```

---

### 交易所模拟器配置 (exchange)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cancel_front_ratio` | float | `0.5` | 撤单前端比例 |

#### cancel_front_ratio 详解

- `0.0` = 悲观假设（撤单全部发生在队列后端，对我们不利）
- `0.5` = 中性假设（撤单均匀分布）
- `1.0` = 乐观假设（撤单全部发生在队列前端，对我们有利）


**示例：**
```xml
<exchange>
    <cancel_front_ratio>0.5</cancel_front_ratio>
</exchange>
```

---

### 运行器配置 (runner)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `delay_out` | int | `0` | 策略→交易所延迟（tick单位，每tick=100ns） |
| `delay_in` | int | `0` | 交易所→策略延迟（tick单位，每tick=100ns） |
| `show_progress` | bool | `false` | 是否显示进度条（需要安装tqdm） |

#### 延迟参数说明

- 时间单位：tick（每tick = 100纳秒）
- 常用换算：
  - 1毫秒 = 10,000 ticks
  - 100毫秒 = 1,000,000 ticks
  - 1秒 = 10,000,000 ticks

**示例：**
```xml
<runner>
    <!-- 模拟100ms往返延迟 -->
    <delay_out>500000</delay_out>   <!-- 50ms 订单发送延迟 -->
    <delay_in>500000</delay_in>     <!-- 50ms 回执接收延迟 -->
    <show_progress>true</show_progress>
</runner>
```

---

### 投资组合配置 (portfolio)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_cash` | float | `100000.0` | 初始现金 |

**示例：**
```xml
<portfolio>
    <initial_cash>1000000.0</initial_cash>  <!-- 100万初始资金 -->
</portfolio>
```

---

### 策略配置 (strategy)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"SimpleStrategy"` | 策略名称 |
| `params.order_interval` | int | `10` | 下单间隔（每隔多少个快照下单） |
| `params.max_active_orders` | int | `5` | 最大活跃订单数 |

**示例：**
```xml
<strategy>
    <name>MyStrategy</name>
    <params>
        <order_interval>5</order_interval>      <!-- 每5个快照下单 -->
        <max_active_orders>10</max_active_orders>  <!-- 最多10个活跃订单 -->
    </params>
</strategy>
```

---

### 回执记录器配置 (receipt_logger)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `verbose` | bool | `false` | 是否实时打印回执到控制台 |
| `output_file` | string | `""` | 回执保存文件路径（留空则不保存） |

**示例：**
```xml
<receipt_logger>
    <verbose>true</verbose>
    <output_file>output/receipts.csv</output_file>
</receipt_logger>
```

---

### 日志配置 (logging)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `debug` | bool | `false` | 是否启用debug模式 |
| `log_file` | string | `""` | 日志文件路径（留空则不保存到文件） |
| `level` | string | `"INFO"` | 日志级别（DEBUG, INFO, WARNING, ERROR） |

**示例：**
```xml
<logging>
    <debug>true</debug>
    <log_file>output/backtest.log</log_file>
    <level>DEBUG</level>
</logging>
```

---

### 快照处理配置 (snapshot)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_interval_tick` | int | `5000000` | 快照最小间隔（tick单位，500ms） |
| `tolerance_tick` | int | `100000` | 时间容差（tick单位，10ms） |

**示例：**
```xml
<snapshot>
    <min_interval_tick>5000000</min_interval_tick>  <!-- 500ms -->
    <tolerance_tick>100000</tolerance_tick>         <!-- 10ms -->
</snapshot>
```

---

## 命令行覆盖

命令行参数可以覆盖配置文件中的设置：

```bash
# 覆盖数据路径
python main.py --data data/custom.pkl

# 启用进度条
python main.py --progress

# 启用debug模式
python main.py --debug

# 保存回执到文件
python main.py --save-receipts output/receipts.csv

# 保存日志到文件
python main.py --log-file output/debug.log

# 实时打印回执
python main.py --verbose-receipts

# 组合使用
python main.py --config custom.xml --progress --debug
```

完整命令行选项：

| 选项 | 简写 | 说明 |
|------|------|------|
| `--config` | `-c` | 指定配置文件路径 |
| `--show-config` | - | 运行前打印配置信息 |
| `--data` | `-d` | 覆盖 data.path |
| `--progress` | `-p` | 覆盖 runner.show_progress 为 true |
| `--verbose-receipts` | `-v` | 覆盖 receipt_logger.verbose 为 true |
| `--save-receipts` | `-s` | 覆盖 receipt_logger.output_file |
| `--debug` | - | 覆盖 logging.debug 为 true |
| `--log-file` | `-l` | 覆盖 logging.log_file |

---

## 环境变量

| 环境变量 | 说明 |
|----------|------|
| `BACKTEST_CONFIG_PATH` | 默认配置文件路径 |

**示例：**
```bash
export BACKTEST_CONFIG_PATH=/path/to/my_config.xml
python main.py
```

---


## 程序化配置

除了配置文件，您也可以在Python代码中直接配置：

```python
from quant_framework.config import (
    BacktestConfig,
    DataConfig,
    TapeConfig,
    RunnerConfig,
    PortfolioConfig,
    load_config,
    save_config,
    print_config
)

# 加载配置文件
config = load_config("config.xml")

# 修改配置
config.data.path = "data/custom.pkl"
config.runner.show_progress = True
config.portfolio.initial_cash = 500000.0

# 打印配置
print_config(config)

# 保存配置
save_config(config, "my_config.xml")

# 创建全新配置
custom_config = BacktestConfig(
    data=DataConfig(path="data/my_data.pkl"),
    runner=RunnerConfig(delay_out=100000, show_progress=True),
    portfolio=PortfolioConfig(initial_cash=200000.0),
)
```

---

## 注意事项

1. **时间单位**：所有时间相关参数使用tick单位（每tick = 100纳秒）
2. **文件路径**：相对路径基于运行目录
3. **参数验证**：系统会验证以下参数：
   - `logging.level`: 必须是 "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" 之一
4. **YAML依赖**：使用YAML格式需要安装PyYAML（`pip install pyyaml`）
