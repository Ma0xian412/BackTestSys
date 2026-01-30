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
| `output_file` | string | `""` | 回执保存路径（可以是文件夹或完整文件路径，留空则不保存）。如果提供文件夹路径，程序会自动生成带时间戳的CSV文件名 |

**示例：**
```xml
<receipt_logger>
    <verbose>true</verbose>
    <!-- 方式1：提供完整文件路径 -->
    <output_file>output/receipts.csv</output_file>
    
    <!-- 方式2：提供文件夹路径，程序自动生成文件名（如 receipts_20240101_120000.csv） -->
    <!-- <output_file>output/</output_file> -->
</receipt_logger>
```

---

### 日志配置 (logging)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `debug` | bool | `false` | 是否启用debug模式 |
| `log_file` | string | `""` | 日志保存路径（可以是文件夹或完整文件路径，留空则不保存到文件）。如果提供文件夹路径，程序会自动生成带时间戳的日志文件名 |
| `level` | string | `"INFO"` | 日志级别（DEBUG, INFO, WARNING, ERROR） |
| `console` | bool | `true` | 是否在终端打印日志 |

**示例：**
```xml
<logging>
    <debug>true</debug>
    <!-- 方式1：提供完整文件路径 -->
    <log_file>output/backtest.log</log_file>
    
    <!-- 方式2：提供文件夹路径，程序自动生成文件名（如 backtest_log_20240101_120000.log） -->
    <!-- <log_file>output/</log_file> -->
    <level>DEBUG</level>
    
    <!-- 是否在终端打印日志（默认开启） -->
    <console>true</console>
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

---

## 输出结果说明

回测完成后，系统会输出以下结果信息。本节详细说明各字段的含义，帮助用户理解回测结果。

### 示例输出

```
============================================================
回测运行结果 (Backtest Execution Results)
============================================================
  intervals (处理的快照区间数): 100
  final_time (最终时间戳 tick): 1706616000000000

诊断信息 (Diagnostics):
  intervals_processed (已处理区间数): 100
  orders_submitted (提交的订单数): 50
  orders_filled (成交的订单数): 35
  receipts_generated (产生的回执数): 45
  cancels_submitted (提交的撤单数): 5

投资组合摘要 (Portfolio Summary):
  Cash (现金余额): 98500.00
  Position (持仓数量): 15
  Realized PnL (已实现盈亏): 150.00

订单摘要 (Order Summary):
  Total orders (总订单数): 50
  Active orders (活跃订单数): 3
  Filled orders (已成交订单数): 35

============================================================
回执记录器摘要 (Receipt Logger Summary)
============================================================
Total Receipts (总回执数): 45
Total Orders (总订单数): 50
Total Order Qty (总下单数量): 500
Total Filled Qty (总成交数量): 350

回执类型分布 (Receipt Type Distribution):
  - Partial Fill (部分成交): 10
  - Full Fill (全部成交): 25
  - Canceled (已撤单): 8
  - Rejected (已拒绝): 2

订单状态分布 (Order Status Distribution):
  - Fully Filled (完全成交): 25
  - Partially Filled (部分成交): 10
  - Unfilled (未成交): 15

成交率 (Fill Rate):
  - By Quantity (按数量): 70.00%
  - By Order Count (按订单数): 50.00%
============================================================
```

### 字段含义详解

#### 回测运行结果 (Backtest Execution Results)

| 输出字段 | 中文含义 | 说明 |
|----------|----------|------|
| intervals | 处理的快照区间数 | 回测过程中处理的市场快照区间总数，每个区间代表两个相邻快照之间的时间段 |
| final_time | 最终时间戳 | 回测结束时的时间戳（tick单位，每tick=100纳秒） |

#### 诊断信息 (Diagnostics)

| 输出字段 | 中文含义 | 说明 |
|----------|----------|------|
| intervals_processed | 已处理区间数 | 实际处理完成的快照区间数量 |
| orders_submitted | 提交的订单数 | 策略向交易所提交的订单总数（包括成交、撤单、拒绝等所有状态） |
| orders_filled | 成交的订单数 | 成功成交（包括部分成交和全部成交）的订单数 |
| receipts_generated | 产生的回执数 | 交易所返回的回执总数，包括成交、撤单、拒绝等各类回执 |
| cancels_submitted | 提交的撤单数 | 策略发起的撤单请求数量 |

#### 投资组合摘要 (Portfolio Summary)

| 输出字段 | 中文含义 | 说明 |
|----------|----------|------|
| Cash | 现金余额 | 回测结束时账户中的现金金额 |
| Position | 持仓数量 | 回测结束时持有的标的数量（正数为多头，负数为空头） |
| Realized PnL | 已实现盈亏 | 已经平仓的交易产生的盈亏总额 |

#### 订单摘要 (Order Summary)

| 输出字段 | 中文含义 | 说明 |
|----------|----------|------|
| Total orders | 总订单数 | 回测期间创建的所有订单数量 |
| Active orders | 活跃订单数 | 回测结束时仍在挂单队列中等待成交的订单数 |
| Filled orders | 已成交订单数 | 完全成交（FILLED状态）的订单数量 |

#### 回执记录器摘要 (Receipt Logger Summary)

| 输出字段 | 中文含义 | 说明 |
|----------|----------|------|
| Total Receipts | 总回执数 | 收到的所有回执数量（一个订单可能产生多个回执） |
| Total Orders | 总订单数 | 已注册到回执记录器的订单数量 |
| Total Order Qty | 总下单数量 | 所有订单的下单数量之和 |
| Total Filled Qty | 总成交数量 | 所有订单的实际成交数量之和 |

#### 回执类型分布 (Receipt Type Distribution)

| 输出字段 | 中文含义 | 说明 |
|----------|----------|------|
| Partial Fill | 部分成交 | 订单部分成交的回执数量 |
| Full Fill | 全部成交 | 订单完全成交的回执数量 |
| Canceled | 已撤单 | 订单被撤销的回执数量 |
| Rejected | 已拒绝 | 订单被交易所拒绝的回执数量 |

#### 订单状态分布 (Order Status Distribution)

| 输出字段 | 中文含义 | 说明 |
|----------|----------|------|
| Fully Filled | 完全成交 | 下单数量全部成交的订单数 |
| Partially Filled | 部分成交 | 仅部分成交（成交数量 < 下单数量）的订单数 |
| Unfilled | 未成交 | 一手都未成交的订单数 |

#### 成交率 (Fill Rate)

| 输出字段 | 中文含义 | 计算公式 | 说明 |
|----------|----------|----------|------|
| By Quantity | 按数量 | 总成交数量 / 总下单数量 | 反映下单数量中有多大比例被成交 |
| By Order Count | 按订单数 | 完全成交订单数 / 总订单数 | 反映提交的订单中有多大比例被完全成交 |

### 如何理解这些指标

1. **成交效率分析**
   - 如果"按数量成交率"高但"按订单数成交率"低，说明大订单容易成交，小订单不容易成交
   - 如果两个成交率都低，可能需要调整下单策略或订单价格

2. **订单管理分析**
   - 活跃订单数较多可能意味着订单挂单时间过长
   - 撤单数量较多可能需要优化下单时机

3. **盈亏分析**
   - Realized PnL 只反映已平仓的盈亏
   - 要计算总盈亏，还需考虑 Position 的浮动盈亏
