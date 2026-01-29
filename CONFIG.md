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
cp config.yaml my_config.yaml
```

2. 编辑配置文件，修改所需参数

3. 运行回测：
```bash
# 使用默认配置文件 (config.yaml)
python main.py

# 使用自定义配置文件
python main.py --config my_config.yaml

# 显示配置信息
python main.py --show-config
```

## 配置文件格式

系统支持 YAML 和 JSON 两种配置文件格式。

### YAML 格式 (推荐)
```yaml
data:
  path: "data/sample.pkl"

tape:
  ghost_rule: "symmetric"
  epsilon: 1.0
```

### JSON 格式
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
```yaml
data:
  path: "data/my_market_data.pkl"
```

---

### Tape构建器配置 (tape)

Tape构建器负责从快照对构建事件Tape，是回测引擎的核心组件。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `ghost_rule` | string | `"symmetric"` | lastvolsplit到单边的映射规则 |
| `ghost_alpha` | float | `0.5` | 比例分配因子（仅当ghost_rule="proportion"时生效） |
| `epsilon` | float | `1.0` | 时间分配的基准权重，防止零长度段 |
| `segment_iterations` | int | `2` | 段迭代次数（已弃用，保留用于向后兼容） |
| `time_scale_lambda` | float | `0.0` | 时间缩放参数，控制事件早期/晚期分布 |
| `cancel_front_ratio` | float | `0.5` | 撤单前端比例 (φ)，控制撤单在队列前端的比例 |
| `crossing_order_policy` | string | `"passive"` | 穿越订单处理策略 |
| `top_k` | int | `5` | 跟踪的价格档位数 |
| `tick_size` | float | `1.0` | 最小价格变动单位 |

#### ghost_rule 详解

| 值 | 说明 |
|----|------|
| `"symmetric"` | 对称分配，lastvolsplit量平分给bid和ask侧 |
| `"proportion"` | 按比例分配，由ghost_alpha控制 |
| `"single_bid"` | 全部分配给bid侧 |
| `"single_ask"` | 全部分配给ask侧 |

#### cancel_front_ratio 详解

- `0.0` = 悲观假设（撤单全部发生在队列后端，对我们不利）
- `0.5` = 中性假设（撤单均匀分布）
- `1.0` = 乐观假设（撤单全部发生在队列前端，对我们有利）

#### crossing_order_policy 详解

| 值 | 说明 |
|----|------|
| `"reject"` | 拒绝穿越订单 |
| `"adjust"` | 自动调整价格到非穿越 |
| `"passive"` | 将穿越订单视为被动订单 |

**示例：**
```yaml
tape:
  ghost_rule: "symmetric"
  epsilon: 1.0
  cancel_front_ratio: 0.5
  crossing_order_policy: "passive"
  top_k: 5
  tick_size: 0.01  # 适用于股票等市场
```

---

### 交易所模拟器配置 (exchange)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `cancel_front_ratio` | float | `0.5` | 撤单前端比例，与tape配置含义相同 |

**示例：**
```yaml
exchange:
  cancel_front_ratio: 0.5
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
```yaml
runner:
  # 模拟100ms往返延迟
  delay_out: 500000   # 50ms 订单发送延迟
  delay_in: 500000    # 50ms 回执接收延迟
  show_progress: true
```

---

### 投资组合配置 (portfolio)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `initial_cash` | float | `100000.0` | 初始现金 |

**示例：**
```yaml
portfolio:
  initial_cash: 1000000.0  # 100万初始资金
```

---

### 策略配置 (strategy)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | string | `"SimpleStrategy"` | 策略名称 |
| `params.order_interval` | int | `10` | 下单间隔（每隔多少个快照下单） |
| `params.max_active_orders` | int | `5` | 最大活跃订单数 |

**示例：**
```yaml
strategy:
  name: "MyStrategy"
  params:
    order_interval: 5      # 每5个快照下单
    max_active_orders: 10  # 最多10个活跃订单
```

---

### 回执记录器配置 (receipt_logger)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `verbose` | bool | `false` | 是否实时打印回执到控制台 |
| `output_file` | string | `""` | 回执保存文件路径（留空则不保存） |

**示例：**
```yaml
receipt_logger:
  verbose: true
  output_file: "output/receipts.csv"
```

---

### 日志配置 (logging)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `debug` | bool | `false` | 是否启用debug模式 |
| `log_file` | string | `""` | 日志文件路径（留空则不保存到文件） |
| `level` | string | `"INFO"` | 日志级别（DEBUG, INFO, WARNING, ERROR） |

**示例：**
```yaml
logging:
  debug: true
  log_file: "output/backtest.log"
  level: "DEBUG"
```

---

### 快照处理配置 (snapshot)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `min_interval_tick` | int | `5000000` | 快照最小间隔（tick单位，500ms） |
| `tolerance_tick` | int | `100000` | 时间容差（tick单位，10ms） |

**示例：**
```yaml
snapshot:
  min_interval_tick: 5000000  # 500ms
  tolerance_tick: 100000      # 10ms
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
python main.py --config custom.yaml --progress --debug
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
export BACKTEST_CONFIG_PATH=/path/to/my_config.yaml
python main.py
```

---

## 配置示例

### 保守策略配置

```yaml
# conservative_config.yaml
data:
  path: "data/market.pkl"

tape:
  cancel_front_ratio: 0.0  # 悲观假设
  crossing_order_policy: "reject"

runner:
  delay_out: 1000000  # 100ms延迟
  delay_in: 1000000

portfolio:
  initial_cash: 50000.0

strategy:
  params:
    order_interval: 20
    max_active_orders: 3

logging:
  level: "INFO"
```

### 高频交易配置

```yaml
# hft_config.yaml
data:
  path: "data/tick_data.pkl"

tape:
  cancel_front_ratio: 0.5
  crossing_order_policy: "passive"
  tick_size: 0.01

runner:
  delay_out: 10000   # 1ms延迟
  delay_in: 10000
  show_progress: true

portfolio:
  initial_cash: 1000000.0

strategy:
  params:
    order_interval: 1
    max_active_orders: 20

logging:
  debug: true
  log_file: "output/hft_debug.log"
```

### 研究模式配置

```yaml
# research_config.yaml
data:
  path: "data/research_sample.pkl"

tape:
  ghost_rule: "symmetric"
  epsilon: 1.0
  cancel_front_ratio: 0.5

runner:
  delay_out: 0
  delay_in: 0
  show_progress: true

receipt_logger:
  verbose: true
  output_file: "output/research_receipts.csv"

logging:
  debug: true
  log_file: "output/research.log"
  level: "DEBUG"
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
config = load_config("config.yaml")

# 修改配置
config.data.path = "data/custom.pkl"
config.runner.show_progress = True
config.portfolio.initial_cash = 500000.0

# 打印配置
print_config(config)

# 保存配置
save_config(config, "my_config.yaml")

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
3. **参数验证**：系统会对部分参数进行基本验证
4. **向后兼容**：一些已弃用的参数仍然保留，以确保旧配置文件可用
5. **YAML依赖**：使用YAML格式需要安装PyYAML（`pip install pyyaml`）
