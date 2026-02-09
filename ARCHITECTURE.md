# BackTestSys 系统设计图

本文档描述回测系统的整体架构设计，包括模块结构、组件交互、数据流和订单生命周期。

---

## 1. 系统模块总览

```
quant_framework/
├── core/               # 核心抽象层：类型、接口、事件、DTO
│   ├── types.py            # 基础类型与数据结构
│   ├── interfaces.py       # 8大核心接口定义
│   ├── events.py           # 仿真事件类型
│   ├── dto.py              # 数据传输对象 (SnapshotDTO, ReadOnlyOMSView)
│   ├── data_loader.py      # 行情数据加载器 (CSV / Pickle / 补帧)
│   └── trading_hours.py    # 交易时段辅助
├── tape/               # Tape构建层
│   └── builder.py          # UnifiedTapeBuilder — 从快照重建执行路径
├── market/             # 行情结构层
│   ├── book.py             # BookView — 订单簿视图
│   └── tape.py             # TradeTapeReconstructor — 成交带重建
├── exchange/           # 交易所模拟层
│   └── simulator.py        # FIFOExchangeSimulator — FIFO队列撮合引擎
├── trading/            # 交易管理层
│   ├── oms.py              # OrderManager + Portfolio — 订单与持仓管理
│   ├── strategy.py         # SimpleStrategy — 策略基类
│   ├── replay_strategy.py  # ReplayStrategy — 历史订单回放策略
│   └── receipt_logger.py   # ReceiptLogger — 回执记录与诊断
├── runner/             # 运行引擎层
│   └── event_loop.py       # EventLoopRunner — 事件循环主引擎
├── analysis/           # 分析层
│   └── metrics.py          # ExecutionQualityMetrics — 执行质量指标
└── config.py           # 配置管理 (XML / YAML / JSON)
```

---

## 2. 核心组件架构图

```mermaid
graph TB
    subgraph 入口层
        MAIN["main.py<br/>程序入口"]
        CFG["config.py<br/>配置加载"]
    end

    subgraph 运行引擎层
        EL["EventLoopRunner<br/>事件循环主引擎<br/>（优先级队列调度）"]
    end

    subgraph 数据层
        CSV["CSV / Pickle<br/>原始行情文件"]
        FEED["IMarketDataFeed<br/>行情数据源"]
        DUP["SnapshotDuplicatingFeed<br/>快照补帧器"]
    end

    subgraph Tape构建层
        TB["UnifiedTapeBuilder<br/>Tape构建器<br/>（纯函数 · 无状态）"]
        TTR["TradeTapeReconstructor<br/>成交带重建器"]
    end

    subgraph 交易所模拟层
        EX["FIFOExchangeSimulator<br/>FIFO队列撮合引擎"]
        QM["IQueueModel<br/>队列模型"]
    end

    subgraph 交易管理层
        OMS["OrderManager<br/>订单管理器"]
        PORT["Portfolio<br/>持仓 · 资金 · PnL"]
        STRAT["IStrategy<br/>交易策略"]
        RL["ReceiptLogger<br/>回执记录器"]
    end

    subgraph 分析层
        MET["ExecutionQualityMetrics<br/>执行质量分析"]
    end

    MAIN -->|加载| CFG
    MAIN -->|创建并运行| EL
    CFG -->|配置参数| EL

    CSV -->|读取| FEED
    FEED -->|原始快照| DUP
    DUP -->|补帧后的<br/>NormalizedSnapshot流| EL

    EL -->|"快照对 (A, B)"| TB
    TB -->|使用| TTR
    TB -->|"TapeSegment[]"| EL

    EL -->|"TapeSegment"| EX
    EX -->|使用| QM
    EX -->|"OrderReceipt[]"| EL

    EL -->|提交订单| OMS
    EL -->|回执| OMS
    OMS -->|持仓更新| PORT

    EL -->|"SnapshotDTO +<br/>ReadOnlyOMSView"| STRAT
    STRAT -->|"Order[]"| EL
    EL -->|"OrderReceipt"| STRAT

    EL -->|回执| RL
    RL -->|CSV / 控制台| MET
```

---

## 3. 核心接口关系图

```mermaid
classDiagram
    class IMarketDataFeed {
        <<interface>>
        +next() NormalizedSnapshot?
        +reset()
    }

    class ITapeBuilder {
        <<interface>>
        +build(prev, curr) List~TapeSegment~
    }

    class ITradeTapeReconstructor {
        <<interface>>
        +reconstruct(prev, curr) List~Tuple~
    }

    class IExchangeSimulator {
        <<interface>>
        +reset()
        +on_order_arrival(order, time, qty) OrderReceipt?
        +on_cancel_arrival(order_id, time) OrderReceipt
        +advance(t_from, t_to, segment) Tuple
        +align_at_boundary(snapshot)
        +get_queue_depth(side, price) Qty
    }

    class IQueueModel {
        <<interface>>
        +init_order(order, level_qty)
        +advance_on_trade(order, px, qty, before, after) Qty
        +advance_on_quote(order, before, after)
    }

    class IStrategy {
        <<interface>>
        +on_snapshot(snapshot_dto, oms_view) List~Order~
        +on_receipt(receipt, snapshot_dto, oms_view) List~Order~
    }

    class IOrderManager {
        <<interface>>
        +submit(order, time)
        +on_receipt(receipt)
        +get_active_orders() List~Order~
        +get_order(id) Order?
        +get_pending_orders() List~Order~
        +get_arrived_orders() List~Order~
        +mark_order_arrived(id, time)
    }

    class EventLoopRunner {
        -feed: IMarketDataFeed
        -tape_builder: ITapeBuilder
        -exchange: IExchangeSimulator
        -strategy: IStrategy
        -oms: IOrderManager
        +run() dict
    }

    class UnifiedTapeBuilder {
        -trade_reconstructor: ITradeTapeReconstructor
    }

    class FIFOExchangeSimulator {
        -queue_model: IQueueModel
    }

    EventLoopRunner --> IMarketDataFeed : 使用
    EventLoopRunner --> ITapeBuilder : 使用
    EventLoopRunner --> IExchangeSimulator : 使用
    EventLoopRunner --> IStrategy : 使用
    EventLoopRunner --> IOrderManager : 使用
    UnifiedTapeBuilder ..|> ITapeBuilder : 实现
    UnifiedTapeBuilder --> ITradeTapeReconstructor : 使用
    FIFOExchangeSimulator ..|> IExchangeSimulator : 实现
    FIFOExchangeSimulator --> IQueueModel : 使用
```

---

## 4. 事件循环处理流程

```mermaid
flowchart TD
    START([开始回测]) --> LOAD[加载配置 & 初始化组件]
    LOAD --> SNAP{读取下一个快照?}

    SNAP -->|有数据| BUILD["Tape构建<br/>build(prev, curr) → TapeSegment[]"]
    SNAP -->|无数据| FINISH([输出统计结果并结束])

    BUILD --> SCHED["调度事件到优先级队列<br/>· SEGMENT_END<br/>· ORDER_ARRIVAL (+delay_out)<br/>· CANCEL_ARRIVAL (+delay_out)"]

    SCHED --> LOOP{事件队列非空?}

    LOOP -->|是| DEQUEUE[按优先级出队事件]
    LOOP -->|否| BOUNDARY["边界对齐<br/>align_at_boundary()"]

    DEQUEUE --> ETYPE{事件类型?}

    ETYPE -->|SEGMENT_END| ADV["exchange.advance()<br/>推进撮合 → OrderReceipt[]"]
    ETYPE -->|ORDER_ARRIVAL| OA["exchange.on_order_arrival()<br/>订单进入交易所队列"]
    ETYPE -->|CANCEL_ARRIVAL| CA["exchange.on_cancel_arrival()<br/>撤单处理"]
    ETYPE -->|RECEIPT_TO_STRATEGY| RTS["strategy.on_receipt()<br/>oms.on_receipt()<br/>receipt_logger.log()"]
    ETYPE -->|INTERVAL_END| IE["strategy.on_snapshot()<br/>收集新订单"]

    ADV -->|生成回执| SCHED_R["调度 RECEIPT_TO_STRATEGY<br/>(+delay_in)"]
    OA --> LOOP
    CA -->|生成回执| SCHED_R
    SCHED_R --> LOOP
    RTS -->|新订单| SCHED_O["调度 ORDER_ARRIVAL<br/>(+delay_out)"]
    SCHED_O --> LOOP
    IE -->|新订单| SCHED_O2["调度 ORDER_ARRIVAL<br/>(+delay_out)"]
    SCHED_O2 --> LOOP

    BOUNDARY --> SNAP
```

---

## 5. 订单生命周期

```mermaid
stateDiagram-v2
    [*] --> NEW : strategy 创建订单
    NEW --> PENDING : OMS.submit()
    PENDING --> LIVE : 订单到达交易所<br/>(ORDER_ARRIVAL 事件)
    LIVE --> PARTIALLY_FILLED : 部分成交<br/>(exchange.advance)
    PARTIALLY_FILLED --> PARTIALLY_FILLED : 继续部分成交
    PARTIALLY_FILLED --> FILLED : 全部成交
    LIVE --> FILLED : 一次性全部成交
    LIVE --> CANCELED : 撤单到达<br/>(CANCEL_ARRIVAL 事件)
    PARTIALLY_FILLED --> CANCELED : 撤单到达
    NEW --> REJECTED : 交易所拒绝
    FILLED --> [*]
    CANCELED --> [*]
    REJECTED --> [*]
```

---

## 6. 数据流水线

```mermaid
flowchart LR
    subgraph 数据输入
        RAW["原始行情文件<br/>(CSV / Pickle)"]
    end

    subgraph 数据预处理
        LOADER["CsvMarketDataFeed /<br/>PickleMarketDataFeed"]
        DUP["SnapshotDuplicatingFeed<br/>（按交易时段补帧）"]
    end

    subgraph Tape构建
        RECON["TradeTapeReconstructor<br/>重建成交带"]
        BUILDER["UnifiedTapeBuilder<br/>构建 TapeSegment"]
    end

    subgraph 撮合仿真
        EXCH["FIFOExchangeSimulator<br/>FIFO 队列撮合"]
    end

    subgraph 输出
        RECEIPT["OrderReceipt<br/>成交/撤单回执"]
        LOG["ReceiptLogger<br/>CSV + 控制台"]
        METRICS["执行质量指标"]
    end

    RAW --> LOADER --> DUP
    DUP -->|"NormalizedSnapshot 流"| BUILDER
    RECON -->|"成交带 [(Price, Qty)]"| BUILDER
    BUILDER -->|"TapeSegment[]"| EXCH
    EXCH --> RECEIPT --> LOG --> METRICS
```

---

## 7. 事件优先级

| 优先级 | 事件类型 | 说明 |
|:---:|:---|:---|
| 1 | `SEGMENT_END` | 段结束 — 先完成内部撮合 |
| 2 | `ORDER_ARRIVAL` | 订单到达交易所 |
| 3 | `CANCEL_ARRIVAL` | 撤单到达交易所 |
| 4 | `RECEIPT_TO_STRATEGY` | 回执到达策略 |
| 5 | `INTERVAL_END` | 区间结束 — 边界对齐与快照回调 |

---

## 8. 双时间线与延迟模型

```mermaid
sequenceDiagram
    participant S as Strategy<br/>(策略端)
    participant EL as EventLoop<br/>(事件引擎)
    participant EX as Exchange<br/>(交易所)

    Note over S,EX: delay_out: 策略→交易所延迟<br/>delay_in: 交易所→策略延迟

    S->>EL: 提交订单 (t₀)
    EL->>EX: ORDER_ARRIVAL (t₀ + delay_out)
    EX->>EX: 排入FIFO队列
    Note over EX: 等待撮合...
    EX->>EL: OrderReceipt (成交时间 tₓ)
    EL->>S: RECEIPT_TO_STRATEGY (tₓ + delay_in)
    S->>S: on_receipt() 处理回执
```

---

## 9. 配置体系

```mermaid
graph LR
    subgraph 配置来源
        XML["config.xml<br/>(主配置文件)"]
        CLI["命令行参数<br/>(--debug, --progress 等)"]
        ENV["环境变量"]
        DEF["数据类默认值"]
    end

    subgraph 配置对象
        BC["BacktestConfig"]
        DC["DataConfig"]
        TC["TapeConfig"]
        EC["ExchangeConfig"]
        RC["RunnerConfig"]
        PC["PortfolioConfig"]
        SC["StrategyConfig"]
        LC["LoggingConfig"]
    end

    DEF -->|最低优先级| BC
    XML -->|覆盖默认值| BC
    CLI -->|覆盖文件配置| BC
    ENV -->|覆盖文件配置| BC

    BC --> DC
    BC --> TC
    BC --> EC
    BC --> RC
    BC --> PC
    BC --> SC
    BC --> LC
```

---

## 10. 关键设计原则

| 设计原则 | 说明 |
|:---|:---|
| **事件驱动架构** | 优先级队列调度，精确模拟时序 |
| **DTO模式** | SnapshotDTO / ReadOnlyOMSView 隔离策略与内部状态 |
| **纯函数Tape构建** | 相同输入 → 相同输出，便于测试和调试 |
| **接口驱动** | 8大抽象接口实现松耦合 |
| **关注点分离** | 数据源 → Tape → 交易所 → OMS → 日志，职责明确 |
| **只读视图** | ReadOnlyOMSView 防止策略意外修改系统状态 |
| **双时间线** | exchtime / recvtime 独立建模，真实反映延迟 |

---

## 11. 开发迭代流程

### 11.1 总体开发流程

```mermaid
flowchart TD
    START([确定迭代目标]) --> WHICH{迭代类型?}

    WHICH -->|开发新策略| S1["① 策略开发"]
    WHICH -->|接入新数据源| S2["② 数据源扩展"]
    WHICH -->|改进撮合模型| S3["③ 撮合模型定制"]
    WHICH -->|修改系统行为| S4["④ 框架核心修改"]

    S1 --> IMPL[编写实现代码]
    S2 --> IMPL
    S3 --> IMPL
    S4 --> IMPL

    IMPL --> TEST["运行单元测试<br/>python test_framework.py"]
    TEST --> PASS{测试通过?}
    PASS -->|否| IMPL
    PASS -->|是| RUN["运行回测验证<br/>python main.py --config config.xml"]
    RUN --> CHECK{结果符合预期?}
    CHECK -->|否| IMPL
    CHECK -->|是| DONE([迭代完成])
```

---

### 11.2 策略开发流程（最常见的迭代场景）

```mermaid
flowchart TD
    A([开始开发新策略]) --> B["1. 创建策略文件<br/>quant_framework/trading/my_strategy.py"]
    B --> C["2. 继承 IStrategy 接口<br/>实现 on_snapshot() 和 on_receipt()"]
    C --> D["3. 在 main.py 中注册策略<br/>修改 create_strategy() 工厂函数"]
    D --> E["4. 配置 config.xml<br/>设置 strategy.name 和 strategy.params"]
    E --> F["5. 编写测试用例<br/>在 test_framework.py 中添加测试"]
    F --> G["6. 运行测试<br/>python test_framework.py"]
    G --> H{测试通过?}
    H -->|否| C
    H -->|是| I["7. 运行完整回测<br/>python main.py --debug --verbose-receipts"]
    I --> J["8. 分析结果<br/>查看回执日志 / 执行质量指标"]
    J --> K{策略表现满意?}
    K -->|否| C
    K -->|是| L([策略开发完成])
```

**策略接口说明：**

```python
# quant_framework/trading/my_strategy.py
from typing import List
from quant_framework.core.interfaces import IStrategy
from quant_framework.core.types import Order, Side, OrderReceipt
from quant_framework.core.dto import SnapshotDTO, ReadOnlyOMSView

class MyStrategy(IStrategy):
    def on_snapshot(self, snapshot: SnapshotDTO, oms_view: ReadOnlyOMSView) -> List[Order]:
        """每次收到行情快照时调用。
        
        可用数据：
        - snapshot.best_bid / snapshot.best_ask  → 最优买卖价
        - snapshot.bids / snapshot.asks           → 多档盘口
        - snapshot.ts_recv                        → 接收时间戳
        - oms_view.get_active_orders()            → 当前活跃订单
        - oms_view.get_portfolio()                → 持仓和资金
        
        返回值：要提交的新订单列表（空列表 = 不操作）
        """
        ...

    def on_receipt(self, receipt: OrderReceipt, snapshot: SnapshotDTO,
                   oms_view: ReadOnlyOMSView) -> List[Order]:
        """每次收到订单回执（成交/撤单/拒绝）时调用。
        
        可用数据：
        - receipt.receipt_type  → "FILL" / "PARTIAL" / "CANCELED" / "REJECTED"
        - receipt.fill_qty      → 本次成交数量
        - receipt.fill_price    → 成交价格
        
        返回值：要提交的新订单列表
        """
        ...
```

---

### 11.3 各迭代场景速查表

| 迭代目标 | 需要修改的文件 | 涉及的接口 | 注册/接入方式 |
|:---|:---|:---|:---|
| **新增交易策略** | `trading/my_strategy.py` (新建) | `IStrategy` | 修改 `main.py` 中 `create_strategy()` |
| **接入新数据格式** | `core/data_loader.py` (新增类) | `IMarketDataFeed` | 修改 `main.py` 中 `create_feed()` |
| **自定义队列模型** | `exchange/simulator.py` (新增类) | `IQueueModel` | 修改 `main.py` 中 `create_exchange()` |
| **修改Tape构建逻辑** | `tape/builder.py` | `ITapeBuilder` | 修改 `main.py` 中 `create_tape_builder()` |
| **新增成交带重建器** | `market/tape.py` (新增类) | `ITradeTapeReconstructor` | 传入 `UnifiedTapeBuilder` 构造函数 |
| **调整配置参数** | `config.xml` | 无需代码修改 | 直接编辑配置文件或使用命令行覆盖 |
| **新增分析指标** | `analysis/metrics.py` | 无固定接口 | 在回测结束后处理结果 |
| **调整回测参数** | 命令行 | — | `--debug` / `--progress` / `--data` 等 |

---

### 11.4 数据源扩展流程

```mermaid
flowchart TD
    A([需要接入新数据格式]) --> B["1. 在 core/data_loader.py 中<br/>新建类实现 IMarketDataFeed"]
    B --> C["2. 实现 next() 方法<br/>返回 NormalizedSnapshot 或 None"]
    C --> D["3. 实现 reset() 方法<br/>支持多次回测"]
    D --> E["4. （可选）用 SnapshotDuplicatingFeed 包装<br/>自动补帧处理时间间隙"]
    E --> F["5. 修改 main.py 的 create_feed()<br/>支持新的 data.format 配置"]
    F --> G["6. 编写测试验证<br/>数据加载 → 快照格式正确"]
    G --> H([完成])
```

**NormalizedSnapshot 必须包含的字段：**

```python
NormalizedSnapshot(
    ts_recv=...,          # 接收时间 (tick)
    ts_exch=...,          # 交易所时间 (tick, 可选)
    bids=[(price, qty)],  # 买盘列表 (Level)
    asks=[(price, qty)],  # 卖盘列表 (Level)
    last=...,             # 最新价 (可选)
    volume=...,           # 累计成交量 (可选)
    turnover=...,         # 累计成交额 (可选)
    last_vol_split=...,   # 分价位成交量 (可选)
)
```

---

### 11.5 撮合模型定制流程

```mermaid
flowchart TD
    A([需要自定义队列行为]) --> B["1. 在 exchange/simulator.py 中<br/>新建类实现 IQueueModel"]
    B --> C["2. 实现 init_order()<br/>初始化订单在队列中的位置"]
    C --> D["3. 实现 advance_on_trade()<br/>成交时推进队列位置"]
    D --> E["4. （可选）实现 advance_on_quote()<br/>盘口变化时更新位置"]
    E --> F["5. 传入 FIFOExchangeSimulator 构造函数<br/>或创建自定义 IExchangeSimulator"]
    F --> G["6. 编写测试验证<br/>队列推进逻辑 + 成交计算"]
    G --> H([完成])
```

---

### 11.6 测试与验证流程

```mermaid
flowchart LR
    subgraph 单元测试
        UT["python test_framework.py<br/>验证各组件独立行为"]
    end

    subgraph 集成验证
        IT["python main.py --config config.xml<br/>--debug --verbose-receipts<br/>完整回测流程"]
    end

    subgraph 结果分析
        RA["查看输出统计<br/>· 成交率 · PnL<br/>· 订单状态分布"]
        LOG["查看回执日志<br/>--save-receipts output.csv"]
        CFG["查看配置<br/>--show-config"]
    end

    UT --> IT --> RA
    IT --> LOG
    IT --> CFG
```

**常用验证命令：**

```bash
# 运行单元测试
python test_framework.py

# 查看当前配置（不执行回测）
python main.py --show-config

# 带调试信息运行回测
python main.py --config config.xml --debug --progress

# 输出详细回执到控制台
python main.py --verbose-receipts

# 保存回执到文件
python main.py --save-receipts output/receipts.csv

# 指定数据文件运行
python main.py --data path/to/market_data.pkl
```

---

### 11.7 迭代开发检查清单

开发新功能时，按以下清单逐步验证：

- [ ] **接口一致性** — 新组件是否正确实现了对应接口的所有方法？
- [ ] **DTO隔离** — 策略是否只通过 SnapshotDTO / ReadOnlyOMSView 访问数据？
- [ ] **工厂注册** — main.py 中的工厂函数是否已更新？
- [ ] **配置支持** — 新参数是否已添加到 config.py 的数据类中？
- [ ] **单元测试** — 是否在 test_framework.py 中添加了对应测试？
- [ ] **回测验证** — 完整回测是否能正常运行并输出合理结果？
- [ ] **文档更新** — ARCHITECTURE.md 和 CONFIG.md 是否已同步更新？
