# X坐标轴队列模型设计文档

## 目录
1. [什么是 `level.x_coord`](#什么是-levelx_coord)
2. [详细示例](#详细示例)
3. [为什么采用这种设计](#为什么采用这种设计)
4. [X轴的含义](#x轴的含义)
5. [与直接消耗Shadow Order的对比](#与直接消耗shadow-order的对比)

---

## 什么是 `level.x_coord`

`level.x_coord` 是**坐标轴队列模型**中的核心概念，表示某个价格档位的**累计队列前端消耗量**（Cumulative Front Consumption）。

在代码中的定义（`quant_framework/exchange/simulator.py`）：

```python
@dataclass
class PriceLevelState:
    """State for a single price level using coordinate-axis model."""
    side: Side
    price: Price
    q_mkt: float = 0.0     # 公开市场队列深度
    x_coord: float = 0.0   # 累计前端消耗量 ← 这就是 x_coord
    queue: List[ShadowOrder] = field(default_factory=list)
```

### 核心公式

```
X_s(p,t) = 累计消耗量 = Σ(成交量 + φ × 撤单量)
```

其中：
- `s`: 方向（bid/ask）
- `p`: 价格
- `t`: 时间
- `φ` (代码中为 `cancel_front_ratio`): 撤单中从队列前端离开的比例（默认0.5）

---

## 详细示例

### 场景设定

假设在bid侧（买方），价格档位 100.00 的队列初始状态：

```
时刻 T=0 时:
- 市场队列深度 Q_mkt = 1000 手
- 累计消耗量 x_coord = 0
- 尾部坐标 Tail = x_coord + Q_mkt = 0 + 1000 = 1000
```

### 第一步：下单入队

你在 T=1 时下了一个买单（100手），系统如何处理？

```
你的订单到达时:
- 当前 Tail = 0 + 1000 = 1000
- 你前面已有的 shadow order 数量 = 0
- 你的队列位置 pos = Tail + 0 = 1000

订单状态:
  - order_id: "ORD001"
  - pos: 1000        ← 你在坐标轴上的起始位置
  - qty: 100         ← 你占据 [1000, 1100) 区间
  - 成交条件: 当 x_coord >= 1100 时完全成交
```

**图解：**
```
X轴（队列位置坐标）:
0 -------- 1000 -------- 1100
|  市场队列  |  你的订单  |
|  (Q_mkt)  | [pos, pos+qty) |
           ↑
        x_coord=0 (队列消耗起点)
```

### 第二步：市场消耗

假设在 T=1 到 T=10 期间，该价格档位发生了：
- 成交量 M = 800 手
- 撤单量 C = 400 手
- φ = 0.5 (假设50%的撤单来自队列前端)

X坐标更新：
```
x_coord增量 = M + φ × C = 800 + 0.5 × 400 = 1000
新的 x_coord = 0 + 1000 = 1000
```

**图解：**
```
X轴（队列位置坐标）:
0 -------- 1000 -------- 1100
           ↑              ↑
        x_coord=1000   你的订单结束位置
           |
     消耗已经到达你的起始位置！
```

此时你的订单已经到达队列最前端，但还没成交（x_coord = pos = 1000）。

### 第三步：继续消耗直到成交

假设在 T=10 到 T=15 期间，继续发生：
- 成交量 M = 100 手
- 撤单量 C = 0 手

X坐标更新：
```
x_coord增量 = 100 + 0.5 × 0 = 100
新的 x_coord = 1000 + 100 = 1100
```

**成交判断：**
```python
# 代码中的成交判断逻辑
threshold = shadow.pos + shadow.original_qty  # 1000 + 100 = 1100
if x_coord >= threshold:  # 1100 >= 1100 ✓
    # 订单完全成交！
```

**最终图解：**
```
X轴（队列位置坐标）:
0 -------- 1000 -------- 1100
                         ↑
                    x_coord=1100
                    
你的订单区间 [1000, 1100) 已被完全消耗
→ 订单完全成交！
```

---

## 为什么采用这种设计

### 核心优势：No-Impact Assumption（零影响假设）

这种坐标轴模型的设计允许我们实现**无影响回测**——即假设你的订单不会影响市场的真实队列状态。

#### 问题场景

想象一下如果有**多个shadow order**在同一档位：

```
时刻 T=0:
- x_coord = 0
- Q_mkt = 1000
- Shadow Order A: pos=1000, qty=50 (占据 [1000, 1050))
- Shadow Order B: pos=1050, qty=100 (占据 [1050, 1150))
- Shadow Order C: pos=1150, qty=200 (占据 [1150, 1350))
```

当市场消耗发生时，所有订单的成交状态可以通过**一个简单的比较**来判断：

```python
# 对于任意订单
if x_coord >= shadow.pos + shadow.remaining_qty:
    # 该订单成交
```

### 优点

1. **O(1) 成交判断**：无需遍历队列，只需比较坐标
2. **独立计算**：每个订单的成交与否只依赖于 `x_coord` 和自己的 `pos`
3. **支持并发订单**：多个订单可以同时存在，互不干扰
4. **撤单处理简单**：只需标记订单状态，无需调整其他订单位置
5. **部分成交精确计算**：`filled_qty = max(0, min(qty, x_coord - pos))`

---

## X轴的含义

### 本质：时间驱动的累计消耗量

X轴代表的是**从队列前端累计消耗的总量**，可以理解为：

```
X坐标 = 历史上所有"离开队列前端"的订单总量
     = 成交量 + (有效的前端撤单量)
```

### 为什么叫"坐标轴"？

这是一个**一维坐标系统**，其中：

- **原点 (0)**：队列建立时的起点
- **X坐标**：随时间增长，表示累计消耗
- **Tail坐标** = X + Q_mkt：队列尾部的当前位置
- **Shadow Order位置**：每个订单占据 [pos, pos+qty) 区间

```
时间轴方向 →

            X增长方向
            ─────────────────→
            
0 ──────── X_coord ──────── Tail ──────── ∞
|           |                |
|  已消耗区域  |   市场队列     |  未来订单区域
|           |   (Q_mkt)     |
            ↑                ↑
       队列消耗前沿       队列尾部
```

### 物理意义

把它想象成一条**传送带**：
- 传送带从左向右移动
- 队列中的订单像物品一样放在传送带上
- `x_coord` 是传送带已经移动的距离
- 当传送带移动超过订单位置时，订单就被送出（成交）

---

## 与直接消耗Shadow Order的对比

### 你的问题
> "每次消耗直接消耗shadow order不完了吗？"

### 直接消耗方法的问题

假设我们采用"直接消耗shadow order"的方法：

```python
# 假想的直接消耗实现
def consume_queue(level, trade_qty):
    for shadow in level.queue:
        if trade_qty <= 0:
            break
        fill = min(shadow.remaining_qty, trade_qty)
        shadow.remaining_qty -= fill
        trade_qty -= fill
```

**问题1：违反零影响假设**

```
场景：
- 市场队列有 1000 手
- 你的 shadow order 有 100 手

真实世界：市场成交 1100 手时，你的100手应该全部成交
直接消耗：市场成交 1000 手时，你的订单就开始被消耗了！

问题：你的订单抢占了本应属于市场队列的成交量
```

**问题2：无法准确计算成交时间**

坐标轴模型可以精确计算成交时刻：
```python
# 找到 x_coord >= threshold 的精确时刻
# 可以通过 segment 的 rate 来插值计算
fill_time = seg.t_start + (threshold - x_at_start) / rate
```

直接消耗方法很难做到这一点。

**问题3：多订单情况复杂**

```
有 3 个 shadow order: A, B, C
直接消耗需要决定：
- 谁先被消耗？
- 按什么比例消耗？
- A 成交后是否影响 B 的位置？

坐标轴模型：
- 每个订单有明确的位置
- 成交判断相互独立
- 无需维护复杂的消耗逻辑
```

### 对比总结

| 特性 | 坐标轴模型 | 直接消耗方法 |
|------|-----------|-------------|
| 零影响假设 | ✓ 天然满足 | ✗ 违反 |
| 成交时间精度 | ✓ 精确插值 | ✗ 只有segment粒度（即只能知道在哪个时间段成交，无法精确到具体时刻） |
| 多订单处理 | ✓ 独立计算 | ✗ 需要复杂的优先级 |
| 代码复杂度 | ✓ 简单的坐标比较 | ✗ 需要维护消耗状态 |
| 部分成交 | ✓ `min(qty, x - pos)` | ✗ 需要跟踪消耗进度 |

---

## 代码中的关键实现

### 1. 计算X坐标（简化示意）
```python
def _get_x_coord(self, side: Side, price: Price, t: int) -> float:
    """获取时刻t的X坐标"""
    level = self._get_level(side, price)
    x = level.x_coord  # 基础值
    
    for seg_idx, seg in enumerate(self._current_tape):
        seg_start = seg.t_start
        seg_end = min(seg.t_end, t)  # 截止到时刻t
        
        # 累加每个segment的消耗
        rate = self._x_rates.get((side, round(price, 8), seg_idx), 0.0)
        x += rate * (seg_end - seg_start)
    
    return x
```

### 2. 成交判断（概念）
```python
def _compute_fill_time(self, shadow: ShadowOrder, qty_to_fill: int):
    threshold = shadow.pos + qty_to_fill
    # 在每个segment中：
    # 1. 计算当前x坐标和rate
    # 2. 如果 x + rate * dt >= threshold，则计算精确成交时刻
    # 3. fill_time = seg.t_start + (threshold - x_at_start) / rate
```

### 3. 订单入队（概念示意）
```python
def _queue_order(self, order, arrival_time, market_qty, remaining_qty, ...):
    level = self._get_level(side, price)
    
    # 概念上的队列位置计算：
    # Tail = X + Q_mkt（队列尾部坐标）
    # 订单位置 = Tail + 前面的shadow order总量
    
    # 实际代码中的实现：
    # q_mkt_t = 当前时刻的市场队列深度（通过_get_q_mkt计算）
    # s_shadow = 前面的shadow order总量
    # pos = int(round(q_mkt_t + s_shadow))
    #
    # 注：当level刚创建时 x_coord=0，所以 pos ≈ Tail + s_shadow
```

---

## 结论

`level.x_coord` 是坐标轴队列模型的核心，它：

1. **追踪累计消耗**：记录从队列前端离开的总量
2. **支持零影响回测**：shadow order不影响市场队列
3. **简化成交计算**：只需比较坐标即可判断成交
4. **精确时间计算**：配合segment rate可以插值计算成交时刻

这种设计是量化回测系统中的标准做法，确保了回测结果的准确性和可重复性。
