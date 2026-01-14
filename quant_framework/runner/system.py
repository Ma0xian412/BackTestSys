import os
import numpy as np
import pandas as pd
from typing import Callable, Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from threading import Thread, Event

from ..core.interfaces import IMarketDataFeed, ISimulationModel, IStrategy, IQueueModel
from ..core.events import EventType, SimulationEvent
from ..execution.engine import ReactiveExecutionEngine
from ..execution.policies import MakerPolicy, TakerPolicy
from ..trading.oms import OrderManager, Portfolio
from ..analysis.metrics import ExecutionQualityMetrics
from ..simulation.context import IntervalContext, OrderIntervalView, MarketIntervalView

# tqdm 是可选依赖：用于显示并行任务进度条
try:
    from tqdm.auto import tqdm  # type: ignore
except Exception:  # pragma: no cover
    tqdm = None  # type: ignore

# 类型定义：工厂函数
SimulationFactory = Callable[[int], ISimulationModel] 
# 接受 seed，返回 Model

# 用于跨进程汇报每个 run 的进度：
# ("progress", run_id, done_intervals, total_intervals)
# ("done", run_id, done_intervals, total_intervals)
ProgressMsg = Tuple[str, int, int, Optional[int]]

def _mc_worker_run(seed: int,
                   run_id: int,
                   order_latency: int,
                   feed_factory: Callable[[], IMarketDataFeed],
                   sim_factory: SimulationFactory,
                   queue_factory: Callable[[], IQueueModel],
                   strategy_factory: Callable[[], IStrategy],
                   output_dir: str = "outputs",
                   export_fills_csv: bool = True,
                   export_orders_csv: bool = True,
                   progress_queue: Any = None) -> Dict[str, Any]:
    """子进程入口：执行一次独立回测并返回结果 dict。

    注意：为便于跨进程传递，返回值使用纯 dict（避免在子进程里构造 DataFrame 的额外开销）。
    """
    feed = feed_factory()
    sim = sim_factory(seed)
    queue = queue_factory()
    strat = strategy_factory()

    # 计算该 run 的区间数（若 feed 支持 __len__）
    total_intervals: Optional[int] = None
    try:
        total_snapshots = len(feed)  # type: ignore[arg-type]
        total_intervals = max(0, int(total_snapshots) - 1)
    except Exception:
        total_intervals = None

    # 进度回调：在子进程内把进度通过队列回传给主进程
    def _progress_cb(done_intervals: int):
        if progress_queue is None:
            return
        try:
            progress_queue.put(("progress", int(run_id), int(done_intervals), total_intervals))
        except Exception:
            pass

    ctx = SingleRunContext(
        feed,
        sim,
        queue,
        strat,
        order_latency=order_latency,
        progress_callback=_progress_cb,
        progress_total=total_intervals,
    )
    res = ctx.run()
    # 导出逐笔成交/逐订单明细（每个 run 独立文件，避免并行冲突）
    fills_csv = None
    orders_csv = None
    try:
        os.makedirs(str(output_dir), exist_ok=True)
        if export_fills_csv:
            fills_csv = os.path.join(str(output_dir), f"fills_run_{run_id}.csv")
            ctx.metrics.export_fills_csv(fills_csv, run_id=run_id)
        if export_orders_csv:
            orders_csv = os.path.join(str(output_dir), f"orders_run_{run_id}.csv")
            ctx.metrics.export_orders_csv(orders_csv)
    except Exception:
        pass
    # 完成信号（用于主进程关闭该 run 的进度条）
    if progress_queue is not None:
        try:
            progress_queue.put(("done", int(run_id), int(ctx.last_progress), total_intervals))
        except Exception:
            pass
    res["run_id"] = run_id
    res["seed"] = int(seed)
    res["fills_csv"] = fills_csv
    res["orders_csv"] = orders_csv
    return res


class SingleRunContext:
    """一次独立的回测运行上下文"""
    def __init__(self, 
                 feed: IMarketDataFeed,
                 sim_model: ISimulationModel,
                 queue_model: IQueueModel,
                 strategy: IStrategy,
                 order_latency: int = 0,
                 progress_callback: Optional[Callable[[int], None]] = None,
                 progress_total: Optional[int] = None):
        
        self.feed = feed
        self.simulator = sim_model
        self.engine = ReactiveExecutionEngine(MakerPolicy(queue_model), TakerPolicy())
        self.portfolio = Portfolio()
        self.oms = OrderManager(self.portfolio)
        self.strategy = strategy
        # 下单到达交易所的延迟（与 ts_exch 同单位；示例数据通常为 ms 或 us）
        self.order_latency = int(order_latency)
        self.metrics = ExecutionQualityMetrics()

        # 进度汇报：done_intervals -> callback
        self.progress_callback = progress_callback
        self.progress_total = progress_total
        self.last_progress = 0
        
        # 绑定
        self.oms.subscribe_fill(self.metrics.on_fill)
        self.oms.subscribe_new(self.metrics.on_order_new)


    def get_portfolio_snapshot(self) -> Dict[str, Any]:
        """预留：账户/持仓/PnL 等指标接口（当前输出不默认包含这些字段）。"""
        return {
            "cash": getattr(self.portfolio, "cash", None),
            "position": getattr(self.portfolio, "position", None),
            "realized_pnl": getattr(self.portfolio, "realized_pnl", None),
        }

    def _collect_results(self) -> Dict[str, Any]:
        """汇总本次运行的指标输出（当前聚焦订单执行质量）。"""
        out: Dict[str, Any] = {}
        try:
            out.update(self.metrics.get_summary())
        except Exception:
            pass
        return out

    def run(self) -> Dict[str, Any]:
        """执行回测循环"""
        self.feed.reset()
        prev = self.feed.next()
        if prev is None:
            return self._collect_results()

        # 待到达交易所的订单：[(due_ts, Order), ...]
        pending_orders: List[tuple[int, Any]] = []

        # 先把第一帧快照作为 SNAPSHOT_ARRIVAL 注入，使策略能够在 A 看到行情并产生
        # “位于 AB 之间到达交易所”的挂单（通过 order_latency 体现）。
        init_ev = SimulationEvent(int(prev.ts_exch), EventType.SNAPSHOT_ARRIVAL, prev)
        self.metrics.update_time(init_ev.ts)
        passive_fills = self.engine.process_event(init_ev)
        self.oms.process_fills(passive_fills)
        orders0 = self.strategy.on_market_tick(self.engine.local_book, self.oms)
        if orders0:
            for o in orders0:
                submit_ts = int(init_ev.ts)
                due_ts = int(init_ev.ts + self.order_latency)
                o.create_time = submit_ts
                try:
                    self.metrics.on_order_submitted(o, submit_ts, due_ts)
                except Exception:
                    pass
                pending_orders.append((due_ts, o))
        
        interval_done = 0

        while True:
            curr = self.feed.next()
            if not curr: break

            t0 = int(prev.ts_exch)
            t1 = int(curr.ts_exch)

            # 区间上下文：用于让仿真模型在不侵入 runner 的情况下，按需决定是否生成 micro events、
            # 以及需要模拟的订单簿深度窗口（例如 bridge 模型会用到；简单模型会忽略）。
            market_view = MarketIntervalView(t0=t0, t1=t1, prev=prev, curr=curr)
            orders_view = OrderIntervalView(
                t0=t0,
                t1=t1,
                active_orders=[o for o in self.engine.active_orders if o.is_active],
                pending_orders=list(pending_orders),
            )
            ctx = IntervalContext(orders=orders_view, market=market_view)

            # 生成事件流（模型自行决定是否需要 micro bridge）
            events = self.simulator.generate_events(prev, curr, context=ctx)

            for ev in events:
                self.metrics.update_time(ev.ts)

                # 0) 先把已经到达交易所的订单注册到撮合器（在当前事件前生效）
                if pending_orders:
                    due_now = [it for it in pending_orders if it[0] <= ev.ts]
                    if due_now:
                        pending_orders = [it for it in pending_orders if it[0] > ev.ts]
                        due_now_sorted = sorted(due_now, key=lambda x: x[0])
                        orders_to_send = []
                        for due_ts, o in due_now_sorted:
                            # 记录“到达交易所”时间（语义上是 due_ts，而不是当前事件时间）
                            try:
                                o.arrival_time = int(due_ts)
                            except Exception:
                                pass
                            orders_to_send.append(o)
                        self.engine.register_orders(orders_to_send)
                        self.oms.register_orders(orders_to_send)
                        # 触发一次“同一时刻的盘口检查”，用于处理到达即 taker 的情况
                        if self.engine.local_book.cur:
                            passive = self.engine.process_event(
                                SimulationEvent(ev.ts, EventType.QUOTE_UPDATE, self.engine.local_book.cur)
                            )
                            self.oms.process_fills(passive)
                
                # 1. 引擎消费事件
                passive_fills = self.engine.process_event(ev)
                self.oms.process_fills(passive_fills)
                
                # 2. 策略触发 (仅在快照到达时，也可扩展为定时触发)
                if ev.type == EventType.SNAPSHOT_ARRIVAL:
                    orders = self.strategy.on_market_tick(self.engine.local_book, self.oms)
                    # 订单不会立刻到达交易所：按用户配置的延迟排队
                    if orders:
                        for o in orders:
                            submit_ts = int(ev.ts)
                            due_ts = int(ev.ts + self.order_latency)
                            o.create_time = submit_ts
                            try:
                                self.metrics.on_order_submitted(o, submit_ts, due_ts)
                            except Exception:
                                pass
                            pending_orders.append((due_ts, o))

            prev = curr

            # 完成一个快照区间，汇报进度（每个区间一次，粒度足够直观且开销低）
            interval_done += 1
            self.last_progress = interval_done
            if self.progress_callback is not None:
                try:
                    self.progress_callback(interval_done)
                except Exception:
                    pass
            
        return self._collect_results()

class UnifiedRunner:
    """
    统一回测入口：支持 N=1 (Deterministic) 和 N>1 (Monte Carlo)
    完全解耦：不依赖具体 Model 类，只依赖 Factory
    """
    def __init__(self, 
                 feed_factory: Callable[[], IMarketDataFeed],
                 sim_factory: SimulationFactory,
                 queue_factory: Callable[[], IQueueModel],
                 strategy_factory: Callable[[], IStrategy],
                 num_runs: int = 1,
                 order_latency: int = 0,
                 max_workers: Optional[int] = None,
                 show_progress: bool = True,
                 show_progress_per_run: bool = True,
                 output_dir: str = "outputs",
                 export_fills_csv: bool = True,
                 export_orders_csv: bool = True):
        
        self.feed_factory = feed_factory
        self.sim_factory = sim_factory
        self.queue_factory = queue_factory
        self.strategy_factory = strategy_factory
        self.num_runs = num_runs
        self.order_latency = int(order_latency)
        self.max_workers = max_workers
        self.show_progress = bool(show_progress)
        self.show_progress_per_run = bool(show_progress_per_run)
        self.output_dir = str(output_dir)
        self.export_fills_csv = bool(export_fills_csv)
        self.export_orders_csv = bool(export_orders_csv)

    def run(self) -> pd.DataFrame:
        if self.num_runs == 1:
            print("Mode: Single Deterministic Run")
            # 这里的 seed 可以固定，或者由外部传入
            return self._run_single_process(seed=0, run_id=0)
        else:
            print(f"Mode: Monte Carlo ({self.num_runs} runs)")
            return self._run_parallel()

    def _run_single_process(self, seed: int, run_id: int):
        # 实例化组件
        feed = self.feed_factory()
        sim = self.sim_factory(seed)
        queue = self.queue_factory()
        strat = self.strategy_factory()
        
        # 单进程模式下也给每一轮提供进度条（若 tqdm 可用）
        total_intervals: Optional[int] = None
        try:
            total_intervals = max(0, int(len(feed)) - 1)  # type: ignore[arg-type]
        except Exception:
            total_intervals = None

        pbar = None
        if self.show_progress and self.show_progress_per_run and tqdm is not None:
            pbar = tqdm(total=total_intervals, desc=f"Run {run_id}", unit="interval")

        def _cb(done_intervals: int):
            if pbar is None:
                return
            # tqdm 需要增量更新
            delta = int(done_intervals) - int(pbar.n)
            if delta > 0:
                pbar.update(delta)

        ctx = SingleRunContext(
            feed,
            sim,
            queue,
            strat,
            order_latency=self.order_latency,
            progress_callback=_cb,
            progress_total=total_intervals,
        )
        result = ctx.run()
        # 导出逐笔成交/逐订单明细
        fills_csv = None
        orders_csv = None
        try:
            os.makedirs(str(self.output_dir), exist_ok=True)
            if self.export_fills_csv:
                fills_csv = os.path.join(str(self.output_dir), f"fills_run_{run_id}.csv")
                ctx.metrics.export_fills_csv(fills_csv, run_id=run_id)
            if self.export_orders_csv:
                orders_csv = os.path.join(str(self.output_dir), f"orders_run_{run_id}.csv")
                ctx.metrics.export_orders_csv(orders_csv)
        except Exception:
            pass
        result['fills_csv'] = fills_csv
        result['orders_csv'] = orders_csv
        if pbar is not None:
            pbar.close()
        result['run_id'] = run_id
        result['seed'] = seed
        return pd.DataFrame([result])

    def _run_parallel(self) -> pd.DataFrame:
        """使用多进程并行运行 Monte Carlo，并显示进度条（若安装了 tqdm）。"""
        seeds = np.random.randint(0, 1_000_000, self.num_runs).tolist()
        tasks = [(int(seed), i) for i, seed in enumerate(seeds)]

        results: List[Dict[str, Any]] = []

        # tqdm 可选：
        # - 总体进度条：完成了多少个 run
        # - 每个 run 的区间进度条：run i 完成了多少个 snapshot-interval
        pbar_runs = None
        run_bars: List[Any] = []
        stop_evt = Event()

        # 进度队列（子进程 -> 主进程）
        mgr = Manager()
        progress_q = mgr.Queue()

        # 尝试估计 total_intervals（仅用于进度展示；失败则 total=None）
        total_intervals: Optional[int] = None
        try:
            probe_feed = self.feed_factory()
            total_intervals = max(0, int(len(probe_feed)) - 1)  # type: ignore[arg-type]
        except Exception:
            total_intervals = None

        if self.show_progress and tqdm is not None:
            pbar_runs = tqdm(total=len(tasks), desc="Monte Carlo", unit="run", position=0)
            if self.show_progress_per_run:
                # 为每个 run 建一个进度条。若 run 数很大，终端会很乱；这里仍按你的需求全部展示。
                leave_per_run = (len(tasks) <= 10)
                for i in range(len(tasks)):
                    run_bars.append(
                        tqdm(total=total_intervals, desc=f"Run {i}", unit="interval", position=i + 1, leave=leave_per_run)
                    )

        def _progress_listener():
            """主进程线程：消费进度队列并更新 tqdm。"""
            while not stop_evt.is_set():
                try:
                    msg: ProgressMsg = progress_q.get(timeout=0.2)
                except Exception:
                    continue
                try:
                    typ, rid, done_i, tot_i = msg
                    if 0 <= rid < len(run_bars) and run_bars:
                        bar = run_bars[rid]
                        # 若 total 在运行时第一次拿到，可设置（tqdm 支持修改 total）
                        if getattr(bar, "total", None) is None and tot_i is not None:
                            bar.total = int(tot_i)
                        delta = int(done_i) - int(bar.n)
                        if delta > 0:
                            bar.update(delta)
                        if typ == "done":
                            bar.close()
                except Exception:
                    # 任何进度更新异常都不应影响回测
                    continue

        listener = Thread(target=_progress_listener, daemon=True)
        listener.start()

        try:
            with ProcessPoolExecutor(max_workers=self.max_workers) as ex:
                futures = []
                for seed, run_id in tasks:
                    fut = ex.submit(
                        _mc_worker_run,
                        seed,
                        run_id,
                        self.order_latency,
                        self.feed_factory,
                        self.sim_factory,
                        self.queue_factory,
                        self.strategy_factory,
                        self.output_dir,
                        self.export_fills_csv,
                        self.export_orders_csv,
                        progress_q,
                    )
                    futures.append(fut)

                completed = 0
                for fut in as_completed(futures):
                    res = fut.result()  # 若子进程异常，这里会抛出
                    results.append(res)
                    completed += 1
                    if pbar_runs is not None:
                        pbar_runs.update(1)
                    else:
                        print(f"Completed {completed}/{len(tasks)} runs")
        finally:
            stop_evt.set()
            try:
                listener.join(timeout=1.0)
            except Exception:
                pass
            if pbar_runs is not None:
                pbar_runs.close()
            # 若 run_bars 未关闭（例如异常提前退出），此处兜底关闭
            for bar in run_bars:
                try:
                    bar.close()
                except Exception:
                    pass

        df = pd.DataFrame(results)
        # 保证输出顺序稳定（按 run_id 排序）
        if "run_id" in df.columns:
            df = df.sort_values("run_id").reset_index(drop=True)
        return df
