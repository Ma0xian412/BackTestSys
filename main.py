from quant_framework.runner.system import UnifiedRunner
from quant_framework.core.data_loader import PickleMarketDataFeed
from quant_framework.market.tape import PreCalculatedTapeReconstructor
from quant_framework.simulation.models.simple import SimpleSnapshotModel
from quant_framework.simulation.models.unified_bridge import UnifiedBridgeModel
from quant_framework.execution.queue_models import ProbabilisticQueueModel
from quant_framework.trading.strategy import LadderTestStrategy

# --- 1. 定义工厂函数 (配置层) ---

DATA_PATH = "data/sample.pkl"

def create_feed():
    return PickleMarketDataFeed(DATA_PATH)

def create_simple_sim(seed: int):
    # 简单模型，seed 其实没用，但接口统一
    tape = PreCalculatedTapeReconstructor()
    return SimpleSnapshotModel(tape, seed)

def create_unified_sim(seed: int):
    # 复杂模型，seed 影响随机路径
    # tick_size: 可选的最小变动单位（建议在单合约/单tick配置的回测里显式传入，避免模型用端点推断误判）
    # 例如：return UnifiedBridgeModel(seed=seed, tick_size=0.01)
    return UnifiedBridgeModel(seed=seed)

def create_queue():
    return ProbabilisticQueueModel(k=-0.5)

def create_strategy():
    return LadderTestStrategy()

# --- 2. 运行 ---

if __name__ == "__main__":
    
    # 场景 A: 传统回测 (N=1, Simple Model)
    print("\n--- Scenario A: Classic Backtest ---")
    runner_a = UnifiedRunner(
        feed_factory=create_feed,
        sim_factory=create_simple_sim, # 注入简单模型
        queue_factory=create_queue,
        strategy_factory=create_strategy,
        num_runs=1
    )
    res_a = runner_a.run()
    print(res_a)

    # 场景 B: 蒙特卡洛回测 (N=5, Unified Bridge Model)
    print("\n--- Scenario B: Monte Carlo Backtest ---")
    runner_b = UnifiedRunner(
        feed_factory=create_feed,
        sim_factory=create_unified_sim, # 注入复杂模型
        queue_factory=create_queue,
        strategy_factory=create_strategy,
        num_runs=5 # 自动切换到多路运行
    )
    res_b = runner_b.run()
    
    # 输出每一轮结果（不做 describe/std 汇总）
    print(res_b)