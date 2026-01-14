from typing import Iterator
from ...core.interfaces import ISimulationModel, ITradeTapeReconstructor
from ...core.types import NormalizedSnapshot
from ...core.events import SimulationEvent, EventType

class SimpleSnapshotModel(ISimulationModel):
    def __init__(self, tape_recon: ITradeTapeReconstructor, seed: int = 0):
        self.tape = tape_recon
        # 简单模型不使用 seed，因为它是确定性的

    def generate_events(self, prev: NormalizedSnapshot, curr: NormalizedSnapshot, context=None) -> Iterator[SimulationEvent]:
        trades = self.tape.reconstruct(prev, curr)
        count = len(trades)
        dt = max(1, curr.ts_exch - prev.ts_exch) // (count + 1)
        ts = prev.ts_exch

        for px, qty in trades:
            ts += dt
            yield SimulationEvent(ts, EventType.TRADE_TICK, (px, qty))
        
        yield SimulationEvent(curr.ts_exch, EventType.SNAPSHOT_ARRIVAL, curr)