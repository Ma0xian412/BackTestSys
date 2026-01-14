from typing import List, Tuple, Optional
from ..core.interfaces import ITradeTapeReconstructor
from ..core.types import NormalizedSnapshot, Price, Qty

class PreCalculatedTapeReconstructor(ITradeTapeReconstructor):
    def reconstruct(self, prev: Optional[NormalizedSnapshot], curr: NormalizedSnapshot) -> List[Tuple[Price, Qty]]:
        return curr.last_vol_split