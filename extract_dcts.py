from jpeg2dct.numpy import load, loads
from typing import Tuple, List


def process(patches: List[Tuple[str, bytes]]):
    for p in patches:
        dct = loads(p)
        
