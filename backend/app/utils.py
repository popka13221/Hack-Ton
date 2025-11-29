from typing import List


def chunked(iterable: List[str], size: int):
    """Yield chunks of a list with given size."""
    for i in range(0, len(iterable), size):
        yield iterable[i : i + size]
