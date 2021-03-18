from typing import Set, List

saves: Set[str] = set()
metad: Set[str] = set()
batch: Set[str] = set()
atoms: Set[str] = set()
saves_fine: Set[str] = set()


def add(to: List[Set[str]], keys: Set[str]):
    for to_key in to:
        to_key.update(keys)
