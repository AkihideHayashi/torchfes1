from typing import Set

saves: Set[str] = set()
metad: Set[str] = set()
batch: Set[str] = set()
atoms: Set[str] = set()


def add(to, keys):
    for to_key in to:
        to_key.update(keys)
