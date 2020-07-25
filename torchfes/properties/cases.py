from typing import Set

save_trj: Set[str] = set()
metadynamics: Set[str] = set()
batch: Set[str] = set()
atoms: Set[str] = set()


def add(to, keys):
    for to_key in to:
        to_key.update(keys)
