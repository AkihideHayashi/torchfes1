from typing import Dict, Tuple


def register(keys: Dict[str, Tuple[Dict[str, bool], str]]):
    for key, (prp, exp) in keys.keys():
        print(key)
