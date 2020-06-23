from typing import List
import torch
from torchfes.fes import MTDHillsData

datas: List[MTDHillsData] = []
with open('hills.pt', 'rb') as f:
    while True:
        try:
            datas.append(torch.load(f))
        except (EOFError, RuntimeError):
            break


print(datas)