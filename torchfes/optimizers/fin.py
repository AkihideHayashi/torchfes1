from .. import properties as p


def default_printer(i, mol):
    print(i, mol[p.frc.norm(p=2, dim=-1).max().item()])


class MaxForce:
    def __init__(self, max_frc: float, max_stp: int, printer=default_printer):
        self.max_frc = max_frc
        self.max_stp = max_stp
        self.printer = printer

    def __call__(self, i, mol):
        self.printer(i, mol)
        if i > self.max_stp:
            return True
        if mol[p.frc].norm(p=2, dim=-1).max() < self.max_frc:
            return True
        return False
