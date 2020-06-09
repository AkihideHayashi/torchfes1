

def coordinate_number(sod, sod0, n, d):
    num = 1 - (sod / sod0) ** (n * 0.5)
    den = 1 - (sod / sod0) ** (d * 0.5)
    return num / den
