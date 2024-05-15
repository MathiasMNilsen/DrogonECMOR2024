import numpy as np

def log_poly(a, b, c, min, max, steps):
    t = np.arange(0, steps)/steps
    exp = np.exp(a + b*t + c*t**2)
    val = min + (max - min)/(1+exp)
    return val