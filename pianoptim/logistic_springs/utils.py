from casadi import exp


# --- Model Definitions ---
def model_cubic(x, params):
    """
    Standard cubic polynomial.
    params: [a, b, c, d]
    """
    a, b, c, d = params
    return a * x**3 + b * x**2 + c * x + d


def model_exponential_decay(x, params):
    """
    Exponential decay function: C + A * exp(-k * x)
    params: [A, k, C] (Amplitude, Rate, Offset)
    """
    A, k, C = params
    return C + A * exp(-k * x)
