"""
Module for calculating the value of pi using various methods.
"""

import math


def calculate_pi_leibniz(iterations=1000000):
    """
    Calculate pi using the Leibniz formula.
    
    The Leibniz formula: π/4 = 1 - 1/3 + 1/5 - 1/7 + 1/9 - ...
    
    Args:
        iterations (int): Number of iterations to perform (default: 1000000)
    
    Returns:
        float: Approximation of pi
    """
    pi = 0
    for i in range(iterations):
        pi += ((-1) ** i) / (2 * i + 1)
    return pi * 4


def calculate_pi_nilakantha(iterations=100000):
    """
    Calculate pi using the Nilakantha series.
    
    The Nilakantha series: π = 3 + 4/(2*3*4) - 4/(4*5*6) + 4/(6*7*8) - ...
    This converges faster than the Leibniz formula.
    
    Args:
        iterations (int): Number of iterations to perform (default: 100000)
    
    Returns:
        float: Approximation of pi
    """
    pi = 3
    for i in range(1, iterations):
        sign = (-1) ** (i + 1)
        denominator = 2 * i * (2 * i + 1) * (2 * i + 2)
        pi += sign * 4 / denominator
    return pi


def calculate_pi_chudnovsky(precision=2):
    """
    Calculate pi using the Chudnovsky algorithm.
    
    This is one of the fastest known algorithms for calculating pi.
    Each term adds approximately 14 digits of precision.
    
    Args:
        precision (int): Number of terms to calculate (default: 2)
    
    Returns:
        float: Approximation of pi
    """
    C = 426880 * math.sqrt(10005)
    K = 6.0
    M = 1.0
    X = 1
    L = 13591409
    S = L
    
    for i in range(1, precision):
        M = M * (K ** 3 - 16 * K) / ((i) ** 3)
        K += 12
        L += 545140134
        X *= -262537412640768000
        S += (M * L) / X
    
    return C / S


def calculate_pi_monte_carlo(samples=1000000):
    """
    Calculate pi using the Monte Carlo method.
    
    This method randomly generates points in a square and counts how many
    fall within a quarter circle inscribed in the square.
    
    Args:
        samples (int): Number of random samples to generate (default: 1000000)
    
    Returns:
        float: Approximation of pi
    """
    import random
    inside_circle = 0
    
    for _ in range(samples):
        x = random.random()
        y = random.random()
        if x * x + y * y <= 1:
            inside_circle += 1
    
    return 4 * inside_circle / samples


def calculate_pi(method='nilakantha', **kwargs):
    """
    Calculate pi using the specified method.
    
    Args:
        method (str): The method to use ('leibniz', 'nilakantha', 'chudnovsky', 'monte_carlo')
        **kwargs: Additional arguments to pass to the specific method
    
    Returns:
        float: Approximation of pi accurate to at least 5 decimal places
    """
    methods = {
        'leibniz': calculate_pi_leibniz,
        'nilakantha': calculate_pi_nilakantha,
        'chudnovsky': calculate_pi_chudnovsky,
        'monte_carlo': calculate_pi_monte_carlo
    }
    
    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")
    
    return methods[method](**kwargs)


if __name__ == "__main__":
    # Demonstrate different methods
    print("Calculating Pi using different methods:")
    print(f"Actual Pi: {math.pi}")
    print(f"Leibniz method: {calculate_pi_leibniz():.10f}")
    print(f"Nilakantha method: {calculate_pi_nilakantha():.10f}")
    print(f"Chudnovsky method: {calculate_pi_chudnovsky():.10f}")
    print(f"Monte Carlo method: {calculate_pi_monte_carlo():.10f}")
    print(f"\nDefault method (Nilakantha): {calculate_pi():.5f}")
