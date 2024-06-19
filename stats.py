from math import sqrt, pi, exp, log

import numpy as np
import sympy as sp
from numpy.typing import ArrayLike


def distribution(x: ArrayLike) -> float:
    x = sorted(x)
    n = len(x)
    x_cumfreq = cumfreq(x)
    x_distribution = [nx/n for nx in x_cumfreq]
    return x_distribution


def cumfreq(x: ArrayLike) -> list[int]:
    x = sorted(x)
    x_freq = freq(x)
    return np.cumsum(x_freq)


def freq(x: ArrayLike) -> list[int]:
    x = sorted(x)
    x_unique = sorted(set(x))
    x_frequences = [x.count(xi) for xi in x_unique]
    return x_frequences


def equal_intervals(x: ArrayLike, n: int) -> tuple[float, list[float], list[float]]:
    x = sorted(x)
    h = [(x[-1] - x[0]) / n for _ in range(n)]
    left = []
    right = []
    for i in range(1, n+1):
        left.append(x[0] + (i-1)*h[0])
        right.append(x[0] + i*h[0])
    return h, left, right


def possibility_intervals(x: ArrayLike, n: int) -> tuple[float, list[float], list[float]]:
    x = sorted(x)
    left = []
    right = []
    h = []
    for i in range(n):
        if i == 0:
            left.append(x[0])
        else:
            left.append((x[i*n] + x[i*n-1])/2)
        if i == n - 1:
            right.append(x[-1])
        else:
            right.append((x[(i+1)*n] + x[(i+1)*n-1])/2)
        h.append(right[i] - left[i])
    return h, left, right


def intervals_mid(intervals: list[list[float], list[float]]):
    mid = [(right + left)/2 for right, left in zip(*intervals)]
    return mid


def intervals_relative_freq_density(x: ArrayLike, intervals: list[list[float], list[float]], h: list[float]) -> list[float]:
    x = sorted(x)
    relative_freq = intervals_relative_freq(x, intervals)
    relative_freq_density = [wi/hi for wi, hi in zip(relative_freq, h)]
    return relative_freq_density


def intervals_relative_freq(x: ArrayLike, intervals: list[list[float], list[float]]) -> list[float]:
    x = sorted(x)
    n = len(x)
    freq = intervals_freq(x, intervals)
    relative_freq = [ni/n for ni in freq]
    return relative_freq


def intervals_freq_density(x: ArrayLike, intervals: list[list[float], list[float]], h: list[float]) -> list[float]:
    x = sorted(x)
    freq = intervals_freq(x, intervals)
    freq_density = [ni/hi for ni, hi in zip(freq, h)]
    return freq_density


def intervals_freq(x: ArrayLike, intervals: list[list[float], list[float]]) -> list[int]:
    x = sorted(x)
    freq = []
    for left, right in zip(*intervals):
        counter = 0
        for xi in x:
            if left <= xi <= right:
                counter += 1
        freq.append(counter)
    return freq


def avg_interval(x: ArrayLike, t: float) -> tuple[float, float]:
    x_avg = avg(x)
    x_delta = delta(x, t)
    left = x_avg - x_delta
    right = x_avg + x_delta
    return left, right


def delta(x: ArrayLike, t: float) -> float:
    x = sorted(x)
    n = len(x)
    x_std = std(x)
    delta = (x_std*t) / (n**0.5)
    return delta


def varience_interval(x: ArrayLike, xi1: float, xi2: float) -> tuple[float, float]:
    x = sorted(x)
    n = len(x)
    x_variance = variance(x)
    left = (n-1)*x_variance/xi1
    right = (n-1)*x_variance/xi2
    return left, right


def variance(x: ArrayLike, fixed=True) -> float:
    x = sorted(x)
    n = len(x)
    x_average = avg(x)
    x_variance = sum((xi - x_average)**2 for xi in x)/n
    if fixed:
        x_variance *= n/(n-1)
    return x_variance


def avg(x: ArrayLike) -> float:
    return np.average(x)


def xi_quad(x: ArrayLike, n: int) -> list[float]:
    return sum(diff_freq(x, n))


def diff_freq(x: ArrayLike, n: int) -> list[float]:
    x = sorted(x)
    _, *intervals = equal_intervals(x, n)
    x_freq = intervals_freq(x, intervals)
    x_obs_freq = obs_freq(x, n)
    diff_freq = [((ni-obs)**2)/obs for ni, obs in zip(x_freq, x_obs_freq)]
    return diff_freq


def obs_freq(x: ArrayLike, n: int) -> list[float]:
    x = sorted(x)
    h, *_ = equal_intervals(x, n)
    x_std = std(x)
    x_z = z(x, n)
    fz = gauss(x_z)
    x_obs_freq = [((hi*len(x))/x_std)*fzi for hi, fzi in zip(h, fz)]
    return x_obs_freq


def z(x: ArrayLike, n: int) -> float:
    x = sorted(x)
    _, *intervals = equal_intervals(x, n)
    mid = intervals_mid(intervals)
    x_freq = intervals_freq(x, intervals)
    mid_freq = []
    for ni, xi in zip(x_freq, mid):
        mid_freq.extend([xi for i in range(ni)])
    x_avg = avg(mid_freq)
    x_std = std(mid_freq)
    x_z = [(xi-x_avg)/x_std for xi in mid]
    return x_z


def norm_z(x: ArrayLike) -> float:
    x = sorted(x)
    x_avg = avg(x)
    x_std = std(x)
    x_z = [(xi-x_avg)/x_std for xi in x]
    return x_z


def std(x: ArrayLike, fixed=True) -> float:
    x = sorted(x)
    x_variance = variance(x, fixed=fixed)
    x_standart_deviation = sqrt(x_variance)
    return x_standart_deviation


def gauss(z: list[float]) -> list[float]:
    z_gauss = [(1/sqrt(2*pi))*exp(-(zi*zi)/2) for zi in z]
    return z_gauss


def colmogor(x: ArrayLike) -> float:
    x = sorted(x)
    n = len(x)
    x_max_diverse = max_diverse(x)
    return sqrt(n)*x_max_diverse


def norm_distribution(x: ArrayLike, fixed=False) -> list[float]:
    x = sorted(x)
    x_obs_freq = obs_freq(x, 100)
    if fixed: 
        n = sum(x_obs_freq)
    else: 
        n = len(x_obs_freq)
    x_norm = [nx/n for nx in np.cumsum(x_obs_freq)]
    return x_norm


def max_diverse(x: ArrayLike) -> float:
    x = sorted(x)
    x_distribution = distribution(x)
    x_norm = norm_distribution(x, fixed=True)
    x_max_diverse = max(abs(empir - hypo)
                        for empir, hypo in zip(x_distribution, x_norm))
    return x_max_diverse


def a_b(x, r, z):
    n = len(x)
    a = 0.5*log((1+r) / (1-r)) - z/sqrt(n-3)
    b = 0.5*log((1+r) / (1-r)) + z/sqrt(n-3)
    return a, b

def correlation_interval(x: ArrayLike, r: float, z: float) -> tuple[float, float]:
    a, b = a_b(x, r, z)
    left = (exp(2*a) - 1) / (exp(2*a) + 1)
    right = (exp(2*b) - 1) / (exp(2*b) + 1)
    return left, right


def static_crit(n, r):
    return r*sqrt(n-2)/sqrt(1-r**2)


def lin_reg(xy: ArrayLike, r) -> sp.Add:
    var = sp.Symbol("x")
    x, y = zip(*xy)
    x_std = std(x)
    y_std = std(y)
    x_avg = avg(x)
    y_avg = avg(y)
    linear_regression = r * y_std/x_std * (var-x_avg) + y_avg
    return linear_regression