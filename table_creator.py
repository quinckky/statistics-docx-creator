import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from pandas import DataFrame

from stats import *


def variation_table(x: ArrayLike) -> DataFrame:
    n = len(x)
    x = sorted(x)
    indexes = list(range(1, n+1))
    names = ("№", "x")
    columns, names = _split_cols([indexes, x], names, 25)
    table = DataFrame(data=columns, columns=names)
    return table


def distribution_table(x: ArrayLike) -> DataFrame:
    x_unique = sorted(set(x))
    x_freq = cumfreq(x)
    x_distribution = distribution(x)
    names = ("x", "nₓ", "F*(x)")
    columns, names = _split_cols([x_unique, x_freq, x_distribution], names, 25)
    table = DataFrame(data=columns, columns=names)
    return table


def interval_hist_table(x: ArrayLike, n: int) -> DataFrame:
    h, *intervals = equal_intervals(x, n)
    freq = intervals_freq(x, intervals)
    freq_density = intervals_freq_density(x, intervals, h)
    relative_freq = intervals_relative_freq(x, intervals)
    relative_freq_density = intervals_relative_freq_density(x, intervals, h)
    names = ("a", "b", "h", "n", "n/h", "w", "w/h")
    columns, names = _split_cols(
        [*intervals, h, freq, freq_density, relative_freq, relative_freq_density], names, n)
    table = DataFrame(data=columns, columns=names)
    return table


def probability_hist_table(x: ArrayLike, n: int) -> DataFrame:
    h, *intervals = possibility_intervals(x, n)
    freq = intervals_freq(x, intervals)
    freq_density = intervals_freq_density(x, intervals, h)
    relative_freq = intervals_relative_freq(x, intervals)
    relative_freq_density = intervals_relative_freq_density(x, intervals, h)
    names = ("a", "b", "h", "n", "n/h", "w", "w/h")
    columns, names = _split_cols(
        [*intervals, h, freq, freq_density, relative_freq, relative_freq_density], names, n)
    table = DataFrame(data=columns, columns=names)
    return table


def hypothesis_table(x: ArrayLike, n: int) -> DataFrame:
    _, *intervals = equal_intervals(x, n)
    mid = intervals_mid(intervals)
    freq = intervals_freq(x, intervals)
    x_z = z(x, n)
    f_z = gauss(x_z)
    x_obs_freq = obs_freq(x, n)
    x_diff_freq = diff_freq(x, n)
    names = ("a", "b", "x", "n", "z", "f(z)", "n'", "(n-n')²/n'")
    columns, names = _split_cols(
        [*intervals, mid, freq, x_z, f_z, x_obs_freq, x_diff_freq], names, n)
    table = DataFrame(data=columns, columns=names)
    return table


def xy_table(xy: ArrayLike) -> DataFrame:
    x, y = zip(*xy)
    x = np.array(x)
    y = np.array(y)
    x_mul_y = x*y
    names = ("x", "y", "xy")
    columns, names = _split_cols(
        [x, y, x_mul_y], names, 25)
    table = DataFrame(data=columns, columns=names)
    return table


def _split_cols(raw_cols: list[list], names: tuple[str], rows_n: int) -> list[list]:
    n = len(raw_cols[0])
    cols = [[] for _ in range(n)]
    columns = []
    for i in range(1, n+1):
        for col, arr in zip(cols, raw_cols):
            col.append(arr[i-1])
            if i % rows_n == 0:
                columns.append(col[:])
                col.clear()
            elif i == n:
                col = pd.Series(col[:]).reindex(range(rows_n))
                columns.append(col)
    columns = np.transpose(columns)
    names = names*(n//rows_n + (1 if n % rows_n else 0))
    return columns, names
