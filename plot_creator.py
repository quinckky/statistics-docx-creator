from io import BytesIO

import matplotlib.pyplot as plt
from numpy.typing import ArrayLike

from stats import *


def distribution_plot(x: ArrayLike) -> BytesIO:
    buffer = BytesIO()
    x = sorted(x)
    plt.xlabel("x")
    plt.ylabel("F*(x)")
    plt.ecdf(x)
    plt.savefig(buffer, format="png")
    plt.close()
    return buffer


def interval_hist_plot(x: ArrayLike, n: int) -> BytesIO:
    buffer = BytesIO()
    x = sorted(x)
    plt.xlabel("x")
    plt.ylabel("F*(x)")
    plt.hist(x, density=True, bins=n)
    plt.savefig(buffer, format="png")
    plt.close()
    return buffer


def probability_hist_plot(x: ArrayLike, n: int) -> BytesIO:
    buffer = BytesIO()
    x = sorted(x)
    h, *intervals = possibility_intervals(x, n)
    f = intervals_relative_freq_density(x, intervals, h)
    left, _ = intervals
    plt.xlabel("x")
    plt.ylabel("F*(x)")
    plt.bar(left, height=f, width=h, align="edge")
    plt.savefig(buffer, format="png")
    plt.close()
    return buffer


def norm_empiric_distrib(x: ArrayLike) -> BytesIO:
    buffer = BytesIO()
    x = sorted(x)
    xp = np.linspace(min(x), max(x), 100)
    norm_x = norm_distribution(xp, fixed=True)
    plt.xlabel("x")
    plt.ylabel("F*(x)")
    plt.ecdf(x)
    plt.plot(xp, norm_x, c="orange")
    plt.savefig(buffer, format="png")
    plt.close()
    return buffer


def lin_reg_plot(xy: ArrayLike, r) -> BytesIO:
    buffer = BytesIO()
    var = sp.Symbol("x")
    x, y = zip(*xy)
    linear_regression = lin_reg(xy, r)
    ox = [min(x), max(x)]
    oy = [linear_regression.subs(var, point) for point in ox]
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(x, y)
    plt.plot(ox, oy, color="orange")
    plt.savefig(buffer, format="png")
    plt.close()
    return buffer