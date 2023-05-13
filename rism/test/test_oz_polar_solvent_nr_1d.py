#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_oz_polar_solvent_nr_1d.py
created time : 2023/04/25
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import cupy as cp
import matplotlib.pyplot as plt
import rism
from rism.core import FFTGrid
from rism.solver.oz_polar_solvent_nr_1d import OZPolarSolventNR1DSolver
from rism.element import *
from rism.environment import CUPY_FLOAT
from rism.unit import *


def get_basis(r, center, width):
    res = cp.exp(-((r - center) ** 2) / (2 * width**2))
    return res.astype(CUPY_FLOAT)


def visualize(grid, h, c, plt_arg=None):
    if plt_arg is None:
        fig, ax = plt.subplots(1, 3, figsize=[25, 9])
    else:
        fig, ax = plt_arg
    g = h + 1
    gamma = h - c

    r = grid.r.get()
    ax[0].plot(r, g.get(), ".-")
    ax[1].plot(r, c.get(), ".-")
    ax[2].plot(r, gamma.get(), ".-")
    ax[0].set_title("g")
    ax[1].set_title("c")
    ax[2].set_title(r"$\gamma$")
    return fig, ax


if __name__ == "__main__":
    temperature = 300
    grid = FFTGrid(r=[0, 40, 2048])
    rho_b = Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton) / NA
    closure = rism.closure.kovalenko_hirata
    basis_set = [get_basis(grid.r, i, 1.0) for i in [0, 0.75, 1.5, 2.25]]

    solver = OZPolarSolventNR1DSolver(
        grid=grid,
        closure=closure,
        basis_set=basis_set,
        temperature=temperature,
        solvent=oxygen(),
        rho_b=rho_b,
    )
    h, c = solver.solve(
        max_iterations=1000,
        error_tolerance=1e-5,
        log_freq=10,
        alpha=0.7,
        nr_max_iterations=100,
        nr_step_size=0.1,
        nr_tolerance=5e-5,
    )
    fig, ax = visualize(grid, h, c)

    h, c = solver.solve(
        max_iterations=1000,
        error_tolerance=1e-5,
        log_freq=10,
        alpha=0.8,
        nr_max_iterations=100,
        nr_step_size=0.1,
        nr_tolerance=5e-5,
    )
    visualize(grid, h, c, (fig, ax))

    h, c = solver.solve(
        max_iterations=1000,
        error_tolerance=1e-5,
        log_freq=10,
        alpha=0.9,
        nr_max_iterations=100,
        nr_step_size=0.05,
        nr_tolerance=5e-5,
    )
    visualize(grid, h, c, (fig, ax))
    plt.show()
