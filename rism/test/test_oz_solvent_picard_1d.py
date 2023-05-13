#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_oz_solvent_picard_1d.py
created time : 2023/04/21
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import matplotlib.pyplot as plt
import rism
from rism.core import FFTGrid
from rism.solver.oz_solvent_picard_1d import OZSolventPicard1DSolver
from rism.element import *
from rism.unit import *


def visualize(grid, h, c):
    fig, ax = plt.subplots(1, 3, figsize=[25, 9])
    g = h + 1
    gamma = h - c

    r = grid.r.get()
    ax[0].plot(r, g.get(), ".-")
    ax[1].plot(r, c.get(), ".-")
    ax[2].plot(r, gamma.get(), ".-")
    ax[0].set_title("g")
    ax[1].set_title("c")
    ax[2].set_title(r"$\gamma$")
    plt.show()


if __name__ == "__main__":
    temperature = 300
    grid = FFTGrid(r=[0, 20, 512])
    rho_b = Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton) / NA
    closure = rism.closure.hnc

    solver = OZSolventPicard1DSolver(
        grid=grid,
        closure=closure,
        temperature=temperature,
        solvent=oxygen(),
        rho_b=rho_b,
    )
    h, c = solver.solve(max_iterations=500, error_tolerance=1e-5)
    visualize(grid, h, c)
    # h, c = solver.solve(np.array([0, 0, 0]), iterations=500, restart_value=(h, c))
    # visualize(grid, h, c, False)
