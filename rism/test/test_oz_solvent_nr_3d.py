#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : oz_solvent_nr_3d.py
created time : 2023/04/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import cupy as cp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import rism
from rism.core import FFTGrid
from rism.solver import OZSolventNR3DSSolver
from rism.environment import CUPY_FLOAT
from rism.unit import *


def get_basis(r, center, width):
    # res = cp.abs(r - center)
    # area = res < width
    # res[area] = -1 / width * res[area] + 1
    # res[~area] = 0
    res = cp.exp(-((r - center) ** 2) / (2 * width**2))
    return res.astype(CUPY_FLOAT)


def visualize(grid, h, c, is_2d=True):
    half = grid.shape[2] // 2
    fig, ax = plt.subplots(1, 3, figsize=[25, 9])
    g = h + 1
    gamma = h - c
    if is_2d:
        x = grid.x[:, :, half].get()
        y = grid.y[:, :, half].get()
        all_res = cp.stack([c, gamma])
        norm = matplotlib.colors.Normalize(
            vmin=all_res.min().get(), vmax=all_res.max().get()
        )
        cb1 = ax[0].contour(x, y, g[:, :, half].get(), 50)
        cb2 = ax[1].contour(x, y, c[:, :, half].get(), 50, norm=norm)
        ax[2].contour(x, y, gamma[:, :, half].get(), 50, norm=norm)
        fig.subplots_adjust(left=0.12, right=0.9)
        position = fig.add_axes([0.02, 0.10, 0.015, 0.80])
        cb1 = fig.colorbar(cb1, cax=position)
        position = fig.add_axes([0.92, 0.10, 0.015, 0.80])
        cb1 = fig.colorbar(cb2, cax=position)
    else:
        x = grid.x[half:, half, half].get()
        ax[0].plot(x, g[half:, half, half].get(), ".-")
        ax[1].plot(x, c[half:, half, half].get(), ".-")
        ax[2].plot(x, gamma[half:, half, half].get(), ".-")
    ax[0].set_title("g")
    ax[1].set_title("c")
    ax[2].set_title(r"$\gamma$")
    plt.show()


if __name__ == "__main__":
    temperature = 300
    center = np.array([0, 0, 0])
    grid = FFTGrid(x=[-20, 20, 256], y=[-20, 20, 256], z=[-20, 20, 256])
    rho_b = Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton) / NA
    closure = rism.closure.kovalenko_hirata

    r = cp.sqrt(
        (grid.x - center[0]) ** 2
        + (grid.y - center[1]) ** 2
        + (grid.z - center[2]) ** 2
    )
    basis_set = [get_basis(r, i, 1.0) for i in [0, 1.0, 1.5, 2.5]]
    # basis_set += [get_basis(r, i, 0.5) for i in np.arange(2.5, 6, 0.5)]

    solver = OZSolventNR3DSSolver(
        grid=grid,
        closure=closure,
        basis_set=basis_set,
        temperature=temperature,
        solvent_type="o",
        rho_b=rho_b,
    )
    h, c = solver.solve(
        center,
        max_iterations=500,
        log_freq=10,
        error_tolerance=5e-5,
        nr_max_iterations=50,
        nr_step_size=0.5,
        nr_tolerance=0.0005,
    )

    visualize(grid, h, c, False)
