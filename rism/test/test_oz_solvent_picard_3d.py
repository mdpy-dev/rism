#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_oz_solvent_picard_3d.py
created time : 2023/04/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import cupy as cp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from rism.core import FFTGrid
from rism.solver import OZSolventPicard3DSolver


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
    grid = FFTGrid(x=[-20, 20, 256], y=[-20, 20, 256], z=[-20, 20, 256])

    solver = OZSolventPicard3DSolver(
        grid=grid, temperature=temperature, solvent_type="o"
    )
    h, c = solver.solve(np.array([4, 4, 0]), iterations=10)
    visualize(grid, h, c)
    h, c = solver.solve(np.array([4, 4, 0]), iterations=100, restart_value=(h, c))
    visualize(grid, h, c)
