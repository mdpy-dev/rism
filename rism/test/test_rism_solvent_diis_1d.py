#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_rism_solvent_picard_1d.py
created time : 2023/04/20
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import matplotlib.pyplot as plt
import rism
from rism.core import FFTGrid
from rism.solver.rism_solvent_diss_1d import RISMSolventDIIS1DSolver
from rism.solvent import *
from rism.unit import *


def visualize(grid, matrix, site_list):
    num_sites = len(site_list)
    r = grid.r.get()

    if True:
        fig, ax = plt.subplots(1, 1, figsize=[16, 9])
        for i in range(num_sites - 1):
            for j in range(i, num_sites - 1):
                ax.plot(
                    r,
                    matrix[i, j].get(),
                    ".-",
                    label="%s-%s" % (site_list[i], site_list[j]),
                )
                ax.legend()
    else:
        fig, ax = plt.subplots(num_sites, num_sites, figsize=[16, 16])
        y_max = matrix.max().get() * 1.1
        for i in range(num_sites):
            for j in range(num_sites):
                ax[i, j].plot(r, matrix[i, j].get(), ".-", label="g")
                ax[i, j].set_title("%s-%s" % (site_list[i], site_list[j]))
                ax[i, j].legend()
                ax[i, j].set_ylim(0, y_max)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    temperature = 300
    grid = FFTGrid(r=[0, 20, 2048])
    rho_b = Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton) / NA
    closure = rism.closure.hnc
    solvent = tip3p()

    solver = RISMSolventDIIS1DSolver(
        grid=grid,
        closure=closure,
        temperature=temperature,
        solvent=solvent,
        rho_b=rho_b,
    )
    print(solver._get_bond_length())
    h_matrix, c_matrix = solver.solve(max_iterations=1000, error_tolerance=1e-5)
    visualize(grid, h_matrix + 1, solver.site_list)
    # h, c = solver.solve(np.array([0, 0, 0]), iterations=500, restart_value=(h, c))
    # visualize(grid, h, c, False)
