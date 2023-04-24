#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : rism_solvent_diss_1d.py
created time : 2023/04/19
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-present, Zhenyu Wei and Southeast University
"""


import time
import cupy as cp
import cupyx.scipy.fft as fft
from rism.environment import CUPY_FLOAT
from rism.core import FFTGrid, Solvent
from rism.potential import VDWPotential
from rism.unit import *


class RISMSolventDIIS1DSolver:
    def __init__(
        self,
        grid: FFTGrid,
        closure,
        solvent: Solvent,
        rho_b: Quantity,
        temperature=Quantity(300, kelvin),
    ) -> None:
        """Create solver for a RISM equation in 3D cartesian coordinate system using Picard iteration

        Args:
            grid (FFTGrid): The grid defining the coordinate system
            closure (Any): The closure for OZ equation from rism.closure
            solvent (Solvent): object of rism.core.Solvent class to describe the topology of solvent sites
            rho_b (`rism.unit.Quantity` or `float`): density of solvent in bulk, Unit: mol_dimension/length_dimension**3
            temperature (`rism.unit.Quantity` or `float`, optional): _description_. Defaults to Quantity(300, kelvin).
        """
        # Read input
        self._grid = grid
        self._closure = closure
        self._solvent = solvent
        # Attributes
        self._num_sites = self._solvent.num_particles
        self._rho_b = CUPY_FLOAT(
            (check_quantity(rho_b, mol / decimeter**3) * NA)
            .convert_to(1 / default_length_unit**3)
            .value
        )
        self._beta = CUPY_FLOAT(
            1
            / (check_quantity(temperature, kelvin) * KB)
            .convert_to(default_energy_unit)
            .value
        )

        # To avoid the singularity point in zero index, we use 1:N+1 as the index
        # Hence the lr, appearing in calculation of k, is dr * (N+1)
        lr = self._grid.dr * (self._grid.shape[0] + 1)
        i_vec = cp.arange(1, self._grid.shape[0] + 1)
        j_vec = cp.arange(1, self._grid.shape[0] + 1)
        forward_factor = (4 * self._grid.dr**2 * lr) / j_vec
        backward_factor = 1 / (2 * lr**2 * self._grid.dr) / i_vec
        self._transform_coefficient = [i_vec, forward_factor, j_vec, backward_factor]

        self._site_list = self._solvent.particle_list
        self._bond_length = self._get_bond_length()
        self._w_k = self._get_w_k_matrix()
        self._exp_u = self._get_exp_u_matrix()

    def _get_empty_matrix(self, shape, dtype=CUPY_FLOAT):
        return cp.zeros([self._num_sites, self._num_sites] + shape, dtype)

    def _fsint(self, target):
        n = target.shape[0]
        # If we do not add point for i=0, value at i=1 will be ignored
        # as the integration turn from f0*sin(0) + f1*sin(k)
        # to f1*sin(0) + f2*sin(k)
        # As the function of target has form i * fi, so the value for i = 0 is 0
        fft_target = cp.hstack([cp.array([0]), target])
        res = -cp.imag(fft.fft(fft_target, n=(2 * n + 1)))[1 : n + 1]
        return res

    def _transform_matrix(self, matrix, mode):
        res = self._get_empty_matrix(self._grid.shape)
        if mode == "forward":
            vec = self._transform_coefficient[0]
            factor = self._transform_coefficient[1]
        elif mode == "backward":
            vec = self._transform_coefficient[2]
            factor = self._transform_coefficient[3]
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                res[i, j] = self._fsint(matrix[i, j] * vec) * factor
        return res

    def _get_bond_length(self):
        length = self._get_empty_matrix([])
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                length[i, j] = CUPY_FLOAT(
                    self._solvent.get_bond(self._site_list[i], self._site_list[j])
                    if i != j
                    else 0
                )
        return length

    def _get_w_k_matrix(self):
        freq = fft.fftfreq(self._grid.shape[0], self._grid.dr)
        w = self._get_empty_matrix(self._grid.shape, dtype=CUPY_FLOAT)
        for i in range(self._num_sites):
            for j in range(i, self._num_sites):
                if i != j:
                    # Denominator ensure the convolution normalized
                    target = cp.exp(-2j * cp.pi * freq * self._bond_length[i, j])
                    target = cp.real(fft.ifft(target))
                    w[i, j] = target.copy()
                    w[j, i] = target.copy()
                else:
                    w[i, j] = cp.zeros_like(self._grid.r)
        return self._transform_matrix(w, "forward")

    def _get_exp_u_matrix(self):
        exp_u = self._get_empty_matrix(self._grid.shape, dtype=CUPY_FLOAT)
        for i in range(self._num_sites):
            type1 = self._solvent.get_particle(self._site_list[i])["type"]
            for j in range(i, self._num_sites):
                type2 = self._solvent.get_particle(self._site_list[j])["type"]
                vdw = VDWPotential(type1, type2)
                exp_u[i, j] = cp.exp(-self._beta * vdw.evaluate(self._grid.r, -1))
                if i != j:
                    exp_u[j, i] = exp_u[i, j].copy()
        return exp_u

    def _get_gamma_matrix(self, c_k_matrix, gamma_k_matrix):
        gamma_matrix = self._get_empty_matrix(self._grid.shape, dtype=CUPY_FLOAT)
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                gamma_matrix[i, j] = self._get_site_gamma_k(
                    c_k_matrix, gamma_k_matrix, i, j
                )
        return self._transform_matrix(gamma_matrix, "backward")

    def _get_site_gamma_k(self, c_k_matrix, gamma_k_matrix, site1, site2):
        factor = self._rho_b
        denominator = 1 - factor * c_k_matrix[site1, site1]
        nominator = cp.zeros_like(denominator, dtype=CUPY_FLOAT)
        for i in range(self._num_sites):
            if i != site1:
                nominator += c_k_matrix[site1, i] * (
                    self._w_k[i, site2]
                    + factor * (gamma_k_matrix[i, site2] + c_k_matrix[i, site2])
                )
            else:
                nominator += c_k_matrix[site1, i] * (
                    self._w_k[i, site2] + factor * c_k_matrix[i, site2]
                )

        return nominator / denominator

    def _get_c_matrix(self, gamma_matrix):
        c_matrix = self._get_empty_matrix(self._grid.shape, dtype=CUPY_FLOAT)
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                c_matrix[i, j] = self._closure(self._exp_u[i, j], gamma_matrix[i, j])
        return c_matrix

    def _rism_forward(self, gamma_matrix, c_k_matrix):
        gamma_k_matrix = self._transform_matrix(gamma_matrix, "forward")
        gamma_matrix_new = self._get_gamma_matrix(c_k_matrix, gamma_k_matrix)
        residual = gamma_matrix - gamma_matrix_new
        return gamma_matrix_new, residual

    def _check_and_log(self, epoch, residual, error_tolerance):
        print("Iteration %d, Residual %.3e" % (epoch, residual))
        is_finished = residual < error_tolerance
        if is_finished:
            print(
                "Stop iterate at %d steps, residual %.3e smaller than tolerance %.3e"
                % (epoch, residual, error_tolerance)
            )
        return is_finished

    def solve(
        self,
        max_iterations,
        error_tolerance=1e-5,
        log_freq=10,
        subspace_size=10,
    ):
        """conduct Picard iteration for the 1D-RISM equation.

        Args:
            max_iterations (`int`): max number of iterations
            error_tolerance (`float`, optional): the error tolerance of iteration. Defaults to 1e-5.
            log_freq (`int`, optional): frequency of showing residual, no verbose when set to -1. Defaults to 10.
            subspace_size (`int`, optional): The number of history vector will be used in the DIIS. Defaults to 10.

        Returns:
            `tuple`: (h, c)
        """

        gamma_list, residual_list = [], []

        s = time.time()
        # Initialization of subspace
        gamma_matrix = self._get_empty_matrix(self._grid.shape, dtype=CUPY_FLOAT)
        for i in range(5 if 5 < subspace_size else subspace_size):
            c_matrix = self._get_c_matrix(gamma_matrix)
            c_k_matrix = self._transform_matrix(c_matrix, "forward")
            gamma_matrix_new, residual = self._rism_forward(gamma_matrix, c_k_matrix)
            gamma_list.append(gamma_matrix)
            residual_list.append(residual)
            gamma_matrix = gamma_matrix_new.copy()

        epoch, is_finished = 0, False
        while epoch < max_iterations and not is_finished:
            # DIIS
            cur_subspace_size = len(residual_list)
            A = cp.zeros((cur_subspace_size + 1, cur_subspace_size + 1), CUPY_FLOAT)
            for i in range(cur_subspace_size):
                for j in range(i, cur_subspace_size):
                    A[i, j] = (residual_list[i] * residual_list[j]).mean()
                    A[j, i] = A[i, j]
            A[-1, :], A[:, -1] = -1, -1
            A[-1, -1] = 0
            b = cp.zeros((cur_subspace_size + 1, 1), CUPY_FLOAT)
            b[-1, 0] = -1
            # Solve
            factor = cp.linalg.solve(A, b)
            gamma_matrix = cp.zeros_like(gamma_matrix)
            for i in range(cur_subspace_size):
                gamma_matrix += factor[i] * gamma_list[i]
            c_k_matrix = self._transform_matrix(
                self._get_c_matrix(gamma_matrix), "forward"
            )
            gamma_matrix, _ = self._rism_forward(gamma_matrix, c_k_matrix)
            c_k_matrix = self._transform_matrix(
                self._get_c_matrix(gamma_matrix), "forward"
            )
            _, residual = self._rism_forward(gamma_matrix, c_k_matrix)
            gamma_list.append(gamma_matrix.copy())
            residual_list.append(residual.copy())
            epoch += 2

            gamma_list = gamma_list[-subspace_size:]
            residual_list = residual_list[-subspace_size:]
            if epoch % log_freq == 0:
                residual = float(cp.sqrt(((residual) ** 2).mean()).get())
                is_finished = self._check_and_log(epoch, residual, error_tolerance)
        e = time.time()
        print("Run solve() for %s s" % (e - s))
        c_matrix = self._transform_matrix(c_k_matrix, "backward")
        return (gamma_matrix + c_matrix), c_matrix

    @property
    def site_list(self):
        return self._site_list

    def visualize(self, gamma_matrix, c_matrix):
        import matplotlib.pyplot as plt

        g_matrix = gamma_matrix + c_matrix + 1
        r = self._grid.r.get()
        fig, ax = plt.subplots(self._num_sites, self._num_sites, figsize=[16, 16])
        y_max = g_matrix.max().get() * 1.1
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                # ax[i, j].plot(r, gamma_matrix[i, j].get(), ".-", label=r"$\gamma$")
                # ax[i, j].plot(r, c_matrix[i, j].get(), ".-", label="c")
                ax[i, j].plot(r, g_matrix[i, j].get(), ".-", label="g")
                ax[i, j].set_title("%s-%s" % (self._site_list[i], self._site_list[j]))
                ax[i, j].legend()
                ax[i, j].set_ylim(0, y_max)
        fig.tight_layout()
        plt.show()
