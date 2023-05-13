#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : rism_solvent_diss_1d.py
created time : 2023/04/19
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
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
        """Create solver for RISM equation in 1D spherical coordinate system using DIIS method

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
        self._k = j_vec * cp.pi / (self._grid.shape[0] + 1) / self._grid.dr

        self._bond_length = self._get_bond_length()
        self._w_k = self._get_w_k_matrix()
        self._exp_u = self._get_exp_u_matrix()

    def _get_empty_matrix(self, shape, dtype=CUPY_FLOAT):
        return cp.zeros([self._num_sites, self._num_sites] + shape, dtype)

    def _matmul(self, A, B):
        res = self._get_empty_matrix(self._grid.shape, dtype=CUPY_FLOAT)
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                for k in range(self._num_sites):
                    res[i, j] += A[i, k] * B[k, j]
        return res

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
                if i != j:
                    length[i, j] = CUPY_FLOAT(self._solvent.get_bond_length(i, j))
        return length

    def _get_w_k_matrix(self):
        w_k = self._get_empty_matrix(self._grid.shape, dtype=CUPY_FLOAT)
        for i in range(self._num_sites):
            for j in range(i, self._num_sites):
                if i != j:
                    kr = self._k * self._bond_length[i, j]
                    target = cp.sin(kr) / kr
                    w_k[i, j] = target.copy()
                    w_k[j, i] = target.copy()
                else:
                    w_k[i, j] = cp.ones(self._grid.shape)
        return w_k

    def _get_exp_u_matrix(self):
        exp_u = self._get_empty_matrix(self._grid.shape, dtype=CUPY_FLOAT)
        for i in range(self._num_sites):
            particle1 = self._solvent.particle_list[i]
            for j in range(i, self._num_sites):
                particle2 = self._solvent.particle_list[j]
                vdw = VDWPotential(particle1, particle2)
                exp_u[i, j] = cp.exp(-self._beta * vdw.evaluate(self._grid.r, -1))
                if i != j:
                    exp_u[j, i] = exp_u[i, j].copy()
        return exp_u

    def _get_gamma_matrix(self, c_k_matrix):
        wc_k_matrix = self._matmul(self._w_k, c_k_matrix)
        wcw_k_matrix = self._matmul(wc_k_matrix, self._w_k)
        A = -wc_k_matrix * self._rho_b
        for i in range(self._num_sites):
            A[i, i] += 1
        inv_A = cp.linalg.inv(A.T).T
        h_k_matrix = self._matmul(inv_A, wcw_k_matrix)
        gamma_matrix = h_k_matrix - c_k_matrix
        return self._transform_matrix(gamma_matrix, "backward")

    def _get_c_matrix(self, gamma_matrix):
        c_matrix = self._get_empty_matrix(self._grid.shape, dtype=CUPY_FLOAT)
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                c_matrix[i, j] = self._closure(self._exp_u[i, j], gamma_matrix[i, j])
        return c_matrix

    def _rism_forward(self, gamma_matrix, c_matrix):
        c_k_matrix = self._transform_matrix(c_matrix, "forward")
        gamma_matrix_new = self._get_gamma_matrix(c_k_matrix)
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
        for i in range(2):
            c_matrix = self._get_c_matrix(gamma_matrix)
            gamma_matrix_new, residual = self._rism_forward(gamma_matrix, c_matrix)
            gamma_list.append(gamma_matrix)
            residual_list.append(residual)
            gamma_matrix = gamma_matrix_new * 0.7 + gamma_matrix_new * 0.3

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
            c_matrix = self._get_c_matrix(gamma_matrix)
            gamma_matrix, _ = self._rism_forward(gamma_matrix, c_matrix)
            c_matrix = self._get_c_matrix(gamma_matrix)
            _, residual = self._rism_forward(gamma_matrix, c_matrix)
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
        return (gamma_matrix + c_matrix), c_matrix
