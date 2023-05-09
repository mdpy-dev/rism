#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : rism_polar_solvent_nr_1d.py
created time : 2023/04/26
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import time
import cupy as cp
import torch as tc
import torch.fft as fft
from torch.autograd import grad
from rism.environment import CUPY_FLOAT, NUMPY_FLOAT, TORCH_FLOAT
from rism.core import FFTGrid, Solvent
from rism.potential import VDWPotential, ElePotential
from rism.unit import *


class RISMPolarSolventNR1DSolver:
    def __init__(
        self,
        grid: FFTGrid,
        closure,
        solvent: Solvent,
        basis_set,
        rho_b: Quantity,
        temperature=Quantity(300, kelvin),
        device=tc.device("cuda"),
    ) -> None:
        """Create solver for a RISM equation in 1D spherical coordinate system using NR iteration

        Reference:
            Gillan, M. J. A new method of solving the liquid structure integral equations. Molecular Physics 38, 1781-1794 (1979).

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
        self._device = device
        self._basis_set = [self._tensor_from_cupy(i) for i in basis_set]
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
        self._conjugate_set = self._get_conjugate_set(self._basis_set)
        self._num_basis = len(self._basis_set)
        self._r = self._tensor_from_cupy(self._grid.r)

        # To avoid the singularity point in zero index, we use 1:N+1 as the index
        # Hence the lr, appearing in calculation of k, is dr * (N+1)
        lr = self._grid.dr * (self._grid.shape[0] + 1)
        i_vec = self._tensor_from_cupy(cp.arange(1, self._grid.shape[0] + 1))
        j_vec = self._tensor_from_cupy(cp.arange(1, self._grid.shape[0] + 1))
        forward_factor = (4 * self._grid.dr**2 * lr) / j_vec
        backward_factor = 1 / (2 * lr**2 * self._grid.dr) / i_vec
        self._transform_coefficient = [i_vec, forward_factor, j_vec, backward_factor]

        self._site_list = self._solvent.particle_list
        self._bond_length = self._get_bond_length()
        self._w_k = self._get_w_k_matrix()

    def _get_empty_matrix(self, shape, dtype=TORCH_FLOAT):
        return tc.zeros(
            [self._num_sites, self._num_sites] + shape,
            dtype=dtype,
            device=self._device,
        )

    def _zeros(self, shape):
        return tc.zeros(shape, dtype=TORCH_FLOAT, device=self._device)

    def _tensor_from_cupy(self, target: cp.ndarray):
        return tc.tensor(target.get(), dtype=TORCH_FLOAT, device=self._device)

    def _cupy_from_tensor(self, target: tc.Tensor):
        return cp.array(target.detach().cpu().numpy(), CUPY_FLOAT)

    def _fsint(self, target):
        n = target.shape[0]
        # If we do not add point for i=0, value at i=1 will be ignored
        # as the integration turn from f0*sin(0) + f1*sin(k)
        # to f1*sin(0) + f2*sin(k)
        # As the function of target has form i * fi, so the value for i = 0 is 0
        pad = tc.tensor([0], dtype=TORCH_FLOAT, device=self._device)
        fft_target = tc.hstack([pad, target])
        res = -tc.imag(fft.fft(fft_target, n=(2 * n + 1)))[1 : n + 1]
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

    def _get_conjugate_set(self, basis_set):
        num_basis = len(basis_set)
        conv_matrix = self._zeros((num_basis, num_basis))
        for i in range(num_basis):
            for j in range(num_basis):
                conv_matrix[i, j] = (basis_set[i] * basis_set[j]).sum()
        B = tc.linalg.inv(conv_matrix)
        conjugate_set = []
        for i in range(num_basis):
            Q = self._zeros(self._grid.shape)
            for j in range(num_basis):
                Q += B[i, j] * basis_set[j]
            conjugate_set.append(Q)
        return conjugate_set

    def _get_bond_length(self):
        length = self._get_empty_matrix([])
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                length[i, j] = (
                    self._solvent.get_bond(self._site_list[i], self._site_list[j])
                    if i != j
                    else 0
                )
        return length

    def _get_w_k_matrix(self):
        freq = fft.fftfreq(self._grid.shape[0], self._grid.dr).to(self._device)
        w = self._get_empty_matrix(self._grid.shape, dtype=TORCH_FLOAT)
        for i in range(self._num_sites):
            for j in range(i, self._num_sites):
                if i != j:
                    # Denominator ensure the convolution normalized
                    target = tc.exp(-2j * tc.pi * freq * self._bond_length[i, j])
                    target = tc.real(fft.ifft(target))
                    w[i, j] = tc.clone(target)
                    w[j, i] = tc.clone(target)
                else:
                    w[i, j] = tc.zeros_like(self._r)
        return self._transform_matrix(w, "forward")

    def _get_exp_u_s_matrix(self, alpha=None):
        exp_u_s = self._get_empty_matrix(self._grid.shape, dtype=TORCH_FLOAT)
        for i in range(self._num_sites):
            type1 = self._solvent.get_particle(self._site_list[i])["type"]
            for j in range(i, self._num_sites):
                type2 = self._solvent.get_particle(self._site_list[j])["type"]
                ele = ElePotential(type1, type2, alpha=alpha)
                vdw = VDWPotential(type1, type2)
                u = ele.evaluate_sr(self._grid.r, -1) + vdw.evaluate(self._grid.r, -1)
                exp_u_s[i, j] = self._tensor_from_cupy(cp.exp(-self._beta * u))
                if i != j:
                    exp_u_s[j, i] = tc.clone(exp_u_s[i, j])
        return exp_u_s

    def _get_u_l_k_matrix(self, alpha=None):
        j_vec = cp.arange(1, self._grid.shape[0] + 1)
        k = j_vec * cp.pi / (self._grid.shape[0] + 1) / self._grid.dr
        u_l_k_matrix = self._get_empty_matrix(self._grid.shape, dtype=TORCH_FLOAT)
        for i in range(self._num_sites):
            type1 = self._solvent.get_particle(self._site_list[i])["type"]
            for j in range(i, self._num_sites):
                type2 = self._solvent.get_particle(self._site_list[j])["type"]
                ele = ElePotential(type1, type2, alpha=alpha)
                u_l_k_matrix[i, j] = self._tensor_from_cupy(
                    self._beta * ele.evaluate_lr_k(k)
                )
                if i != j:
                    u_l_k_matrix[j, i] = tc.clone(u_l_k_matrix[i, j])
        return u_l_k_matrix

    def _get_h_matrix(self, h_k_matrix, c_k_matrix):
        h_matrix = self._get_empty_matrix(self._grid.shape, dtype=TORCH_FLOAT)
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                h_matrix[i, j] = self._get_site_h_k(h_k_matrix, c_k_matrix, i, j)
        return self._transform_matrix(h_matrix, "backward")

    def _get_site_h_k(self, h_k_matrix, c_k_matrix, site1, site2):
        denominator = 1 - self._rho_b * c_k_matrix[site1, site1]
        nominator = tc.zeros_like(denominator, dtype=TORCH_FLOAT)
        for i in range(self._num_sites):
            if i != site1:
                nominator += c_k_matrix[site1, i] * (
                    self._w_k[i, site2] + self._rho_b * h_k_matrix[i, site2]
                )
            else:
                nominator += c_k_matrix[site1, i] * self._w_k[i, site2]

        return nominator / denominator

    def _get_c_matrix(self, gamma_matrix, exp_u_matrix):
        c_matrix = self._get_empty_matrix(self._grid.shape, dtype=TORCH_FLOAT)
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                c_matrix[i, j] = self._closure(exp_u_matrix[i, j], gamma_matrix[i, j])
        return c_matrix

    def _rism_forward(self, gamma_s_matrix, c_s_matrix, u_l_k_matrix):
        h_matrix = gamma_s_matrix + c_s_matrix
        h_k_matrix = self._transform_matrix(h_matrix, "forward")
        c_s_k_matrix = self._transform_matrix(c_s_matrix, "forward")
        c_k_matrix = c_s_k_matrix - u_l_k_matrix
        h = self._get_h_matrix(h_k_matrix, c_k_matrix)
        gamma_s_matrix_new = h - c_s_matrix
        residual = gamma_s_matrix - gamma_s_matrix_new
        return gamma_s_matrix_new, residual

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
        alpha=0.9,
        nr_max_iterations=5,
        nr_step_size=0.1,
        nr_tolerance=1e-3,
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
        exp_u_s_matrix = self._get_exp_u_s_matrix(alpha)
        u_l_k_matrix = self._get_u_l_k_matrix(alpha)

        s = time.time()
        gamma_s_matrix = self._get_empty_matrix(self._grid.shape, dtype=TORCH_FLOAT)
        c_s_matrix = self._get_c_matrix(gamma_s_matrix, exp_u_s_matrix)

        # Initialization decomposition
        alpha_matrix = self._get_empty_matrix([self._num_basis])
        delta_gamma_s_matrix = tc.clone(gamma_s_matrix)
        for i in range(self._num_sites):
            for j in range(self._num_sites):
                for k in range(self._num_basis):
                    alpha_matrix[i, j, k] = (
                        self._conjugate_set[k] * gamma_s_matrix[i, j]
                    ).sum()
                    delta_gamma_s_matrix[i, j] -= (
                        alpha_matrix[i, j, k] * self._basis_set[k]
                    )
        alpha_matrix.requires_grad_(True)

        total_epoch, is_finished = 0, False
        while total_epoch < max_iterations and not is_finished:
            nr_epoch = 0
            is_un_inv, is_within_nr_tolerance = False, False
            while (
                nr_epoch < nr_max_iterations
                and not is_un_inv
                and not is_within_nr_tolerance
            ):
                # New gamma from alpha and delta_gamma
                gamma_s_matrix = self._get_empty_matrix(self._grid.shape, TORCH_FLOAT)
                for i in range(self._num_sites):
                    for j in range(self._num_sites):
                        for k in range(self._num_basis):
                            gamma_s_matrix[i, j] += (
                                alpha_matrix[i, j, k] * self._basis_set[k]
                            )
                gamma_s_matrix += delta_gamma_s_matrix
                gamma_s_prime_matrix, residual = self._rism_forward(
                    gamma_s_matrix, c_s_matrix, u_l_k_matrix
                )
                # Newton-Raphson for new {a}
                alpha_prime_matrix = self._get_empty_matrix(
                    [self._num_basis], TORCH_FLOAT
                )
                for i in range(self._num_sites):
                    for j in range(self._num_sites):
                        for k in range(self._num_basis):
                            alpha_prime_matrix[i, j, k] = (
                                self._conjugate_set[k] * gamma_s_prime_matrix[i, j]
                            ).sum()
                # Loss
                alpha_matrix = alpha_matrix.view(-1)
                alpha_prime_matrix = alpha_prime_matrix.view(-1)
                loss = (alpha_matrix - alpha_prime_matrix).abs()
                nr_residual = loss.mean().detach()
                is_within_tolerance = nr_residual < nr_tolerance
                num_vars = self._num_basis * self._num_sites**2
                inv_jacobian = self._zeros((num_vars, num_vars))
                for i in range(num_vars):
                    is_retain = i != (num_vars - 1)
                    inv_jacobian[i, :] = grad(
                        loss[i],
                        alpha_matrix,
                        retain_graph=is_retain,
                    )[0]
                inv_jacobian, is_un_inv = tc.linalg.inv_ex(inv_jacobian)
                nr_epoch += 1
                total_epoch += 1
                if total_epoch % log_freq == 0:
                    is_finished = self._check_and_log(
                        total_epoch, tc.sqrt((residual**2).mean()), error_tolerance
                    )
                if is_un_inv:
                    print(
                        "\t(Inner NR) Stop NR iterate at %d steps, Singularity Jacobian"
                        % total_epoch
                    )
                elif is_within_tolerance:
                    print(
                        "\t(Inner NR) Stop NR iterate at %d steps, d_alpha %.3e smaller than tolerance %.3e"
                        % (total_epoch, nr_residual, nr_tolerance)
                    )
                else:
                    alpha_matrix = (
                        alpha_matrix - tc.matmul(inv_jacobian, loss) * nr_step_size
                    )
                alpha_matrix = tc.clone(
                    alpha_matrix.view(self._num_sites, self._num_sites, self._num_basis)
                )
                alpha_matrix.requires_grad_(True)
            # delta_gamma_prime
            delta_gamma_s_matrix = tc.clone(gamma_s_matrix)
            for i in range(self._num_sites):
                for j in range(self._num_sites):
                    for k in range(self._num_basis):
                        delta_gamma_s_matrix[i, j] -= (
                            alpha_matrix[i, j, k] * self._basis_set[k]
                        )
        e = time.time()
        print("Run solve() for %s s" % (e - s))
        h_matrix = gamma_s_matrix + c_s_matrix
        return self._cupy_from_tensor(h_matrix), self._cupy_from_tensor(c_s_matrix)

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
