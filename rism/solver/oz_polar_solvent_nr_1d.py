#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : oz_polar_solvent_picard_1d.py
created time : 2023/04/21
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import time
import cupy as cp
from numpy import dtype
import torch as tc
import torch.fft as fft
from torch.autograd import grad
from rism.environment import CUPY_FLOAT, NUMPY_FLOAT, TORCH_FLOAT
from rism.core import FFTGrid
from rism.potential import VDWPotential, ElePotential
from rism.unit import *


class OZPolarSolventNR1DSolver:
    def __init__(
        self,
        grid: FFTGrid,
        closure,
        basis_set,
        solvent_type: str,
        rho_b: Quantity,
        temperature=Quantity(300, kelvin),
        device=tc.device("cuda"),
    ) -> None:
        """Create solver for a Ornstein-Zernike equation in 1D spherical coordinate system using NR iteration

        Reference:
            Gillan, M. J. A new method of solving the liquid structure integral equations. Molecular Physics 38, 1781-1794 (1979).

        Args:
            grid (FFTGrid): The grid defining the coordinate system
            closure (Any): The closure for OZ equation from rism.closure
            basis_set (list): List of basis functions
            solvent_type (str): particle type of the solvent
            rho_b (`rism.unit.Quantity` or `float`): density of solvent in bulk, Unit: mol_dimension/length_dimension**3
            temperature (`rism.unit.Quantity` or `float`, optional): _description_. Defaults to Quantity(300, kelvin).
        """
        # Read input
        self._device = device
        self._grid = grid
        self._closure = closure
        self._basis_set = [self._tensor_from_cupy(i) for i in basis_set]
        self._solvent_type = solvent_type
        self._rho_b = NUMPY_FLOAT(
            (check_quantity(rho_b, mol / decimeter**3) * NA)
            .convert_to(1 / default_length_unit**3)
            .value
        )
        self._beta = NUMPY_FLOAT(
            1
            / (check_quantity(temperature, kelvin) * KB)
            .convert_to(default_energy_unit)
            .value
        )
        # Other attributes
        self._conjugate_set = self._get_conjugate_set(self._basis_set)
        self._num_basis = len(self._basis_set)
        self._r = self._tensor_from_cupy(self._grid.r)

    def _tensor_from_cupy(self, target: cp.ndarray):
        return tc.tensor(target.get(), dtype=TORCH_FLOAT, device=self._device)

    def _cupy_from_tensor(self, target: tc.Tensor):
        return cp.array(target.detach().cpu().numpy(), CUPY_FLOAT)

    def _zeros(self, shape):
        return tc.zeros(shape, dtype=TORCH_FLOAT, device=self._device)

    def _fsint(self, target):
        n = target.shape[0]
        # If we do not add point for i=0, value at i=1 will be ignored
        # as the integration turn from f0*sin(0) + f1*sin(k)
        # to f1*sin(0) + f2*sin(k)
        # As the function of target has form i * fi, so the value for i = 0 is 0
        pad = self._tensor_from_cupy(cp.array([0]))
        fft_target = tc.hstack([pad, target])
        res = -tc.imag(fft.fft(fft_target, n=(2 * n + 1)))[1 : n + 1]
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

    def _get_u(self, alpha=None):
        j_vec = cp.arange(1, self._grid.shape[0] + 1)
        k = j_vec * cp.pi / (self._grid.shape[0] + 1) / self._grid.dr
        vdw = VDWPotential(self._solvent_type, self._solvent_type)
        ele = ElePotential(self._solvent_type, self._solvent_type, alpha=alpha)
        u_s = ele.evaluate_sr(self._grid.r, -1)
        u_s += vdw.evaluate(self._grid.r, -1)
        u_l = ele.evaluate_lr(self._grid.r, -1)
        u_l_k = ele.evaluate_lr_k(k)
        return (
            self._tensor_from_cupy(u_s),
            self._tensor_from_cupy(u_l),
            self._tensor_from_cupy(u_l_k),
        )

    def _check_and_log(self, epoch, residual, error_tolerance):
        print("Iteration %d, Residual %.3e" % (epoch, residual))
        is_finished = residual < error_tolerance
        if residual < error_tolerance:
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
        restart_value=None,
        nr_max_iterations=5,
        nr_step_size=0.1,
        nr_tolerance=1e-3,
        alpha=None,
    ):
        """conduct Newton-Raphson iteration for the 1D-Ornstein-Zernike equation.

        Args:
            max_iterations (`int`): max number of iterations
            error_tolerance (`float`, optional): the error tolerance of iteration. Defaults to 1e-5.
            log_freq (`int`, optional): frequency of showing residual, no verbose when set to -1. Defaults to 10.
            restart_value (`tuple`, optional): (h, c) to define start point of iterations. Defaults to None as using an internal initial guess.
            nr_max_iterations (`int`, optional): number of iteration of inner Newton-Raphson iteration. Defaults to 10.
            nr_step_size (`float`, optional): step size of Newton-Raphson iteration. Defaults to 0.1.
            nr_tolerance (`float`, optional): tolerance of d_alpha in the Newton-Raphson iteration. Defaults to 1e-3.

        Returns:
            `tuple`: (h, c)
        """
        # To avoid the singularity point in zero index, we use 1:N+1 as the index
        # Hence the lr, appearing in calculation of k, is dr * (N+1)
        lr = self._grid.dr * (self._grid.shape[0] + 1)
        i_vec = self._tensor_from_cupy(cp.arange(1, self._grid.shape[0] + 1))
        j_vec = self._tensor_from_cupy(cp.arange(1, self._grid.shape[0] + 1))
        forward_factor = (4 * self._grid.dr**2 * lr) / j_vec
        backward_factor = 1 / (2 * lr**2 * self._grid.dr) / i_vec

        u_s, u_l, u_l_k = self._get_u(alpha=alpha)
        exp_u_s = tc.exp(-self._beta * u_s)
        scaled_u_l_k = self._beta * u_l_k

        if restart_value is None:
            gamma_s = self._zeros(self._grid.shape)
            c_s = self._closure(exp_u_s, gamma_s)
            c_s_k = self._fsint(c_s * i_vec) * forward_factor
            c_k = c_s_k - scaled_u_l_k
            h_k = c_k / (1 - self._rho_b * c_k)
            h = self._fsint(h_k * j_vec) * backward_factor
            gamma_s = h - c_s
        else:
            h, c_s = restart_value
            gamma_s = self._tensor_from_cupy(h - c_s)

        # Initialization decomposition
        alpha = self._zeros(self._num_basis)
        delta_gamma_s = tc.clone(gamma_s)
        for i in range(self._num_basis):
            alpha[i] = (self._conjugate_set[i] * gamma_s).sum()
            delta_gamma_s -= alpha[i] * self._basis_set[i]
        alpha.requires_grad_(True)

        total_epoch, is_finished = 0, False
        min_residual, min_epoch, min_res = 1, 0, []
        s = time.time()
        while total_epoch < max_iterations and not is_finished:
            nr_epoch = 0
            while nr_epoch < nr_max_iterations and not is_finished:
                # New gamma from alpha and delta_gamma
                gamma_s = self._zeros(self._grid.shape)
                for i in range(self._num_basis):
                    gamma_s += alpha[i] * self._basis_set[i]
                gamma_s += delta_gamma_s
                # c
                c_s = self._closure(exp_u_s, gamma_s)
                c_s_k = self._fsint(c_s * i_vec) * forward_factor
                c_k = c_s_k - scaled_u_l_k
                # gamma'
                h_k = c_k / (1 - self._rho_b * c_k)
                h = self._fsint(h_k * j_vec) * backward_factor
                gamma_s_prime = h - c_s

                # Newton-Raphson for new {a}
                alpha_prime = self._zeros(self._num_basis)
                for i in range(self._num_basis):
                    alpha_prime[i] = (self._conjugate_set[i] * gamma_s_prime).sum()
                # Loss
                loss = (alpha - alpha_prime).abs()
                nr_residual = loss.mean().detach()
                jacobian = self._zeros((self._num_basis, self._num_basis))
                for i in range(self._num_basis):
                    is_retain = i != (self._num_basis - 1)
                    jacobian[i, :] = grad(loss[i], alpha, retain_graph=is_retain)[0]
                inv_jacobian, is_un_inv = tc.linalg.inv_ex(jacobian)
                if is_un_inv:
                    print(
                        "\t(Inner NR) Stop NR iterate at %d steps, Singularity Jacobian"
                        % total_epoch
                    )
                    alpha = tc.clone(alpha)
                    alpha.requires_grad_(True)
                    break
                alpha = alpha - tc.matmul(inv_jacobian, loss) * nr_step_size
                alpha.requires_grad_(True)

                nr_epoch += 1
                total_epoch += 1
                if nr_residual <= nr_tolerance:
                    print(
                        "\t(Inner NR) Stop NR iterate at %d steps, d_alpha %.3e smaller than tolerance %.3e"
                        % (total_epoch, nr_residual, nr_tolerance)
                    )
                    break

                # Verbose
                if total_epoch % log_freq == 0:
                    residual = float(tc.sqrt(((gamma_s - gamma_s_prime) ** 2).mean()))
                    if min_residual > residual:
                        min_residual = residual
                        min_epoch = total_epoch
                        min_res = [tc.clone(gamma_s), tc.clone(c_s)]
                    is_finished = self._check_and_log(
                        total_epoch, residual, error_tolerance
                    )
                    is_finished |= residual >= 10 * min_residual

            # delta_gamma_prime
            delta_gamma_s = tc.clone(gamma_s_prime)
            for i in range(self._num_basis):
                delta_gamma_s -= alpha[i] * self._basis_set[i]

        e = time.time()
        print(
            "Residual %.3e achieve the minimum at step %d" % (min_residual, min_epoch)
        )
        print("Run solve() for %s s" % (e - s))
        h = min_res[0] + min_res[1]
        return self._cupy_from_tensor(h), self._cupy_from_tensor(min_res[1])

    def visualize(self, gamma, c):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=[25, 9])
        g = gamma + c + 1

        r = self._grid.r.get()
        ax[0].plot(r, g.get(), ".-")
        ax[1].plot(r, c.get(), ".-")
        ax[2].plot(r, gamma.get(), ".-")
        ax[0].set_title("g")
        ax[1].set_title("c")
        ax[2].set_title(r"$\gamma$")
        plt.show()
