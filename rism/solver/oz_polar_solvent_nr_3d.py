#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : oz_polar_solvent_nr_3d.py
created time : 2023/04/18
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import time
import cupy as cp
import torch as tc
import torch.fft as fft
from torch.autograd import grad
from rism.environment import CUPY_FLOAT, TORCH_FLOAT, NUMPY_FLOAT
from rism.core import FFTGrid
from rism.potential import VDWPotential, ElePotential
from rism.unit import *


class OZPolarSolventNr3DSolver:
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
        """Create solver for a 3D Ornstein-Zernike equation in 3D cartesian coordinate system using Picard iteration

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
        self._x = self._tensor_from_cupy(self._grid.x)
        self._y = self._tensor_from_cupy(self._grid.y)
        self._z = self._tensor_from_cupy(self._grid.z)

    def _tensor_from_cupy(self, target: cp.ndarray):
        return tc.tensor(target.get(), dtype=TORCH_FLOAT, device=self._device)

    def _cupy_from_tensor(self, target: tc.Tensor):
        return cp.array(target.detach().cpu().numpy(), CUPY_FLOAT)

    def _zeros(self, shape):
        return tc.zeros(shape, dtype=TORCH_FLOAT, device=self._device)

    def _get_u(self, coordinate):
        vdw = VDWPotential(self._solvent_type, self._solvent_type)
        ele = ElePotential(self._solvent_type, self._solvent_type)
        r = cp.sqrt(
            (self._grid.x - coordinate[0]) ** 2
            + (self._grid.y - coordinate[1]) ** 2
            + (self._grid.z - coordinate[2]) ** 2
        )
        vdw = vdw.evaluate(r, -1)
        ele_long = ele.evaluate_long(r, -1)
        ele_short = ele.evaluate_short(r, -1)

        return self._tensor_from_cupy(vdw + ele_short), self._tensor_from_cupy(ele_long)

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

    def _get_convolve_shift(self):
        k, l, n = tc.meshgrid(
            fft.fftfreq(
                self._grid.shape[0],
                self._grid.dx,
                dtype=TORCH_FLOAT,
                device=self._device,
            ),
            fft.fftfreq(
                self._grid.shape[1],
                self._grid.dy,
                dtype=TORCH_FLOAT,
                device=self._device,
            ),
            fft.fftfreq(
                self._grid.shape[2],
                self._grid.dz,
                dtype=TORCH_FLOAT,
                device=self._device,
            ),
            indexing="ij",
        )
        shift = tc.exp(
            -2j
            * tc.pi
            * (
                k * self._grid.lx * 0.5
                + l * self._grid.ly * 0.5
                + n * self._grid.lz * 0.5
            )
        )
        return shift

    def _get_center_shift(self, coordinate):
        k, l, n = tc.meshgrid(
            fft.fftfreq(
                self._grid.shape[0],
                self._grid.dx,
                dtype=TORCH_FLOAT,
                device=self._device,
            ),
            fft.fftfreq(
                self._grid.shape[1],
                self._grid.dy,
                dtype=TORCH_FLOAT,
                device=self._device,
            ),
            fft.fftfreq(
                self._grid.shape[2],
                self._grid.dz,
                dtype=TORCH_FLOAT,
                device=self._device,
            ),
            indexing="ij",
        )
        shift = tc.exp(
            2j * tc.pi * (k * coordinate[0] + l * coordinate[1] + n * coordinate[2])
        )
        return shift

    def _check_and_log(self, epoch, residual, error_tolerance):
        print("Iteration %d;" % epoch, "Residual %.3e" % residual)
        is_finished = residual < error_tolerance
        if is_finished:
            print(
                "Stop iterate at %d steps, residual %.3e smaller than tolerance %.3e"
                % (epoch, residual, error_tolerance)
            )
        return is_finished

    def solve(
        self,
        coordinate,
        max_iterations,
        error_tolerance=1e-5,
        log_freq=10,
        restart_value=None,
        nr_max_iterations=5,
        nr_step_size=0.1,
        nr_tolerance=1e-3,
    ):
        """conduct Newton-Raphson iteration for the 3D-Ornstein-Zernike equation.

        Args:
            coordinate (`cp.ndarray` or `np.ndarray`): center coordinate of target
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
        # Initialize factor
        u = self._get_u(coordinate)
        u_short, u_long = self._get_u(coordinate)
        exp_u_short = tc.exp(-self._beta * u_short)
        u_long_k = fft.fftn(u_long * self._beta)
        dv = self._grid.dx * self._grid.dy * self._grid.dz
        convolve_shift = self._get_convolve_shift()
        center_shift = self._get_center_shift(coordinate)
        factor = self._rho_b * dv * convolve_shift * center_shift

        if restart_value is None:
            gamma = self._zeros(self._grid.shape)
            c_short = self._closure(exp_u_short, gamma)
            c_short_k = fft.fftn(c_short)
            c_k = c_short_k - u_long_k
            gamma_k = (factor * c_k * c_short_k - u_long_k) / (1 - factor * c_k)
            gamma = tc.real(fft.ifftn(gamma_k))
        else:
            h, c = restart_value
            gamma = self._tensor_from_cupy(h - c)

        # Initialization decomposition
        alpha = self._zeros(self._num_basis)
        delta_gamma = tc.clone(gamma)
        for i in range(self._num_basis):
            alpha[i] = (self._conjugate_set[i] * gamma).sum()
            delta_gamma -= alpha[i] * self._basis_set[i]
        alpha.requires_grad_(True)

        total_epoch, is_finished = 0, False
        s = time.time()
        while total_epoch < max_iterations and not is_finished:
            nr_epoch = 0
            while nr_epoch < nr_max_iterations:
                # New gamma from alpha and delta_gamma
                gamma = self._zeros(self._grid.shape)
                for i in range(self._num_basis):
                    gamma += alpha[i] * self._basis_set[i]
                gamma += delta_gamma
                # c
                c_short = self._closure(exp_u_short, gamma)
                c_k = fft.fftn(c_short - u_long)
                # gamma'
                gamma_prime_k = factor * c_k**2 / (1 - factor * c_k) - u_long_k
                gamma_prime = tc.real(fft.ifftn(gamma_prime_k))

                # Newton-Raphson for new {a}
                alpha_prime = self._zeros(self._num_basis)
                for i in range(self._num_basis):
                    alpha_prime[i] = (self._conjugate_set[i] * gamma_prime).sum()
                # Loss
                loss = (alpha - alpha_prime).abs()
                nr_residual = loss.mean().detach()
                jacobian = self._zeros((self._num_basis, self._num_basis))
                for i in range(self._num_basis):
                    jacobian[i, :] = grad(loss[i], alpha, retain_graph=True)[0]
                dl_da = grad(loss.sum(), alpha)[0]
                inv_jacobian, is_un_inv = tc.linalg.inv_ex(jacobian)
                if is_un_inv:
                    print(
                        "\t(Inner NR) Singularity Jacobian at iteration %d"
                        % total_epoch
                    )
                    alpha = alpha - dl_da * 0.01
                else:
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
                    residual = float(tc.sqrt(((gamma - gamma_prime) ** 2).mean()))
                    is_finished = self._check_and_log(
                        total_epoch, residual, error_tolerance
                    )

            # delta_gamma_prime
            delta_gamma = tc.clone(gamma_prime)
            for i in range(self._num_basis):
                delta_gamma -= alpha[i] * self._basis_set[i]
        e = time.time()
        print("Run solve() for %s s" % (e - s))
        c = self._closure(exp_u_short, gamma)
        return self._cupy_from_tensor(c + gamma), self._cupy_from_tensor(c)
