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
import cupyx.scipy.fft as fft
from rism.environment import CUPY_FLOAT
from rism.core import FFTGrid
from rism.potential import VDWPotential, ElePotential
from rism.unit import *


class OZPolarSolventPicard1DSolver:
    def __init__(
        self,
        grid: FFTGrid,
        closure,
        solvent_type: str,
        rho_b: Quantity,
        temperature=Quantity(300, kelvin),
    ) -> None:
        """Create solver for a 3D Ornstein-Zernike equation in 3D cartesian coordinate system using Picard iteration

        Args:
            grid (FFTGrid): The grid defining the coordinate system
            closure (Any): The closure for OZ equation from rism.closure
            solvent_type (str): particle type of the solvent
            rho_b (`rism.unit.Quantity` or `float`): density of solvent in bulk, Unit: mol_dimension/length_dimension**3
            temperature (`rism.unit.Quantity` or `float`, optional): _description_. Defaults to Quantity(300, kelvin).
        """
        # Read input
        self._grid = grid
        self._closure = closure
        self._solvent_type = solvent_type
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

    def _get_u(self, alpha=None):
        j_vec = cp.arange(1, self._grid.shape[0] + 1)
        k = j_vec * cp.pi / (self._grid.shape[0] + 1) / self._grid.dr
        vdw = VDWPotential(self._solvent_type, self._solvent_type)
        ele = ElePotential(self._solvent_type, self._solvent_type, alpha=alpha)
        u_s = ele.evaluate_sr(self._grid.r, -1)
        u_s += vdw.evaluate(self._grid.r, -1)
        u_l = ele.evaluate_lr(self._grid.r, -1)
        u_l_k = ele.evaluate_lr_k(k)
        return u_s.astype(CUPY_FLOAT), u_l.astype(CUPY_FLOAT), u_l_k.astype(CUPY_FLOAT)

    def _fsint(self, target):
        n = target.shape[0]
        # If we do not add point for i=0, value at i=1 will be ignored
        # as the integration turn from f0*sin(0) + f1*sin(k)
        # to f1*sin(0) + f2*sin(k)
        # As the function of target has form i * fi, so the value for i = 0 is 0
        fft_target = cp.hstack([cp.array([0]), target])
        res = -cp.imag(fft.fft(fft_target, n=(2 * n + 1)))[1 : n + 1]
        return res

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
        alt=0.8,
        alpha=None,
        restart_value=None,
    ):
        """conduct Picard iteration for the 3D-Ornstein-Zernike equation.

        Args:
            coordinate (`cp.ndarray` or `np.ndarray`): center coordinate of target
            max_iterations (`int`): max number of iterations
            error_tolerance (`float`, optional): the error tolerance of iteration. Defaults to 1e-5.
            log_freq (`int`, optional): frequency of showing residual, no verbose when set to -1. Defaults to 10.
            alt (`float`, optional): relaxation coefficient. Defaults to 0.9.
            restart_value (`tuple`, optional): (h, c) to define start point of iterations. Defaults to None as using an internal initial guess.

        Returns:
            `tuple`: (h, c)
        """
        # To avoid the singularity point in zero index, we use 1:N+1 as the index
        # Hence the lr, appearing in calculation of k, is dr * (N+1)
        lr = self._grid.dr * (self._grid.shape[0] + 1)
        i_vec = cp.arange(1, self._grid.shape[0] + 1)
        j_vec = cp.arange(1, self._grid.shape[0] + 1)
        forward_factor = (4 * self._grid.dr**2 * lr) / j_vec
        backward_factor = 1 / (2 * lr**2 * self._grid.dr) / i_vec

        u_s, u_l, u_l_k = self._get_u(alpha=alpha)
        exp_u_s = cp.exp(-self._beta * u_s).astype(CUPY_FLOAT)
        scaled_u_l_k = self._beta * u_l_k

        if restart_value is None:
            gamma_s = self._grid.zeros_field()
            c_s = self._closure(exp_u_s, gamma_s)
        else:
            h, c_s = restart_value
            gamma_s = h - c_s

        epoch, is_finished = 0, False
        alta, altb = CUPY_FLOAT(alt), CUPY_FLOAT(1 - alt)
        s = time.time()

        while epoch < max_iterations and not is_finished:
            c_s_k = self._fsint(c_s * i_vec) * forward_factor
            c_k = c_s_k - scaled_u_l_k
            h_k = c_k / (1 - self._rho_b * c_k)
            h = self._fsint(h_k * j_vec) * backward_factor
            gamma_s, gamma_s_pre = (h - c_s), gamma_s
            gamma_s = gamma_s * alta + gamma_s_pre * altb
            c_s = self._closure(exp_u_s, gamma_s)
            if epoch % log_freq == 0:
                # self.visualize(gamma_s, c_s)
                residual = float(cp.sqrt(((gamma_s - gamma_s_pre) ** 2).mean()).get())
                is_finished = self._check_and_log(epoch, residual, error_tolerance)
            epoch += 1

        e = time.time()
        print("Run solve() for %s s" % (e - s))
        h = gamma_s + c_s
        return h, c_s

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
