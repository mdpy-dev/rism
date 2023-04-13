#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : oz_solvent_picard_3d.py
created time : 2023/04/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import cupy as cp
import cupyx.scipy.fft as fft
from rism.environment import CUPY_FLOAT
from rism.core import FFTGrid
from rism.potential import VDWPotential
from rism.unit import *


class OZSolventPicard3DSolver:
    def __init__(
        self, grid: FFTGrid, solvent_type: str, temperature=Quantity(300, kelvin)
    ) -> None:
        """Create solver for a 3D Ornstein-Zernike equation in 3D cartesian coordinate system using Picard iteration

        Args:
            grid (FFTGrid): The grid defining the coordinate system
            solvent_type (str): particle type of the solvent
            temperature (`rism.unit.Quantity` or `float`, optional): _description_. Defaults to Quantity(300, kelvin).
        """
        # Read input
        self._grid = grid
        self._solvent_type = solvent_type
        # Add constant
        self._beta = CUPY_FLOAT(
            1
            / check_quantity_value(
                check_quantity(temperature, kelvin) * KB, default_energy_unit
            )
        )
        self._rho_b = CUPY_FLOAT(
            check_quantity_value(
                Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton),
                1 / default_length_unit**3,
            )
        )

    def _get_u(self, coordinate):
        vdw = VDWPotential(self._solvent_type, self._solvent_type)
        r = cp.sqrt(
            (self._grid.x - coordinate[0]) ** 2
            + (self._grid.y - coordinate[1]) ** 2
            + (self._grid.z - coordinate[2]) ** 2
        )
        u = vdw.evaluate(r, Quantity(10, kilocalorie_permol))
        return u.astype(CUPY_FLOAT)

    def _get_convolve_shift(self):
        k, l, n = cp.meshgrid(
            fft.fftfreq(self._grid.shape[0], self._grid.dx),
            fft.fftfreq(self._grid.shape[1], self._grid.dy),
            fft.fftfreq(self._grid.shape[2], self._grid.dz),
            indexing="ij",
        )
        shift = cp.exp(
            -2j
            * cp.pi
            * (
                k * self._grid.lx * 0.5
                + l * self._grid.ly * 0.5
                + n * self._grid.lz * 0.5
            )
        )
        return shift

    def _get_center_shift(self, coordinate):
        k, l, n = cp.meshgrid(
            fft.fftfreq(self._grid.shape[0], self._grid.dx),
            fft.fftfreq(self._grid.shape[1], self._grid.dy),
            fft.fftfreq(self._grid.shape[2], self._grid.dz),
            indexing="ij",
        )
        shift = cp.exp(
            2j * cp.pi * (k * coordinate[0] + l * coordinate[1] + n * coordinate[2])
        )
        return shift

    def solve(
        self,
        coordinate,
        iterations,
        error_tolerance=1e-5,
        verbose_freq=10,
        alt=0.9,
        restart_value=None,
    ):
        """conduct Picard iteration for the 3D-Ornstein-Zernike equation.

        Args:
            coordinate (`cp.ndarray` or `np.ndarray`): center coordinate of target
            iterations (`int`): number of iterations
            error_tolerance (`float`, optional): the error tolerance of iteration. Defaults to 1e-5.
            verbose_freq (`int`, optional): frequency of showing residual, no verbose when set to -1. Defaults to 10.
            alt (`float`, optional): relaxation coefficient. Defaults to 0.9.
            restart_value (`tuple`, optional): (h, c) to define start point of iterations. Defaults to None as using an internal initial guess.

        Returns:
            `tuple`: (h, c)
        """
        coordinate = cp.array(coordinate, CUPY_FLOAT)
        u = self._get_u(coordinate)
        exp_u = cp.exp(-self._beta * u).astype(CUPY_FLOAT)

        if restart_value is None:
            gamma = self._grid.zeros_field()
            c = (exp_u - CUPY_FLOAT(1)) * (gamma + CUPY_FLOAT(1))
        else:
            h, c = restart_value
            gamma = h - c

        dv = CUPY_FLOAT(self._grid.dx * self._grid.dy * self._grid.dz)
        convolve_shift = self._get_convolve_shift()
        center_shift = self._get_center_shift(coordinate)
        factor = self._rho_b * dv * convolve_shift * center_shift

        i, is_finished = 0, False
        alta, altb = CUPY_FLOAT(alt), CUPY_FLOAT(1 - alt)
        while i < iterations and not is_finished:
            ck = fft.fftn(c)
            gamma_k = factor * ck**2 / (1 - factor * ck)
            gamma_pre = gamma.copy()
            gamma = cp.real(fft.ifftn(gamma_k))
            gamma = gamma * alta + gamma_pre * altb
            c, c_pre = (exp_u - CUPY_FLOAT(1)) * (gamma + CUPY_FLOAT(1)), c

            if i % verbose_freq == 0:
                h = gamma + c
                h_pre = gamma_pre + c_pre
                residual = cp.abs(h - h_pre).mean() / cp.abs(h.mean())
                print("Iteration %d, Residual %.3e" % (i, residual))
                if residual < error_tolerance:
                    is_finished = True
                    print(
                        "Stop iterate at %d steps, residual %.3e smaller than tolerance %.3e"
                        % (i, residual, error_tolerance)
                    )
            i += 1
        return h, c
