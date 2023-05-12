#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : fft_grid.py
created time : 2023/04/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import numpy as np
import cupy as cp
from rism.environment import CUPY_FLOAT


class FFTGrid:
    def __init__(self, **coordinate_data) -> None:
        # Input
        # Set grid information and coordinate
        self._coordinate_data = coordinate_data
        self._coordinate_label = list(coordinate_data.keys())
        grid, d_grid, l_grid = self._meshing(coordinate_data)
        self._shape = list(grid[0].shape)
        for index, key in enumerate(self._coordinate_label):
            setattr(self, key, grid[index])
            setattr(self, "d" + key, d_grid[index])
            setattr(self, "l" + key, l_grid[index])
        self._num_dimensions = len(self._coordinate_label)

    def _meshing(self, coordinate_data):
        grid = []
        d_grid = []
        l_grid = []
        for key, value in coordinate_data.items():
            x, dx = cp.linspace(
                start=value[0],
                stop=value[1],
                num=value[2],
                endpoint=False,
                retstep=True,
                dtype=CUPY_FLOAT,
            )
            grid.append(x)
            d_grid.append(dx)
            l_grid.append(dx * value[2])
        return cp.meshgrid(*grid, indexing="ij"), d_grid, l_grid

    def zeros_field(self, dtype=CUPY_FLOAT):
        return cp.zeros(self._shape, dtype)

    def ones_field(self, dtype=CUPY_FLOAT):
        return cp.ones(self._shape, dtype)

    @property
    def coordinate_label(self) -> list[str]:
        return self._coordinate_label

    @property
    def coordinate_data(self) -> dict:
        return self._coordinate_data

    @property
    def num_dimensions(self) -> int:
        return self._num_dimensions

    @property
    def shape(self) -> list[int]:
        return self._shape

    @property
    def num_points(self) -> int:
        return int(np.prod(self._shape))
