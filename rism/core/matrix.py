#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : matrix.py
created time : 2023/05/12
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import cupy as cp
import numpy as np
from rism.core.fft_grid import FFTGrid
from rism.error import ArrayShapeError
from rism.environment import *


class Matrix:
    def __init__(self, grid: FFTGrid, site_list: list, dtype=CUPY_FLOAT) -> None:
        self._grid = grid
        self._site_list = site_list
        self._dtype = dtype
        self._num_sites = len(self._site_list)
        self._num_data = int((self._num_sites + 1) * self._num_sites * 0.5)
        self._data = cp.zeros([self._num_data] + self._grid.shape, self._dtype)

    def _parse_site(self, site):
        if isinstance(site, str):
            return self._site_list.index(site)
        elif isinstance(site, int):
            if site >= self._num_sites:
                raise IndexError("The site index out of range")
            return site
        else:
            raise KeyError("Only int and str are supported argument")

    def _parse_site_pair(self, site1, site2):
        site1 = self._parse_site(site1)
        site2 = self._parse_site(site2)
        if site1 > site2:
            site1, site2 = site2, site1
        start_index = np.cumsum([self._num_sites - i for i in range(self._num_sites)])
        start_index = [0] + list(start_index)[:-1]
        index = start_index[site1] + site2 - site1
        return index

    def _check_val_shape(self, val: cp.ndarray):
        is_matched = True
        if len(val.shape) != self._grid.num_dimensions:
            is_matched = False
        else:
            for i, j in zip(val.shape, self._grid.shape):
                if i != j:
                    is_matched = False
                    break

        if not is_matched:
            raise ArrayShapeError(
                "item of matrix should have shape %s, while matrix with %s is provided"
                % (tuple(self._grid.shape), val.shape)
            )

    def __getitem__(self, key):
        site1, site2 = key
        index = self._parse_site_pair(site1, site2)
        return self._data[index]

    def __setitem__(self, key, val):
        self._check_val_shape(val)
        site1, site2 = key
        index = self._parse_site_pair(site1, site2)
        if isinstance(val, np.ndarray):
            val = cp.array(val)
        self._data[index] = val.astype(self._dtype)

    @property
    def grid(self):
        return self._grid

    @property
    def site_list(self):
        return self._site_list

    @property
    def num_sites(self):
        return self._num_sites
