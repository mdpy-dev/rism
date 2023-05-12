#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : matrix_parser.py
created time : 2023/05/12
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import h5py
from rism.core import FFTGrid, Matrix
from rism.environment import CUPY_FLOAT
from rism.error import SuffixError


class MatrixParser:
    def __init__(self, matrix_file_path: str) -> None:
        if not matrix_file_path.endswith(".mat"):
            raise SuffixError("The file should end with .mat suffix")
        self._file_path = matrix_file_path

    def parse(self):
        with h5py.File(self._file_path, "r") as f:
            # site_list
            site_list = [bytes.decode(i) for i in f["site_list"][()]]
            # coordinate_data
            coordinate_data = {}
            for key in f["grid"].keys():
                coordinate_data[key] = f["grid/" + key][()]
            # Create matrix
            grid = FFTGrid(**coordinate_data)
            matrix = Matrix(grid, site_list, CUPY_FLOAT)
            # Load data
            for i in range(matrix.num_sites):
                for j in range(i, matrix.num_sites):
                    matrix[i, j] = f["data/%d-%d" % (i, j)][()]
        return matrix
