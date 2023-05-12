#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : matrix_writer.py
created time : 2023/05/12
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


import h5py
from rism.core import Matrix
from rism.error import SuffixError


class MatrixWriter:
    def __init__(self, matrix_file_path: str) -> None:
        if not matrix_file_path.endswith(".mat"):
            raise SuffixError("The file should end with .mat suffix")
        self._file_path = matrix_file_path

    def write(self, matrix: Matrix):
        with h5py.File(self._file_path, "w") as f:
            self._write_info(f, matrix)
            self._write_grid(f, matrix)
            self._write_data(f, matrix)

    def _write_info(self, handle: h5py.File, matrix: Matrix):
        handle["site_list"] = matrix.site_list

    def _write_grid(self, handle: h5py.File, matrix: Matrix):
        handle.create_group("grid")
        coordinate_data = matrix.grid.coordinate_data
        for key, val in coordinate_data.items():
            handle["grid/" + key] = val

    def _write_data(self, handle: h5py.File, matrix: Matrix):
        handle.create_group("data")
        for i in range(matrix.num_sites):
            for j in range(i, matrix.num_sites):
                handle["data/%d-%d" % (i, j)] = matrix[i, j].get()
