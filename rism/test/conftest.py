#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : conftest.py
created time : 2021/09/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest

test_order = [
    "environment",
    "base_dimension",
    "unit",
    "unit_definition",
    "quantity",
    "check_quantity",
    "geometry",
    "pbc",
    "particle_select",
    "particle",
    "topology",
    "tile_list",
    "state",
    "trajectory",
    "ensemble",
    "grid",
    "charmm_toppar_parser",
    "psf_parser",
    "pdb_parser",
    "pdb_writer",
    "dcd_parser",
    "xyz_parser",
    "hdf5_parser",
    "hdf5_writer",
    "grid_parser",
    "grid_writer",
    "constraint",
    "electrostatic_cutoff_constraint",
    "charmm_vdw_constraint",
    "charmm_bond_constraint",
    "charmm_angle_constraint",
    "charmm_dihedral_constraint",
    "charmm_improper_constraint",
    "recipe",
    "charmm_recipe",
    "minimizer",
    "steepest_descent_minimizer",
    "conjugate_gradient_minimizer",
    "integrator",
    "verlet_integrator",
    "langevin_integrator",
    "analyser_result",
    "device",
]


def pytest_collection_modifyitems(items):
    current_index = 0
    for test in test_order:
        indexes = []
        for id, item in enumerate(items):
            if "test_" + test + ".py" in item.nodeid:
                indexes.append(id)
        for id, index in enumerate(indexes):
            items[current_index + id], items[index] = (
                items[index],
                items[current_index + id],
            )
        current_index += len(indexes)
