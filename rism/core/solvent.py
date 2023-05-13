#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : solvent.py
created time : 2023/04/20
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


from collections import Counter
from rism.core import Particle
from rism.error import DuplicatedNameError, SolventIncompleteError
from rism.unit import *


class Solvent:
    def __init__(self, particle_list: list[Particle]) -> None:
        self._particle_list = particle_list
        self._bond_dict = {}
        self._name_list = [i.name for i in self._particle_list]
        self._num_particles = len(self._name_list)
        if len(set(self._name_list)) != self._name_list:
            count = dict(Counter(self._name_list))
            for key, val in count.items():
                if val != 1:
                    raise DuplicatedNameError("Duplicate particle name %s" % key)

    def _parse_index(self, index):
        if isinstance(index, str):
            return self._name_list.index(index)
        elif isinstance(index, Particle):
            return self._name_list.index(index.name)
        elif isinstance(index, int):
            if index >= self._num_particles:
                raise IndexError("The site index out of range")
            return index
        else:
            raise KeyError("Only int, str, Particle are supported argument")

    def _get_bond_name(self, index1, index2):
        index1 = self._parse_index(index1)
        index2 = self._parse_index(index2)
        if index1 > index2:
            index1, index2 = index2, index1
        bond_name = "%s-%s" % (self._name_list[index1], self._name_list[index2])
        return bond_name

    def add_bond(self, index1, index2, bond_length):
        bond_name = self._get_bond_name(index1, index2)
        if not bond_name in self._bond_dict.keys():
            self._bond_dict[bond_name] = float(
                check_quantity_value(bond_length, default_length_unit)
            )

    def get_particle(self, index):
        index = self._parse_index(index)
        return self._particle_list[index]

    def get_bond_length(self, index1, index2):
        bond_name = self._get_bond_name(index1, index2)
        if bond_name in self._bond_dict.keys():
            return self._bond_dict[bond_name]
        else:
            raise KeyError("Bond %s does not exists in the topology" % bond_name)

    def check_completeness(self):
        bond_list = list(self._bond_dict.keys())
        is_complete = True
        for i in range(self._num_particles):
            for j in range(i, self._num_particles):
                bond_name = self._get_bond_name(i, j)
                if not bond_name in bond_list:
                    is_complete = False
                    break
        if not is_complete:
            raise SolventIncompleteError("Solvent missing %s bond" % bond_name)

    def asdict(self):
        res = {}
        res["particle"] = [i.asdict() for i in self._particle_list]
        res["bond"] = self._bond_dict
        return res

    @property
    def num_particles(self) -> int:
        return self._num_particles

    @property
    def particle_list(self) -> list[Particle]:
        return self._particle_list

    @property
    def name_list(self) -> list[str]:
        return self._name_list

    @property
    def num_bonds(self) -> int:
        return len(self.bond_list)

    @property
    def bond_list(self) -> list:
        return list(self._bond_dict.keys())
