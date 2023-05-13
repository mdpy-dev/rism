#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : element.py
created time : 2023/05/13
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""


from rism.core import Particle, Solvent
from rism.unit import *

# Particle
convert_factor = 2 ** (5 / 6)


def carbon(name="c"):
    return Particle(
        name=name,
        sigma=Quantity(1.992 * convert_factor, angstrom),
        epsilon=Quantity(0.070, kilocalorie_permol),
        q=Quantity(0.0, elementary_charge),
    )


def oxygen(name="o"):
    return Particle(
        name=name,
        sigma=Quantity(1.768 * convert_factor, angstrom),
        epsilon=Quantity(0.1521, kilocalorie_permol),
        q=Quantity(0.4238 * -2, elementary_charge),
    )


def hydrogen(name="h"):
    return Particle(
        name=name,
        sigma=Quantity(0.225 * convert_factor, angstrom),
        epsilon=Quantity(0.046, kilocalorie_permol),
        q=Quantity(0.4238, elementary_charge),
    )


def lithium(name="li"):
    return Particle(
        name=name,
        sigma=Quantity(1.298 * convert_factor, angstrom),
        epsilon=Quantity(0.00233, kilocalorie_permol),
        q=Quantity(1.0, elementary_charge),
    )


def sodium(name="na"):
    return Particle(
        name=name,
        sigma=Quantity(1.411 * convert_factor, angstrom),
        epsilon=Quantity(0.047, kilocalorie_permol),
        q=Quantity(1.0, elementary_charge),
    )


def potassium(name="k"):
    return Particle(
        name=name,
        sigma=Quantity(1.764 * convert_factor, angstrom),
        epsilon=Quantity(0.087, kilocalorie_permol),
        q=Quantity(1.0, elementary_charge),
    )


def calcium(name="ca"):
    return Particle(
        name=name,
        sigma=Quantity(1.367 * convert_factor, angstrom),
        epsilon=Quantity(0.120, kilocalorie_permol),
        q=Quantity(2.0, elementary_charge),
    )


def chlorine(name="cl"):
    return Particle(
        name=name,
        sigma=Quantity(2.270 * convert_factor, angstrom),
        epsilon=Quantity(0.150, kilocalorie_permol),
        q=Quantity(-1.0, elementary_charge),
    )


# Solvent


def tip3p():
    solvent = Solvent([oxygen("o"), hydrogen("h1"), hydrogen("h2")])
    solvent.add_bond("o", "h1", 0.9572)
    solvent.add_bond("o", "h2", 0.9572)
    solvent.add_bond("h1", "h2", 1.5139)
    return solvent
