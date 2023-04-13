__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__copyright__ = "(C)Copyright 2021-present, mdpy organization"
__license__ = "BSD-3"

from rism.unit import *


convert_factor = 2 ** (5 / 6)
FF_DICT = {
    "c": {
        "sigma": Quantity(1.992 * convert_factor, angstrom),
        "epsilon": Quantity(0.070, kilocalorie_permol),
        "val": Quantity(0.0, elementary_charge),
    },
    "o": {
        "sigma": Quantity(1.768 * convert_factor, angstrom),
        "epsilon": Quantity(0.1521, kilocalorie_permol),
        "val": Quantity(-2.0, elementary_charge),
    },
    "h": {
        "sigma": Quantity(0.225 * convert_factor, angstrom),
        "epsilon": Quantity(0.046, kilocalorie_permol),
        "val": Quantity(1.0, elementary_charge),
    },
    "li": {
        "sigma": Quantity(1.298 * convert_factor, angstrom),
        "epsilon": Quantity(0.00233, kilocalorie_permol),
        "val": Quantity(1.0, elementary_charge),
    },
    "na": {
        "sigma": Quantity(1.411 * convert_factor, angstrom),
        "epsilon": Quantity(0.047, kilocalorie_permol),
        "val": Quantity(1.0, elementary_charge),
    },
    "k": {
        "sigma": Quantity(1.764 * convert_factor, angstrom),
        "epsilon": Quantity(0.087, kilocalorie_permol),
        "val": Quantity(1.0, elementary_charge),
    },
    "ca": {
        "sigma": Quantity(1.367 * convert_factor, angstrom),
        "epsilon": Quantity(0.120, kilocalorie_permol),
        "val": Quantity(1.0, elementary_charge),
    },
    "cl": {
        "sigma": Quantity(2.270 * convert_factor, angstrom),
        "epsilon": Quantity(0.150, kilocalorie_permol),
        "val": Quantity(-1.0, elementary_charge),
    },
}
