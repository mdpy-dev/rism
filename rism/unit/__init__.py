__author__ = "Zhenyu Wei"
__maintainer__ = "Zhenyu Wei"
__copyright__ = "(C)Copyright 2021-present, mdpy organization"
__license__ = "BSD"


UNIT_PRECISION = 1e-6
QUANTITY_PRECISION = 1e-6

from rism.unit.base_dimension import BaseDimension
from rism.unit.unit import Unit
from rism.unit.quantity import Quantity

# BaseDimension
from rism.unit.unit_definition import (
    length,
    mass,
    time,
    temperature,
    charge,
    mol_dimension,
)
from rism.unit.unit_definition import force, energy, power, velocity, acceleration

# Unit
from rism.unit.unit_definition import (
    meter,
    decimeter,
    centimeter,
    millimeter,
    micrometer,
    nanometer,
    angstrom,
)
from rism.unit.unit_definition import kilogram, gram, amu, dalton
from rism.unit.unit_definition import day, hour, minute
from rism.unit.unit_definition import (
    second,
    millisecond,
    microsecond,
    nanosecond,
    picosecond,
    femtosecond,
)
from rism.unit.unit_definition import kelvin
from rism.unit.unit_definition import (
    coulomb,
    elementary_charge,
    ampere,
    volt,
    ohm,
    farad,
    siemens,
    hertz,
)
from rism.unit.unit_definition import mol, kilomol
from rism.unit.unit_definition import (
    joule,
    kilojoule,
    joule_permol,
    kilojoule_permol,
    calorie,
    kilocalorie,
    calorie_premol,
    kilocalorie_permol,
    ev,
    hartree,
)
from rism.unit.unit_definition import newton, kilonewton
from rism.unit.unit_definition import (
    kilojoule_permol_over_angstrom,
    kilojoule_permol_over_nanometer,
    kilocalorie_permol_over_angstrom,
    kilocalorie_permol_over_nanometer,
)
from rism.unit.unit_definition import watt, kilowatt

# Default Unit
default_length_unit = angstrom
default_mass_unit = dalton
default_time_unit = femtosecond
default_temperature_unit = kelvin
default_charge_unit = elementary_charge
default_mol_unit = mol

default_frequency_unit = 1 / default_time_unit
default_velocity_unit = default_length_unit / default_time_unit
default_accelerated_velocity_unit = default_velocity_unit / default_time_unit
default_energy_unit = (
    default_mass_unit * default_length_unit**2 / default_time_unit**2
)
default_power_unit = default_energy_unit / default_time_unit
default_force_unit = default_energy_unit / default_length_unit
default_current_unit = default_charge_unit / default_time_unit
default_voltage_unit = default_power_unit / default_current_unit
default_resistance_unit = default_voltage_unit / default_current_unit
default_capacitance_unit = default_charge_unit / default_voltage_unit
default_conductance_unit = 1 / default_resistance_unit
default_electric_intensity_unit = default_voltage_unit / default_length_unit


from rism.unit.quantity import Quantity

# Constant
KB = Quantity(1.38064852e-23, Unit(energy / temperature, 1))
NA = Quantity(6.0221e23, Unit(1 / mol_dimension, 1))
EPSILON0 = Quantity(
    8.85418e-12, second**2 * coulomb**2 / meter**3 / kilogram
).convert_to(
    default_time_unit**2
    * default_charge_unit**2
    / default_length_unit**3
    / default_mass_unit
)

# Utils
from rism.unit.utils import check_quantity, check_quantity_value

__all__ = [
    "Quantity",
    "default_length_unit",
    "default_mass_unit",
    "default_time_unit",
    "default_temperature_unit",
    "default_charge_unit",
    "default_mol_unit",
    "default_frequency_unit",
    "default_velocity_unit",
    "default_accelerated_velocity_unit",
    "default_energy_unit",
    "default_power_unit",
    "default_force_unit",
    "default_current_unit",
    "default_voltage_unit",
    "default_resistance_unit",
    "default_capacitance_unit",
    "default_conductance_unit",
    "default_electric_intensity_unit",
    "meter",
    "decimeter",
    "centimeter",
    "millimeter",
    "micrometer",
    "nanometer",
    "angstrom",
    "kilogram",
    "gram",
    "amu",
    "dalton",
    "day",
    "hour",
    "minute",
    "second",
    "millisecond",
    "microsecond",
    "nanosecond",
    "picosecond",
    "femtosecond",
    "kelvin",
    "coulomb",
    "elementary_charge",
    "ampere",
    "volt",
    "ohm",
    "farad",
    "siemens",
    "hertz",
    "mol",
    "kilomol",
    "joule",
    "kilojoule",
    "joule_permol",
    "kilojoule_permol",
    "calorie",
    "kilocalorie",
    "calorie_premol",
    "kilocalorie_permol",
    "ev",
    "hartree",
    "newton",
    "kilonewton",
    "kilojoule_permol_over_angstrom",
    "kilojoule_permol_over_nanometer",
    "kilocalorie_permol_over_angstrom",
    "kilocalorie_permol_over_nanometer",
    "watt",
    "kilowatt",
    "NA",
    "KB",
    "EPSILON0",
    "check_quantity",
    "check_quantity_value",
]
