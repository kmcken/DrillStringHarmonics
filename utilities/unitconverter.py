"""
This module provides unit conversion capabilities based on the Energistics Unit of Measure Standard v1.0.

For more information regarding the Energistics standard, please see
http://www.energistics.org/asset-data-management/unit-of-measure-standard

Author: Kirtland McKenna
Copyright 2019 McKenna Engineering
"""

from lxml import etree
import pkg_resources
try:
    import numpy as np
except:
    __numpyEnabled = False
else:
    __numpyEnabled = True


# Load Energistics symbols and factors
resource_package = __name__
xmlFile = pkg_resources.resource_string(resource_package, "/units.xml")
root = etree.fromstring(xmlFile)

tag = etree.QName('http://www.energistics.org/energyml/data/uomv1', 'unit')
__units = {}
for unitXml in root.iter(tag.text):
    unit = {}
    isBase = False
    for field in unitXml:
        t = etree.QName(field.tag)

        try:
            unit[t.localname] = float(eval(field.text.replace("PI", "np.pi")))
        except:
            unit[t.localname] = field.text

        if t.localname == "isBase":
            unit["baseUnit"] = unit["symbol"]
            unit["A"] = 0.0
            unit["B"] = 1.0
            unit["C"] = 1.0

    __units[unit["symbol"]] = unit


def isUnit(symbol):
    if symbol not in __units.keys():
        raise ValueError('Units: {0} is not a valid unit symbol.'.format(symbol))


def add_custom_unit(symbol, name, base_unit, a, b, c, d=0):
    """
    Adds a custom unit defined as:\n
    y=(a+b*value)/(c+d*value)\n
    where\n
    offset = a/c\n
    scale = b/c\n

    All current Units have d=0, so this can safely be ignored

    Set the force flag to True to force an override of existing symbol
    """
    # global units
    if symbol not in __units.keys():
        __units[symbol] = {'symbol': symbol, 'name': name, "baseUnit": base_unit, "A": a, "B": b, "C": c, "D": d}


def from_to(value, source_unit, target_unit):
    """
    Converts value(s) from a non-SI unit to the desired target_unit

    :param value: Value to convert
    :param source_unit: Source unit symbol
    :type source_unit: str
    :param target_unit: Target unit symbol
    :type target_unit: str
    :return: The value converted to target_unit
    """
    try:
        __units[target_unit]
    except KeyError:
        raise KeyError(str(target_unit) + ' is an incorrect unit symbol.')
    try:
        __units[source_unit]
    except KeyError:
        raise KeyError(source_unit + ' is an incorrect unit symbol.')

    if __units[target_unit]['baseUnit'] != __units[source_unit]['baseUnit']:
        raise KeyError('source_unit ' + str(source_unit) + ' and target_unit ' + str(target_unit) +
                       ' do not share the same base unit.')

    return from_si(to_si(value, source_unit), target_unit)


def from_si(value, target_unit):
    """
    Takes value(s) in SI, and converts it to a value in the desired TARGETUNIT

    :param value: The value to convert (can be a list)
    :type value: float or list
    :param target_unit: The relevant unit symbol as a string
    :type target_unit: str
    :return: The value converted to TARGETUNIT
    """
    try:
        __units[target_unit]
    except KeyError:
        raise KeyError(str(target_unit) + ' is an incorrect unit symbol.')

    global __numpyEnabled
    offset = __units[target_unit]["A"] * 1.0 / __units[target_unit]["C"]
    scale = __units[target_unit]["B"] * 1.0 / __units[target_unit]["C"]
    if __numpyEnabled:
        y = np.divide(value, scale) - np.divide(offset, scale)
    else:
        scaled_offset = offset / scale
        if hasattr(value, "__iter__"):

            y = [(v / scale)-scaled_offset for v in value]
        else:
            y = value/scale - scaled_offset

    return y


def to_si(value, source_unit):
    """
    Takes value(s) in SOURCEUNIT and converts it to a value in the SI unit for the relevant quantity

    :param value: The value to convert (can be a list)
    :type value: float or list
    :param source_unit: The relevant unit symbol as a string
    :type source_unit: str
    :return: The value converted to SI
    """
    try:
        __units[source_unit]
    except KeyError:
        raise KeyError(source_unit + ' is an incorrect unit symbol.')

    global __numpyEnabled
    offset = __units[source_unit]["A"] * 1.0 / __units[source_unit]["C"]
    scale = __units[source_unit]["B"] * 1.0 / __units[source_unit]["C"]
    if __numpyEnabled:
        y = np.multiply(value, scale) + offset
    else:
        if hasattr(value, "__iter__"):
            y = [(v * scale) + offset for v in value]
        else:
            y = value*scale + offset
    return y


def base_unit(unit):
    """
    What is the base unit

    :param unit: unit symbol
    :type unit: str
    :return: baseUnit
    :rtype: str
    """
    return __units[unit]['baseUnit']


def set_numpy_enabled(enabled):
    global __numpyEnabled
    __numpyEnabled = enabled
