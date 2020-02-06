"""
This module calculates the natural frequencies of lateral vibration drilling tubulars.

Author: Kirtland McKenna
Copyright 2019 McKenna Engineering
"""
from utilities import unitconverter as units
from copy import copy
import numpy as np
import drillstring


def nat_freq_drillpipe(mode, outer_diameter, inner_diameter, unit_weight, mud_weight, modulus, hole_diameter,
                       delta_measured_depth, inclination, dogleg):
    """
    Natural frequency of lateral vibration of the drill pipe.

    :param mode: mode number, n
    :type mode: int
    :param outer_diameter: outer diameter, D (m)
    :type outer_diameter: float
    :param inner_diameter: inner diameter, d (m)
    :type inner_diameter: float
    :param unit_weight: mass per unit length, m_bar (kg/m)
    :type unit_weight: float
    :param mud_weight: mud weight, (kg/m)
    :type mud_weight: float
    :param modulus: Young's Modulus, E (kg/(m.s2))
    :type modulus: float
    :param hole_diameter: hole diameter, H (m)
    :type hole_diameter: float
    :param delta_measured_depth: length of section between survey stations
    :type delta_measured_depth: float
    :param inclination: hole inclination, inc (rad)
    :type inclination: float
    :param dogleg: dogleg severity, dls (rad)
    :type dogleg: float
    :return: natural frequency, f (rad/s)
    :rtype: float
    """
    g = 9.80665

    if inclination <= np.radians(1):
        inclination = np.radians(1)

    I = np.pi / 64 * (outer_diameter ** 4 - inner_diameter ** 4)
    BF = (tags.read('DrillPipeDensity') - mud_weight) / tags.read('DrillPipeDensity')
    m_bar = unit_weight * BF + mud_weight * np.pi / 4 * (inner_diameter ** 2)

    L = length_unsupported(hole_diameter, outer_diameter, inner_diameter, modulus, m_bar, inclination)
    Lp = pass_through_length(hole_diameter, outer_diameter, delta_measured_depth, dogleg)

    if Lp < L:
        L = Lp

    return np.sqrt(((mode + 0.5) * np.pi)**4 * modulus * I / (unit_weight * BF * L**4)
                   + g * np.sin(inclination) / (hole_diameter - outer_diameter) / 2)


def natural_frequency_bha(bha_sections, weight_on_bit, mode):
    """
    Natural frequency of lateral vibration in the BHA

    :param bha_sections: BHA sections including inclination
    :type bha_sections: list
    :param weight_on_bit: actual weight on bit
    :type weight_on_bit: float
    :param mode: vibration shape mode, n
    :type mode: int
    :return: natural frequency, omega (rad/s)
    :rtype: float
    """
    k = stiffness_bha(bha_sections, weight_on_bit, mode)
    mass = bha_sections[0]['Mass'] + bha_sections[0]['FluidMass']
    return np.sqrt(k / mass)


def stiffness_bha(bha_sections, weight_on_bit, mode):
    """
    Lateral stiffness of the BHA

    :param bha_sections: BHA sections including inclination
    :type bha_sections: list
    :param weight_on_bit: actual weight on bit
    :type weight_on_bit: float
    :param mode: vibration shape mode, n
    :type mode: int
    :return: BHA stiffness, k
    :rtype: float
    """
    g = 9.80665
    weight, tension, weight_lateral = list(), list(), list()
    for section in bha_sections:
        weight.append((section['Mass']) * g * np.cos(section['Inclination']))
        weight_lateral.append((section['Mass'] + section['FluidMass']) * g * np.sin(section['Inclination']))

    tension.append(-1 * weight_on_bit)
    for i in range(1, len(weight)):
        tension.append(tension[i - 1] + weight[i - 1])

    modulus = tags.read('DrillPipeYoungsModulus')
    alpha = (mode) ** 2 * np.pi ** 2

    k1, k2, k3 = 0, 0, 0
    for i in range(0, len(bha_sections)):
        k1 += alpha ** 2 * bha_sections[i]['Clearance'] ** 2 * modulus * bha_sections[i]['BendingMoment'] / \
              bha_sections[i]['Length'] ** 3
        k2 += alpha * bha_sections[i]['Clearance'] ** 2 * tension[i] / bha_sections[i]['Length']
        k3 += weight_lateral[i] * bha_sections[i]['Clearance'] ** 2 / np.sum(np.array(weight_lateral)) * bha_sections[i]['Length']
    return (k1 + k2) / k3


def length_supported(hole_diameter, outer_diameter, inner_diameter, modulus, unit_weight, inclination):
    """
    The maximum packed length before sag contacts wellbore between packers

    :param hole_diameter: hole diameter (m)
    :type hole_diameter: float
    :param outer_diameter: pipe outer diameter, OD (m)
    :type outer_diameter: float
    :param inner_diameter: pipe inner diameter, ID (m)
    :type inner_diameter: float
    :param modulus: Young's modulus, E (Pa)
    :type modulus: float
    :param unit_weight: weight per unit length, w (kg/m)
    :type unit_weight: float
    :param inclination: hole inclination, inc (rad)
    :type inclination: float
    :return: maximum supported packed length (m)
    :rtype: float
    """

    if inclination == 0:
        return None
    delta = (hole_diameter - units.to_si(0.25, 'in') - outer_diameter) / 2
    I = drillstring.bending_inertia(outer_diameter, inner_diameter)
    return np.power(384 * modulus * I * delta / (5 * unit_weight * np.sin(inclination)), 0.25)


def length_unsupported(hole_diameter, outer_diameter, inner_diameter, modulus, unit_weight, inclination):
    """
    The maximum unsupported length before sag contacts wellbore between packers

    :param hole_diameter: hole diameter (m)
    :type hole_diameter: float
    :param outer_diameter: pipe outer diameter, OD (m)
    :type outer_diameter: float
    :param inner_diameter: pipe inner diameter, ID (m)
    :type inner_diameter: float
    :param modulus: Young's modulus, E (Pa)
    :type modulus: float
    :param unit_weight: weight per unit length, w (kg/m)
    :type unit_weight: float
    :param inclination: hole inclination, inc (rad)
    :type inclination: float
    :return: maximum supported packed length (m)
    :rtype: float
    """

    if inclination == 0:
        return None
    delta = (hole_diameter - units.to_si(0.25, 'in') - outer_diameter) / 2  # Minus the stabilizer clearance
    I = drillstring.bending_inertia(outer_diameter, inner_diameter)
    return np.power(8 * modulus * I * delta / (unit_weight * np.sin(inclination)), 0.25)


def section_details(bha_dict=None):
    """
    Finds all of the possible BHA sections between stabilizers for the lateral vibration calculation

    :param bha_dict: BHA details
    :type bha_dict: dict
    :return: {length, moment of inertia, pipe mass, radial clearance}
    :rtype: list
    """
    if bha_dict is None:
        bha_dict = tags.read('BHADetails')

    sections = list()
    short_list = list()     # {'length': None, 'inertia': None, 'mass': None, 'clearance': None}
    for element in reversed(bha_dict):
        if drillstring.isStab(element) is True:
            stab_element = element
            stab_element['Length'] = element['Length'] / 2
            short_list.append(stab_element)
            sections.append(copy(short_list))
            short_list.clear()
            short_list.append(stab_element)
        else:
            if element['BHA'] is not True:
                sections.append(copy(short_list))
                break
            short_list.append(element)

    def ave_unit(length, value):
        values = 0
        for i in range(0, len(length)):
            values += value[i] * length[i]
        return values / np.sum(length)

    sections_details = list()
    for section in sections:
        length, mass, inertia, fluid_mass, rot_inertia, od, idp = list(), list(), list(), list(), list(), list(), list()
        for element in section:
            length.append(element['Length'])
            mass.append(element['UnitWeight'] * element['Length'])
            inertia.append(drillstring.bending_inertia(element['OD'], element['NominalID']))
            fluid_mass.append(np.pi / 4 * element['NominalID'] ** 2 * element['Length'] * tags.read('MudWeight'))
            rot_inertia.append(drillstring.rotational_inertia(element['OD'], element['NominalID'],
                                                              tags.read('DrillPipeDensity'), element['Length']))
            od.append(element['Length'] * element['OD'])
            idp.append(element['Length'] * element['NominalID'])

        section_od, section_id = drillstring.effective_pipe_geometry(np.sum(rot_inertia), np.sum(mass),
                                                                      tags.read('DrillPipeDensity'), np.sum(length))
        sections_details.append({'Length': np.sum(length), 'Mass': np.sum(mass), 'FluidMass': np.sum(fluid_mass),
                                 'OD': np.sum(od) / np.sum(length), 'ID': np.sum(idp) / np.sum(length),
                                 'BendingMoment': ave_unit(length, inertia),
                                 'Clearance': (tags.read('BitDiameter') - section_od) / 2})

    tags.write('BHASections', sections_details)


def sections_inclination(depth, sections=None):
    """
        Finds all of the possible BHA sections between stabilizers for the lateral vibration calculation

        :param depth: measured depth of bit for vibration calculations, md (m)
        :type depth: float
        :param sections: BHA sections details
        :type sections: list
        :return: {length, moment of inertia, pipe mass, radial clearance}
        :rtype: list
        """
    if sections is None:
        sections = tags.read('BHASections')

    survey = tags.read('Survey')
    md, inc = survey[0], survey[1]

    i = 0
    try:
        while md[i] <= depth:
            i += 1
    except IndexError:
        i = len(md)
    n = i - 1

    def inc_average(depth, inclination, total_depth):
        delta_md, avg_inc = list(), list()
        for i in range(0, len(depth) - 1):
            delta_md.append(depth[i + 1] - depth[i])
            avg_inc.append(0.5 * inclination[i + 1] + 0.5 * inclination[i])
        delta_md.append(total_depth - depth[-1])
        avg_inc.append(inclination[-1])
        return delta_md, avg_inc

    md_delta, inc_avg = inc_average(md[:n], inc[:n], depth)
    well_inc = 0
    for i in range(0, len(md_delta)):
        well_inc += md_delta[i] * inc_avg[i] / np.sum(np.array(md_delta))

    for section in sections:
        length, inclination = list(), list()
        while np.sum(np.array(length)) <= section['Length']:
            length.append(md_delta.pop(-1)), inclination.append(inc_avg.pop(-1))
        section_inc = 0
        for i in range(0, len(length)):
            section_inc += length[i] * inclination[i] / np.sum(np.array(length))
        section['Inclination'] = section_inc
        md_delta.append(length.pop(-1)), inc_avg.append(inclination.pop(-1))
    return sections


def update_sections(sections, delta_measured_depth, inclination, dogleg):
    TotL = 0
    for section in sections:
        TotL += section['Length']

    i = len(dogleg) - 1
    srv_len = 0
    while srv_len < TotL:
        srv_len += delta_measured_depth[i]
        i -= 1

    def unsupported(modulus, moment, clearance, unitmass, inc):
        return np.power(8 * modulus * moment * clearance / (unitmass * np.sin(inc)), 0.25)

    L = unsupported(tags.read('DrillPipeYoungsModulus'), sections[-1]['BendingMoment'], sections[-1]['Clearance'],
                    sections[-1]['Mass'] / sections[-1]['Length'], sections[-1]['Inclination'])
    L1 = pass_through_length(tags.read('BitDiameter'), sections[-1]['OD'], delta_measured_depth[i], dogleg[i])
    if L1 < L:
        L = L1

    sections[-1]['Mass'] = sections[-1]['Mass'] / sections[-1]['Length'] * L
    sections[-1]['FluidMass'] = sections[-1]['FluidMass'] / sections[-1]['Length'] * L
    sections[-1]['Length'] = L


def pass_through_length(hole_diameter, pipe_diameter, arc_length, dls):
    """
    The straight length that straight pipe can pass through a curved hole

    :param hole_diameter: hole diameter, H (m)
    :type hole_diameter: float
    :param pipe_diameter: pipe outer diameter, OD (m)
    :type pipe_diameter: float
    :param arc_length: length from last survey station (m)
    :type arc_length: float
    :param dls: dogleg (rad)
    :type dls: float
    :return: straight length, l (m)
    :rtype: float
    """

    if dls == 0:
        return 1000

    r = arc_length / dls
    return np.sqrt(r ** 2 - (r - (hole_diameter - pipe_diameter)) ** 2)
