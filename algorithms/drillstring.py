"""
This module calculates the pipe properties of drilling tubulars.

Author: Kirtland McKenna
Copyright 2019 McKenna Engineering
"""
from utilities import unitconverter as units
from VibrationModel import lateral
import config
from copy import copy
import numpy as np


def resonance_spring(mode, spring_constant, mass):
    """
    Simple resonance of a spring-mass system.

    :param mode: mode number, n
    :type mode: int
    :param spring_constant: spring constant, k
    :type spring_constant: float
    :param mass: mass, m
    :type mass: float
    :return: natural frequency for mode, n
    :rtype: float
    """
    return (2 * mode - 1) / (2 * np.pi) * np.sqrt(spring_constant / mass)


def stabilizer_locations(bha_dict):
    """
    Returns the distance from bit to the center of each stabilizer (if present).

    :param bha_dict: BHA dictionary
    :type bha_dict: dict
    :return: stabilizer locations (m)
    :rtype: list
    """

    pos = list([0])
    length = 0
    for element in reversed(bha_dict):
        if config.isStab(element) is True:
            pos.append(length + 0.5 * element['Length'])
        length += element['Length']
    return pos


def points_of_contact(stab_locations, length, hole_diameter, outer_diameter, inner_diameter, youngs_modulus,
                      unit_weight, inclination):
    """
    Returns the points of contact in the BHA


    :param stab_locations: stabilizer locations
    :type stab_locations: list
    :param length: section length
    :type length: float
    :param hole_diameter: hole diameter (m)
    :type hole_diameter: float
    :param outer_diameter: pipe outer diameter, OD (m)
    :type outer_diameter: float
    :param inner_diameter: pipe inner diameter, ID (m)
    :type inner_diameter: float
    :param youngs_modulus: Young's modulus, E (Pa)
    :type youngs_modulus: float
    :param unit_weight: weight per unit length, w (kg/m)
    :type unit_weight: float
    :param inclination: hole inclination, inc (rad)
    :type inclination: float
    :return: points of contact locations (m)
    :rtype: list
    """

    poc = copy(stab_locations)
    unsupported = lateral.length_unsupported(hole_diameter, outer_diameter, inner_diameter, youngs_modulus,
                                             unit_weight, inclination)
    end_pt = stab_locations[-1]

    if unsupported is None or unsupported > length - end_pt:
        poc.append(length)
    else:
        poc.append(end_pt + unsupported)
    return poc


def area(outer_diameter, inner_diameter):
    """
    Cross sectional area, A

    :param outer_diameter: outer diameter, OD (m)
    :type outer_diameter: float
    :param inner_diameter: inner diameter, ID (m)
    :type inner_diameter: float
    :return: cross-sectional area, A (m2)
    :rtype: float
    """
    return np.pi / 4 * (outer_diameter ** 2 - inner_diameter ** 2)


def rotational_inertia(outer_diameter, inner_diameter, material_density, length):
    """
    Rotational inertia, J
    
    :param outer_diameter: outer diameter, OD (m)
    :type outer_diameter: float
    :param inner_diameter: inner diameter, ID (m)
    :type inner_diameter: float
    :param material_density: density, rho (kg/m3)
    :type material_density: float
    :param length: total length of element, L (m)
    :type length: float
    :return: rotational inertia, J (kg.m2)
    :rtype: float
    """
    I = polar_inertia(outer_diameter, inner_diameter)
    return material_density * length * I


def string_impedance(pipe_body_impedance, halsey_factor):
    """
    Calculates the drill string impedance using the Halsey method to account for the tool joints.

    :param pipe_body_impedance: drill pipe body impedance, C_p (kg.m2/s)
    :type pipe_body_impedance: float
    :param halsey_factor: Halsey wave number correction factor, kappa
    :type halsey_factor: float
    :return: drill string impedance, C_dp (kg.m2/s)
    :rtype: float
    """
    return pipe_body_impedance * halsey_factor


def impedance_speed(modulus, area, speed):
    """
    Characteristic impedance using wave propagation speed

    :param modulus: Young's or shear modulus, E/G (Pa)
    :type modulus: float
    :param area: area or moment of inertiat, A/I (m2/m4)
    :type area: float
    :param speed: wave propagation speed, c (m/s)
    :type speed: float
    :return: characteristic imperdance, C (kg.m2/s or kg/s)
    :rtype: float
    """

    return modulus * area / speed


def impedance(material_density, material_shear_modulus, inertia_polar):
    """
    Characteristic Impedance

    :param material_density: material density (kg/m3)
    :type material_density: float
    :param material_shear_modulus: shear modulus (Pa)
    :type material_shear_modulus: float
    :param inertia_polar: cross-sectional polar moment of inertia (m4)
    :type inertia_polar: float
    :return: characteristic impedance (Nms)
    :rtype: float
    """
    return inertia_polar * np.sqrt(material_density * material_shear_modulus)


def impedance_ratio(impedance_tooljoint, impedance_pipebody):
    """
    Drill pipe/tool joint characteristic impedance ratio.

    :param impedance_tooljoint: tool joint impedance (Nms)
    :type impedance_tooljoint: float
    :param impedance_pipebody: nominal joint impedance (Nms)
    :type impedance_pipebody: float
    :return: drill pipe impedance ratio (Euc)
    :rtype: float
    """
    return 0.5 * (impedance_tooljoint / impedance_pipebody + impedance_pipebody / impedance_tooljoint)


def effective_spring_const(string_length, relative_length, modulus, inertia_drillpipe, inertia_tooljoint):
    """
    Effective drill string torsional compliance with the tool joint effect.

    :param string_length: total length of the drill string, L (m)
    :type string_length: float
    :param relative_length: tool joint relative length
    :type relative_length: float
    :param modulus: Young's or shear modulus, E/G (Pa)
    :type modulus: float
    :param inertia_drillpipe: drill pipe inertia, Idp (m4)
    :type inertia_drillpipe: float
    :param inertia_tooljoint: tool joint inertia, Itj (m4)
    :type inertia_tooljoint: float
    :return: effective spring constant of the drill string, k_dp (kg.m2/s2)
    :rtype: float
    """

    compliance = (1 - relative_length) * string_length / (modulus * inertia_drillpipe) + \
                 relative_length * string_length / (modulus * inertia_tooljoint)

    return 1 / compliance


def linear_average(length_tooljoint, length_pipebody, value_tooljoint, value_pipebody):
    """
    Drill pipe characteristic impedance with the tool joints.

    :param length_tooljoint: tool joint length (m)
    :type length_tooljoint: float
    :param length_pipebody: nominal joint length (m)
    :type length_pipebody: float
    :param value_tooljoint: tool joint impedance
    :type value_tooljoint: float
    :param value_pipebody: nominal joint impedance
    :type value_pipebody: float
    :return: average drill pipe value
    :rtype: float
    """
    A = value_tooljoint * length_tooljoint / (length_tooljoint + length_pipebody) / 3
    B = value_pipebody * length_pipebody / (length_tooljoint + length_pipebody)
    return A + B


def polar_inertia(outer_diameter, inner_diameter):
    """
    Polar moment of inertia for pipe.

    :param outer_diameter: outer diameter, OD (m)
    :type outer_diameter: float
    :param inner_diameter: inner diameter, ID (m)
    :type inner_diameter: float
    :return: Polar moment of inertia, I<polar> (m4)
    :rtype: float
    """
    return np.pi / 32 * (outer_diameter ** 4 - inner_diameter ** 4)


def bending_inertia(outer_diameter, inner_diameter):
    """
    Bending moment of inertia for pipe.

    :param outer_diameter: outer diameter, OD (m)
    :type outer_diameter: float
    :param inner_diameter: inner diameter, ID (m)
    :type inner_diameter: float
    :return: moment of inertia, I (m4)
    :rtype: float
    """
    return np.pi / 64 * (outer_diameter ** 4 - inner_diameter ** 4)


def unit_volume(outer_diameter, inner_diameter):
    """
    Unit volume (V/L) of a pipe.

    :param outer_diameter: outer diameter, OD (m)
    :type outer_diameter: float
    :param inner_diameter: inner diameter, ID (m)
    :type inner_diameter: float
    :return: unit volume, V/L (m2)
    :rtype: float
    """
    return np.pi / 4 * (outer_diameter ** 2 - inner_diameter ** 2)


def unit_weight(outer_diameter, inner_diameter, material_density):
    """
    Unit weight (M/L) of a pipe.

    :param outer_diameter: outer diameter, OD (m)
    :type outer_diameter: float
    :param inner_diameter: inner diameter, ID (m)
    :type inner_diameter: float
    :param material_density: material density, rho (kg/m3)
    :type material_density: float
    :return: unit volume, M/L (kg/m)
    :rtype: float
    """
    return material_density * unit_volume(outer_diameter, inner_diameter)


def wave_speed(modulus, material_density):
    """
    Wave propagation speed.

    :param modulus: shear modulus of elasticity, E or G (Pa)
    :type modulus: float
    :param material_density: material density, rho (kg/m3)
    :type material_density: float
    :return: wave propagation speed, c (m/s)
    :rtype: float
    """
    return np.sqrt(modulus / material_density)


def effective_pipe_geometry(total_rotational_inertia, mass, material_density, length):
    """
    Effective pipe geometry. Used for lumped BHA

    :param total_rotational_inertia: rotational inertia, J (kg.m2)
    :type total_rotational_inertia: float
    :param mass: lumped mass, m (kg)
    :type mass: float
    :param material_density: average density, rho (kg/m3)
    :type material_density: float
    :param length: length, L (m)
    :type length: float
    :return: outer diameter, OD (m), inner diameter, ID (m)
    :rtype: list
    """

    A = total_rotational_inertia / mass
    B = 1 / (2 * np.pi) * mass / (material_density * length)

    try:
        d = 2 * np.sqrt(A - B)
        D = 2 * np.sqrt(A + B)
    except RuntimeWarning:
        raise ValueError('Effective Pipe Geometry: imaginary number, check inputs')
    return D, d


def lumped_rotational_inertia(pipe_list):
    """
    Lumped rotational inertia in the pipe list (BHA)

    :param pipe_list: List of pipes of interest from BHA
    :type pipe_list: list
    :return: lumped rotational inertia, J (kg.m2)
    :rtype: float
    """

    inertia = 0
    for pipe in pipe_list:
        inertia += rotational_inertia(pipe['OD'], pipe['NominalID'], density(pipe['Material']), pipe['Length'])
    return inertia


def lumped_length(pipe_list):
    """
    Lumped length in the pipe list (BHA)

    :param pipe_list: List of pipes of interest from BHA
    :type pipe_list: list
    :return: lumped length, L (m)
    :rtype: float
    """

    length = 0
    for pipe in pipe_list:
        length += pipe['Length']
    return length


def lumped_mass(pipe_list):
    """
    Lumped mass in the pipe list (BHA)

    :param pipe_list: List of pipes of interest from BHA
    :type pipe_list: list
    :return: lumped mass, m (kg)
    :rtype: float
    """

    mass = 0
    for pipe in pipe_list:
        if pipe['UnitWeight'] == 0:
            weight = unit_weight(pipe['OD'], pipe['NominalID'], density(pipe['Material']))
        else:
            weight = pipe['UnitWeight']

        mass += weight * pipe['Length']
    return mass


def isStab(bha_element):
    """
    Checks if BHA element is a stabilizer from text recognition.

    :param bha_element: BHA element details
    :type bha_element: dict
    :return: bool
    :rtype: bool
    """
    nonmag = configxml.setting_option('Stabilizer Options').strip().split(',')

    for name in nonmag:
        if name in bha_element['Description'] or name in bha_element['Type']:
            return True
    return False


def avg_surveys(survey, depth_of_interest=None):
    """
    The average values between survey points

    :param survey: survey file [[md], [inc], [azi]]
    :type survey: list
    :param depth_of_interest: depth of interest (return up to this depth)
    :type depth_of_interest: float
    :return: avg survey values [[delta MD], [avg_inc], [avg_azi], [dls]]
    :rtype: list
    """
    if isinstance(depth_of_interest, float) is False:
        depth_of_interest = survey[0][-1]

    delta_md, avg_inc, avg_azi, dls = list(), list(), list(), list()
    i = 1
    while survey[0][i] <= depth_of_interest:
        delta_md.append(np.abs(survey[0][i] - survey[0][i - 1]))
        dls.append(dogleg_severity([survey[1][i - 1], survey[1][i]], [survey[2][i - 1], survey[2][i]]))
        avg_inc.append(np.average(np.array([survey[1][i], survey[1][i - 1]])))
        avg_azi.append(np.average(np.array([survey[2][i], survey[2][i - 1]])))
        i += 1
        if i == len(survey[0]):
            break

    return delta_md, avg_inc, avg_azi, dls


def dogleg_severity(inc, azi):
    """
    Angle of curvature between survey points

    :param inc: inclination, [inc_prev, inc_current] (rad)
    :type inc: list
    :param azi: azimuth, [azi_prev, azi_current] (rad)
    :type azi: list
    :return: dogleg angle
    :rtype: float
    """

    return np.arccos(np.cos(inc[1] - inc[0]) - np.sin(inc[0]) * np.sin(inc[1]) * (1 - np.cos(azi[1] - azi[0])))

