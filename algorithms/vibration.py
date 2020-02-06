"""
This module calculates the natural frequencies of drilling tubulars.

Author: Kirtland McKenna
Copyright 2019 McKenna Engineering
"""
from copy import copy
import numpy as np


def ax1_mode1(length, mass_bha, unit_mass_pipe=29.019, area=3.3856e-3, modulus=2.0684e11):
    """
    First mode natural frequency of axial vibration, f_1 (Hz)

    :param length: drill pipe length (bit depth - bha length), L (m)
    :type length: float
    :param mass_bha: mass of the BHA, m (kg)
    :type mass_bha: float
    :param unit_mass_pipe: mass of the drill pipe, m (kg/m)
    :type unit_mass_pipe: float
    :param area: drill pipe cross-sectional area, A (m2)
    :type area: float
    :param modulus: Young's modulus, E (Pa)
    :type modulus: float
    :return: natural frequency, omega (Hz)
    :rtype: float
    """
    k = modulus * area / (4 * length)
    return np.sqrt(k / (mass_bha + unit_mass_pipe * length / 3))


def ax1(mode, length, mass_bha, impedance=1.364e5, speed=5133.166, lam=1.03, tolerance=1e-10):
    """
    Nth mode natural frequency of axial vibration, f_1 (Hz)

    :param mode: mode of natural frequency
    :type mode: int
    :param length: drill string length, L (m)
    :type length: float
    :param impedance: characteristic impedance of the drill pipe, C (kg/s)
    :type impedance: float
    :param mass_bha: BHA mass, m (kg)
    :type mass_bha: float
    :param speed: compression wave propagation speed, c (m/s)
    :type speed: float
    :param lam: H-factor
    :type lam: float
    :param tolerance: error tolerance
    :type tolerance: float
    :return: Natural frequency, f (rad/s)
    :rtype: float
    """
    freq, freq_prev, error = 0, 1, 1

    while error > tolerance:
        invtan = np.arctan(freq_prev * mass_bha / (impedance * lam))
        freq = speed / (2 * length) * ((2 * mode - 1) * np.pi - 2 * invtan) / 2
        error = square_error(freq, freq_prev)
        freq_prev = copy(freq)
    return freq


def ax2(mode, length, inertia_bha, impedance=296.392, speed=3192.348, lam=1.04, tolerance=1e-10):
    """
    Natural frequency of torsional vibration, f_2 (Hz).

    :param mode: mode of natural frequency
    :type mode: int
    :param length: drill string length, L (m)
    :type length: float
    :param inertia_bha: BHA Mass Rotational Inertia, J (kg.m2)
    :type inertia_bha: float
    :param impedance: characteristic impedance of the drill pipe, C (Nms)
    :type impedance: float
    :param speed: shear wave propagation speed, c (m/s)
    :type speed: float
    :param lam: H-factor
    :type lam: float
    :param tolerance: error tolerance
    :type tolerance: float
    :return: Natural frequency, f (rad/s)
    :rtype: float
    """
    freq, freq_prev, error = 0, 1, 1

    while error > tolerance:
        invtan = np.arctan(freq_prev * inertia_bha / (impedance * lam))
        freq = speed / (2 * length * lam) * ((2 * mode - 1) * np.pi - 2 * invtan)
        error = square_error(freq, freq_prev)
        freq_prev = copy(freq)
    return freq


def square_error(x1, x2):
    """
    Relative square error
    :param x1:
    :type x1: float
    :param x2:
    :type x2: float
    :return: relative square error
    :rtype: float
    """
    return np.sqrt((x1 - x2) ** 2)/np.sqrt(x1 * x2)
