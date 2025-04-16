from math import pi
import yaml
import pkgutil
import numpy as np

TwoPi = 2.0 * pi

Charge_SI = 1.6021766208e-19  # in Coulomb

H_ = 4.1356676969e-3  # Planck constant h in eV.ps
Hbar_ = H_ / TwoPi # hbar in eV.ps

Mu_B_ = 5.7883817982e-5  # Bohr magneton mu_B in eV/T, 1T = 1 V.s.m^-2 = 1e-8 V.ps.\AA^-2

Orbitals: dict = yaml.safe_load(pkgutil.get_data(__package__, 'orbitals.yml').decode('utf-8'))

Eta_4 = 1.0e-4
Eta_6 = 1.0e-6
Eta_8 = 1.0e-8

_Cart = [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0]]

Cart = np.array(_Cart, dtype=float)

S_ = np.array([
    [[ 0.0,  1.0],
     [ 1.0,  0.0]],
    [[ 0.0, -1.0j],
     [ 1.0j, 0.0]],
    [[ 1.0,  0.0],
     [ 0.0, -1.0]]
    ], dtype=complex)