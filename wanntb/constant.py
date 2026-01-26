from math import pi
import yaml
import pkgutil
import numpy as np

'''
Units used in the code:
energy: eV
time: ps 
length: \AA
1 T = 1e-8 V.ps.\AA^-2
1 mu_B = 5.7883817982e-5 eV.T^-1
1 h = 4.1356676969e-3 eV.ps 
'''


TwoPi = 2.0 * pi

Charge_SI = 1.6021766208e-19  # in Coulomb

H_ = 4.1356676969e-3  # Planck constant h in eV.ps
Hbar_ = H_ / TwoPi # hbar in eV.ps

Mu_B_ = 5.7883817982e-5  # Bohr magneton mu_B in eV/T, 1T = 1 V.s.m^-2 = 1e-8 V.ps.\AA^-2

Orbitals: dict = yaml.safe_load(pkgutil.get_data(__package__, 'orbitals.yml').decode('utf-8'))

Eta_4 = 1.0e-4
Eta_6 = 1.0e-6
Eta_8 = 1.0e-8

Cart = np.eye(3, dtype=np.float64)

S_ = np.array([
    [[ 0.0,  1.0],
     [ 1.0,  0.0]],
    [[ 0.0, -1.0j],
     [ 1.0j, 0.0]],
    [[ 1.0,  0.0],
     [ 0.0, -1.0]]
    ], dtype=complex)

Berry_Task = {
    'ahc':{
        'itask': 0,
        'columns' : ['sigma_x', 'sigma_y', 'sigma_z'],
        'units_fermi': 'e^2/h/\AA'
    },
    'shc':{
        'itask': 10,
        'columns' : ['sigma_ab^x', 'sigma_ab^y', 'sigma_ab^z'],
        'units_fermi': 'e^2/h/\AA(hbar/2e)'
    },
    'morb': {
        'itask': 20,
        'columns' : ['morb1_xyz', 'morb2_xyz', 'morb_xyz'],
        'units_fermi': 'mu_B/u.c.'
    },
}
