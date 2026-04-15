import pkgutil
from math import pi

import numpy as np
import yaml

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

V2 = np.sqrt(2.0)

Charge_SI = 1.6021766208e-19  # in Coulomb

H_ = 4.1356676969e-3  # Planck constant h in eV.ps
Hbar_ = H_ / TwoPi # hbar in eV.ps

Mu_B_ = 5.7883817982e-5  # Bohr magneton mu_B in eV/T, 1T = 1 V.s.m^-2 = 1e-8 V.ps.\AA^-2

Orbitals: dict = yaml.safe_load(pkgutil.get_data(__package__, 'orbitals.yml').decode('utf-8'))

EPS2 = 0.01
EPS3 = 1.0e-3
EPS4 = 1.0e-4
EPS5 = 1.0e-5
EPS6 = 1.0e-6
EPS7 = 1.0e-7
EPS8 = 1.0e-8

Cart = np.eye(3, dtype=np.float64)

#Pauli metrics
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


# Default tolerances for specific purposes
DEFAULT_SYMM_TOLERANCE = EPS5  # For spglib symmetry detection
DEFAULT_HAM_TOLERANCE = 0.05   # For Hamiltonian element mismatch warnings
DEFAULT_DEGENERATE_TOLERANCE = EPS6  # For band degeneracy detection
DEFAULT_MAGNETIC_TOLERANCE = EPS3  # For magnetic symmetry detection
DEFAULT_POSITION_TOLERANCE = EPS3  # For identical position detection

# Maximum array sizes (for pre-allocation compatibility)
MAX_NUM_SYMM = 1536
MAX_NUM_ATOMS = 1024
MAX_L = 3  # Maximum angular momentum (f-orbitals)

# String length limits
MAXLEN = 512
MEDIUMLEN = 256
SHORTLEN = 128

# Orbital angular momentum labels
ORBITAL_LABELS = {
    0: 's',
    1: 'p',
    2: 'd',
    3: 'f'
}

# Real harmonic orbital names for each l value
# Order follows Wannier90 convention
REAL_HARMONIC_NAMES = {
    0: ['s'],
    1: ['pz', 'px', 'py'],
    2: ['dz2', 'dxz', 'dyz', 'dx2-y2', 'dxy'],
    3: ['fz3', 'fxz2', 'fyz2', 'fz(x2-y2)', 'fxyz', 'fx(x2-3y2)', 'fy(3x2-y2)']
}

# Mapping from orbital name to (l, m)
ORBITAL_NAME_TO_LM = {
    's': (0, 0),
    'pz': (1, 0), 'px': (1, 1), 'py': (1, 2),
    'dz2': (2, 0), 'dxz': (2, 1), 'dyz': (2, 2), 'dx2-y2': (2, 3), 'dxy': (2, 4),
    'fz3': (3, 0), 'fxz2': (3, 1), 'fyz2': (3, 2),
    'fz(x2-y2)': (3, 3), 'fxyz': (3, 4), 'fx(x2-3y2)': (3, 5), 'fy(3x2-y2)': (3, 6)
}

ORBITAL_NAME_FULL_SHELL = {
    's': [(0, 0)],
    'p': [(1, 0), (1, 1), (1, 2)],
    'd': [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)],
    'f': [(3, 0), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6)]
}

# Handle alternative naming conventions
ORBITAL_NAME_MAP = {
        'dz^2': 'dz2',
        'd3z^2-r^2': 'dz2',
        'dx^2-y^2': 'dx2-y2',
        'dx2y2': 'dx2-y2',
        'dx2': 'dx2-y2',
        'dzx': 'dxz',
        'dzy': 'dyz',
    }