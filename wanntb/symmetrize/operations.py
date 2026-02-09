"""Symmetry operations utilities.

This module provides utility functions for applying symmetry operations
to positions, finding equivalent atoms, and coordinate transformations.
"""

import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict
from numba import njit
try:
    import spglib
except ImportError:
    pass

from ..constant import DEFAULT_MAGNETIC_TOLERANCE, DEFAULT_SYMM_TOLERANCE, DEFAULT_POSITION_TOLERANCE


def get_symmetry(real_lattice: np.ndarray, positions: np.ndarray, types: np.ndarray,
                 magmom_str: str|None = None,
                 tol: float = DEFAULT_SYMM_TOLERANCE,
                 tol_mag: float = DEFAULT_MAGNETIC_TOLERANCE):
    start = datetime.now()
    print('---------- start get_symmetry ----------')
    symm = None
    is_magnetic = False
    if magmom_str is not None: # magnetic systems
        magmom = parse_magmom_string(magmom_str, positions.shape[0])
        assert positions.shape[0] == magmom.shape[0], 'MAGMOM length mismatch'
        cell = (real_lattice, positions, types, magmom)
        symm = spglib.get_magnetic_symmetry(cell, symprec=tol, mag_symprec=tol_mag)
        is_magnetic = True
        data = spglib.get_symmetry_dataset(cell, symprec=tol)
    else: # non-magnetic systems
        cell = (real_lattice, positions, types)
        symm = spglib.get_symmetry(cell, symprec=tol)
        data = spglib.get_symmetry_dataset(cell, symprec=tol)
    print(data.hall)
    print(data.international)
    print(data.site_symmetry_symbols)
    print('time used: %24.2f <-- get_symmetry' % (datetime.now() - start).total_seconds())
    return SymmetryOperators(symm, is_magnetic=is_magnetic)


class SymmetryOperators:

    def __init__(self, symm, is_magnetic=True):
        self.rotations = symm['rotations'].astype(np.float64)
        self.n_operators = self.rotations.shape[0]
        self.translations = symm['translations']
        self.time_reversals = symm['time_reversals'] if is_magnetic else -np.ones(self.n_operators)
        self.is_enabled = np.ones(self.n_operators, dtype=bool)
        for idx in range(self.n_operators):
            assert np.allclose(self.rotations[idx], np.round(self.rotations[idx]), atol=DEFAULT_SYMM_TOLERANCE), \
                f"WARNING: Symmetry operation rotation matrix is not compatible with integer lattice!\n"

    def print_symmetry(self):
        print(f'number of operators: {self.n_operators}')
        for idx in range(self.n_operators):
            print(f'No. {idx}')
            print('Rotation:')
            print(np.array2string(self.rotations[idx], precision=0, suppress_small=True))
            print(f'Translation: {np.array2string(self.translations[idx], precision=6, suppress_small=True)}')
            print(f'Time_reversal: {self.time_reversals[idx]}')


    def __getitem__(self, idx):
        return self.rotations[idx], self.translations[idx], self.time_reversals[idx]

    def __len__(self):
        return self.n_operators

    def set_disabled(self, indices):
        self.is_enabled[:] = True
        for idx in indices:
            self.is_enabled[idx] = False

    def set_enabled(self, indices):
        self.is_enabled[:] = False
        for idx in indices:
            self.is_enabled[idx] = True

    def n_enabled(self):
        return np.sum(self.is_enabled.astype(int), dtype=int)


@njit(nogil=True)
def apply_symmetry_to_position(position: np.ndarray,
                               rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """Apply symmetry operation to a position in direct coordinates.

    {R|t}r = R*r + t

    Args:
        position: Position in direct coordinates [3,].
        rotation: rotation matrix (3x3) of the Symmetry operator.
        translation: translation vector (3) of the Symmetry operator

    Returns:
        Transformed position [3,].
    """
    return rotation @ position + translation

@njit(nogil=True)
def find_equivalent_atom(position: np.ndarray,
                        all_positions: np.ndarray,
                        lattice: np.ndarray,
                        tolerance: float = DEFAULT_POSITION_TOLERANCE) -> Tuple[int, np.ndarray]:
    """Find which atom a position corresponds to (modulo lattice vectors).

    Args:
        position: Position in direct coordinates [3,].
        all_positions: All atomic/orbital positions [n, 3].
        lattice: Lattice vectors [3, 3].
        tolerance: Position tolerance in Cartesian coordinates.

    Returns:
        Tuple of (index, rvec) where:
            index: Index of matching position, or -1 if not found.
            rvec: Lattice translation R such that position = all_positions[index] + R.
    """
    for i in range(all_positions.shape[0]):
        diff = position - all_positions[i]
        # Reduce to nearest image
        rvec = np.round(diff)
        diff_reduced = diff - rvec
        # Convert to Cartesian for distance check
        diff_cart = diff_reduced @ lattice
        if np.linalg.norm(diff_cart) < tolerance:
            return i, rvec
    return -1, np.zeros(3)


@njit(nogil=True)
def compose_symmetries(rotation1, translation1, time_reversal1,
                       rotation2, translation2, time_reversal2) -> Tuple[np.ndarray, np.ndarray, int]:
    """Compose two symmetry operations: symm1 ∘ symm2.

    {R1|t1} {R2|t2} = {R1*R2 | R1*t2 + t1}

    Args:
        symm1: First symmetry operation (applied second).
        symm2: Second symmetry operation (applied first).

    Returns:
        Composed symmetry operation.
    """
    new_rotation = rotation1 @ rotation2
    new_translation = rotation1 @ translation2 + translation1

    # Combine time reversal (XOR logic)
    if time_reversal1 >= 0 and time_reversal2 >= 0:
        new_tr = (time_reversal1 + time_reversal2) % 2
    else:
        new_tr = -1

    return new_rotation, new_translation, new_tr

@njit(nogil=True)
def is_identity(rotation: np.ndarray, translation: np.ndarray, tolerance: float = DEFAULT_SYMM_TOLERANCE) -> bool:
    """Check if symmetry operation is identity.

    Args:
        rotation: rotation matrix (3x3) of the Symmetry operator.
        translation: translation vector (3) of the Symmetry operator
        tolerance: Tolerance for comparison.

    Returns:
        True if identity operation.
    """
    is_rot_identity = np.allclose(rotation, np.eye(3), atol=tolerance)
    trans_reduced = translation - np.round(translation)
    is_trans_zero = np.all(np.abs(trans_reduced) < tolerance)
    return is_rot_identity and is_trans_zero

@njit(nogil=True)
def is_inversion(rotation: np.ndarray, translation: np.ndarray, tolerance: float = DEFAULT_SYMM_TOLERANCE) -> bool:
    """Check if symmetry operation is inversion.

    Args:
        rotation: rotation matrix (3x3) of the Symmetry operator.
        translation: translation vector (3) of the Symmetry operator
        tolerance: Tolerance for comparison.

    Returns:
        True if inversion operation.
    """
    is_rot_inversion = np.allclose(rotation, -np.eye(3), atol=tolerance)
    trans_reduced = translation - np.round(translation)
    is_trans_zero = np.all(np.abs(trans_reduced) < tolerance)
    return is_rot_inversion and is_trans_zero

@njit(nogil=True)
def get_rotation_order(rotation, tranlation, time_reversal) -> int:
    """Determine the order of a rotation operation.

    Args:
        symm: Symmetry operator.

    Returns:
        Order n such that symm^n = identity. Returns -1 if > 6.
    """
    current_r, current_t, current_tr = rotation, tranlation, time_reversal
    for n in range(1, 7):
        if is_identity(current_r, current_t):
            return n
        current_r, current_t, current_tr = compose_symmetries(rotation, tranlation, time_reversal,
                                                              current_r, current_t, current_tr)
    return -1

@njit(nogil=True)
def translate_to_primitive_cell(position: np.ndarray) -> np.ndarray:
    """Translate position to primitive cell [0, 1)^3.

    Args:
        position: Position in direct coordinates [3,].

    Returns:
        Position in primitive cell [3,].
    """
    return position - np.floor(position)

@njit(nogil=True)
def rotation_matrix_from_axis_angle(axis: np.ndarray,
                                    angle: float) -> np.ndarray:
    """Create rotation matrix from axis and angle.

    Uses Rodrigues' rotation formula.

    Args:
        axis: Rotation axis (unit vector) [3,].
        angle: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    return np.eye(3) + s * K + (1 - c) * (K @ K)



@njit(nogil=True)
def get_magnetic_order_type(magmom: np.ndarray,
                           tolerance: float = DEFAULT_MAGNETIC_TOLERANCE) -> int:
    """Determine the type of magnetic order.

    Args:
        magmom: Magnetic moments [natom, 3].
        tolerance: Tolerance for zero detection.

    Returns:
        Magnetic type: 0=non-magnetic, -1=AFM/compensated, 1=other, 2=FM/FiM.
    """
    # Total moment
    total_moment = np.sum(magmom, axis=0)
    total_magnitude = np.linalg.norm(total_moment)

    # Count non-zero moments
    magnitudes = np.linalg.norm(magmom, axis=1)
    nonzero_count = np.sum(magnitudes > tolerance)

    if nonzero_count == 0:
        return 0  # Non-magnetic
    elif total_magnitude < tolerance:
        return -1  # AFM or compensated
    else:
        return 2  # FM or FiM


def parse_magmom_string(magmom_str: str, n_atom: int) -> np.ndarray:
    """Parse MAGMOM string to array.

    Supports VASP format with N*val syntax:
    - "3*0.0 1*3.0" = [0, 0, 0, 3.0] for collinear
    - "0 0 3 0 0 -3" for non-collinear (SOC)

    Args:
        magmom_str: MAGMOM string.
        n_atom: Number of atoms.

    Returns:
        Magnetic moments [natom, 3] in Cartesian coordinates.
    """
    # Parse the string
    values = []
    parts = magmom_str.split()
    for part in parts:
        if '*' in part:
            rep_str, val_str = part.split('*')
            rep = int(rep_str)
            val = float(val_str)
            values.extend([val] * rep)
        else:
            values.append(float(part))
    # set the values of magmom
    values = np.array(values, dtype=np.float64)
    magmom = np.zeros((n_atom, 3), dtype=np.float64)
    if len(values) == 3 * n_atom:
        magmom[:, :] = values.reshape((n_atom,3))
    elif len(values) == n_atom:
        magmom[:, 2] = values
    else:
        raise ValueError(f"MAGMOM length {len(values)} does not match "
                         f"{n_atom} (collinear) or {3*n_atom} (non-collinear)")

    return magmom

@njit(nogil=True)
def rotate_magmom_to_saxis(magmom: np.ndarray,
                          saxis: np.ndarray) -> np.ndarray:
    """Rotate magnetic moments from SAXIS frame to Cartesian.

    In VASP, SAXIS defines the spin quantization axis. Moments in the
    input are given relative to SAXIS and need to be rotated to
    the Cartesian frame.

    Args:
        magmom: Magnetic moments in SAXIS frame [natom, 3].
        saxis: Spin axis direction [3,].

    Returns:
        Magnetic moments in Cartesian frame [natom, 3].
    """
    saxis = saxis / np.linalg.norm(saxis)

    # Rotation angles
    sx, sy, sz = saxis

    # Handle special cases
    if np.abs(sx) < 1e-10 and np.abs(sy) < 1e-10:
        # SAXIS along z
        if sz > 0:
            return magmom
        else:
            return -magmom

    # General rotation: rotate z-axis to SAXIS
    # First rotate around z by alpha, then around new y by beta

    # alpha = azimuthal angle of SAXIS
    if abs(sx) < 1e-10:
        alpha = np.pi / 2 if sy > 0 else -np.pi / 2
    else:
        alpha = np.arctan(sy / sx)
        if sx < 0:
            alpha += np.pi

    # beta = polar angle of SAXIS
    beta = np.arctan(np.sqrt(sx**2 + sy**2) / sz) if np.abs(sz) > 1e-10 else np.pi / 2

    # Rotation matrix
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)

    rot = np.array([
        [cb * ca, -sa, sb * ca],
        [cb * sa, ca, sb * sa],
        [-sb, 0, cb]
    ])

    # Apply rotation to each moment
    magmom_cart = np.zeros_like(magmom)
    for i in range(len(magmom)):
        magmom_cart[i] = rot @ magmom[i]

    return magmom_cart


def get_spacegroup_info(lattice: np.ndarray,
                        positions: np.ndarray,
                        atom_types: np.ndarray,
                        symprec: float = DEFAULT_SYMM_TOLERANCE
                        ) -> Dict:
    """Get space group information using spglib.

    Args:
        lattice: Lattice vectors [3, 3].
        positions: Atomic positions [natom, 3].
        atom_types: Atomic type indices [natom,].
        symprec: Symmetry precision.

    Returns:
        Dictionary with space group information:
            - 'international': International symbol
            - 'number': Space group number
            - 'schoenflies': Schoenflies symbol
            - 'point_group': Point group symbol
    """

    cell = (lattice, positions, atom_types)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    if dataset is None:
        return {
            'international': 'Unknown',
            'number': 0,
            'schoenflies': 'Unknown',
            'point_group': 'Unknown'
        }

    # Use attribute interface (spglib >= 2.0) or dict interface (older)
    try:
        international = dataset.international
        number = dataset.number
        pointgroup = dataset.pointgroup
        hall = dataset.hall
        hall_number = dataset.hall_number
    except AttributeError:
        # Fallback to dict interface for older spglib versions
        international = dataset['international']
        number = dataset['number']
        pointgroup = dataset['pointgroup']
        hall = dataset['hall']
        hall_number = dataset['hall_number']

    # Get Schoenflies symbol
    try:
        if hall_number:
            spg_type = spglib.get_spacegroup_type(hall_number)
            schoenflies = getattr(spg_type, 'schoenflies', None) or spg_type.get('schoenflies', 'Unknown')
        else:
            schoenflies = 'Unknown'
    except Exception:
        schoenflies = 'Unknown'

    return {
        'international': international,
        'number': number,
        'schoenflies': schoenflies,
        'point_group': pointgroup,
        'hall': hall,
        'hall_number': hall_number
    }


def get_equivalent_atoms(lattice: np.ndarray,
                        positions: np.ndarray,
                        atom_types: List[int],
                        symprec: float = DEFAULT_SYMM_TOLERANCE
                        ) -> np.ndarray:
    """Get equivalent atom mapping.

    Args:
        lattice: Lattice vectors [3, 3].
        positions: Atomic positions [natom, 3].
        atom_types: Atomic type indices [natom,].
        symprec: Symmetry precision.

    Returns:
        Array of equivalent atom indices [natom,].
    """

    cell = (lattice, positions, atom_types)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    if dataset is None:
        return np.arange(len(positions))

    try:
        return dataset.equivalent_atoms
    except AttributeError:
        return dataset['equivalent_atoms']


def get_wyckoff_letters(lattice: np.ndarray,
                       positions: np.ndarray,
                       atom_types: np.ndarray,
                       symprec: float = DEFAULT_SYMM_TOLERANCE
                       ) -> List[str]:
    """Get Wyckoff letters for each atom.

    Args:
        lattice: Lattice vectors [3, 3].
        positions: Atomic positions [natom, 3].
        atom_types: Atomic type indices [natom,].
        symprec: Symmetry precision.

    Returns:
        List of Wyckoff letters.
    """

    cell = (lattice, positions, atom_types)

    dataset = spglib.get_symmetry_dataset(cell, symprec=symprec)

    if dataset is None:
        return ['a'] * len(positions)

    wyckoff_letters = 'abcdefghijklmnopqrstuvwxyz'
    try:
        wyckoffs = dataset.wyckoffs
    except AttributeError:
        wyckoffs = dataset['wyckoffs']
    return [wyckoff_letters[w] for w in wyckoffs]
