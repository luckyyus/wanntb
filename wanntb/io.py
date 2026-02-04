from typing import Tuple, List, Optional

import numpy as np

from .symmetrize._spg import SymmetryOperator
from .constant import TwoPi, EPS5, EPS6
from .utility import hermiization_R


def read_poscar(filename: str, real_lattice) -> Tuple[np.ndarray, np.ndarray,
                                         List[str], List[int]]:
    """Read VASP POSCAR file.

    Args:
        filename: Path to POSCAR file.
        real_lattice: the lattice from Hamiltonian file

    Returns:
        Tuple of (lattice, positions, atom_names, atom_counts) where:
            lattice: Lattice vectors [3, 3] in Angstrom.
            positions: Atomic positions [natom, 3] in direct coordinates.
            atom_names: List of element names.
            atom_counts: Number of atoms of each type.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file format is invalid.
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    line_idx = 0

    # Skip comment line(s) until we find scaling factor
    while line_idx < len(lines):
        line = lines[line_idx].strip()
        # Check if line starts with a number (scaling factor)
        try:
            first_val = float(line.split()[0])
            if not any(c.isalpha() for c in line.split()[0]):
                break
        except (ValueError, IndexError):
            pass
        line_idx += 1

    # Read scaling factor
    scaling = float(lines[line_idx].strip())
    line_idx += 1

    # Read lattice vectors
    lattice = np.zeros((3, 3))
    for i in range(3):
        parts = lines[line_idx].split()
        lattice[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        lattice[i] *= scaling
        line_idx += 1

    assert np.allclose(lattice, real_lattice, atol=EPS5), 'lattice mismatch'
    # Read atom names (6th line in VASP 5.x format)
    atom_names = []
    line = lines[line_idx].strip()
    # Check if this line contains element names or numbers
    parts = line.split()
    try:
        # If first part is a number, this is old POSCAR format
        int(parts[0])
        # No element names in file - this is old format
        # We'll use generic names
        atom_counts = [int(p) for p in parts]
        atom_names = [f"X{i+1}" for i in range(len(atom_counts))]
    except ValueError:
        # Element names present
        atom_names = parts
        line_idx += 1
        # Read atom counts
        parts = lines[line_idx].split()
        atom_counts = [int(p) for p in parts]

    line_idx += 1

    # Check for Selective dynamics or Direct/Cartesian
    line = lines[line_idx].strip()
    if line[0].upper() == 'S':  # Selective dynamics
        line_idx += 1
        line = lines[line_idx].strip()

    # Determine coordinate type
    is_cartesian = line[0].upper() in ['C', 'K']
    line_idx += 1

    # Read atomic positions
    total_atoms = sum(atom_counts)
    positions = np.zeros((total_atoms, 3))
    for i in range(total_atoms):
        parts = lines[line_idx].split()
        positions[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
        line_idx += 1

    # Convert Cartesian to Direct if needed
    if is_cartesian:
        inv_lattice = np.linalg.inv(lattice)
        positions = positions @ inv_lattice

    return lattice, positions, atom_names, atom_counts


def read_tb_file(tb_file='wannier90_tb.dat'):
    seedname = tb_file.split('/')[-1].split('_')[0]
    # read tb file
    with open(tb_file, 'r') as f:
        line = f.readline()
        print("reading tb file %s ( %s )" % (tb_file, line.strip()))
        # real lattice
        real_lattice = np.array([f.readline().split()[:3] for i in range(3)], dtype=np.float64)
        print('real lattice:')
        print(real_lattice)
        recip_lattice = np.linalg.inv(real_lattice).T * TwoPi
        print('reciprocal lattice:')
        print(recip_lattice)
        num_wann = int(f.readline())
        n_Rpts = int(f.readline())
        # degenerate rpt
        ndegen = []
        while len(ndegen) < n_Rpts:
            ndegen += f.readline().split()
        n_degen = np.array(ndegen, dtype=np.uint8)
        # initialize R_vec and Ham_R
        irpt = []
        ham_R = np.zeros((n_Rpts, num_wann, num_wann), dtype=np.complex128)
        # read each R_vec[nRvec0, 3] and Ham_R[n_Rpts, num_wann, num_wann]
        for ir in range(n_Rpts):
            f.readline()
            irpt.append(f.readline().split())
            hh = np.array(
                [[f.readline().split()[2:4] for n in range(num_wann)] for m in range(num_wann)],
                dtype=float).transpose((2, 1, 0))
            ham_R[ir, :, :] = (hh[0, :, :] + 1j * hh[1, :, :])
        # R_vec = np.ascontiguousarray(irpt, dtype=np.float64)  # 为了和k.R正常一些
        R_vec = np.ascontiguousarray(irpt, dtype=np.int16)  # 为了和k.R正常一些
        ham_R = np.ascontiguousarray(ham_R)
        print('ham_R: %s %s' % (ham_R.dtype, list(ham_R.shape)))
        print('R_vec: %s %s' % (R_vec.dtype, list(R_vec.shape)))
        print('n_degen: %s %s' % (n_degen.dtype, list(n_degen.shape)))
        r_mat_R = np.zeros((n_Rpts, 3, num_wann, num_wann), dtype=np.complex128)
        for ir in range(n_Rpts):
            f.readline()
            assert (np.array(f.readline().split(), dtype=np.int16) == R_vec[ir]).all()
            aa = np.array([[f.readline().split()[2:8]
                            for n in range(num_wann)]
                           for m in range(num_wann)], dtype=np.float64)
            r_mat_R[ir, :, :, :] = (aa[:,:,0::2] + 1j*aa[:,:,1::2]).transpose((2,1,0))
        print('r_mat_R: %s %s' % (r_mat_R.dtype, list(r_mat_R.shape)))
    for ir in range(n_Rpts):
        ham_R[ir] /= n_degen[ir]
        r_mat_R[ir] /= n_degen[ir]
    ham_R = hermiization_R(ham_R, R_vec)
    r_mat_R = hermiization_R(r_mat_R, R_vec)
    n_degen[:] = 1
    # real_lattice[3,3] float
    # recip_lattice[3,3] float
    # ham_R[n_Rpts, num_wann, num_wann] complex
    # R_vec[n_Rpts, 3] int16
    # n_degen[n_Rpts] uint8 # all waight is one
    # r_mat_R[n_Rpts, 3, num_wann, num_wann] complex
    return {'seedname': seedname,
            'num_wann': num_wann,
            'real_lattice': real_lattice,
            'recip_lattice': recip_lattice,
            'n_Rpts': n_Rpts,
            'ham_R': ham_R,
            'R_vec': R_vec,
            'n_degen': n_degen,
            'r_mat_R': r_mat_R}


def read_spin_file(R_vec, n_Rpts, num_wann, ss_file='wannier90_SS_R.dat'):
    ss_R = np.zeros((n_Rpts, 3, num_wann, num_wann), dtype=np.complex128)
    with open(ss_file, 'r') as f:
        line = f.readline()
        print("reading spin file %s ( %s )" % (ss_file, line.strip()))
        assert int(f.readline()) == num_wann
        assert int(f.readline()) == n_Rpts
        ndegen = []
        while len(ndegen) < n_Rpts:
            ndegen += f.readline().split()
        n_degen = np.array(ndegen, dtype=np.uint8)

        for ir in range(n_Rpts):
            f.readline()
            assert (np.array(f.readline().split(), dtype=np.int32) == R_vec[ir]).all(), 'R_vec mismatch'
            aa = np.array([[f.readline().split()[2:8]
                            for n in range(num_wann)]
                           for m in range(num_wann)], dtype=np.float64)
            ss_R[ir, :, :, :] = (aa[:,:,0::2] + 1j*aa[:,:,1::2]).transpose((2,1,0))
        print('ss_R: %s %s' % (ss_R.dtype, list(ss_R.shape)))
    for ir in range(n_Rpts):
        ss_R[ir] /= n_degen[ir]
    _ss_R = hermiization_R(ss_R, R_vec)
    return _ss_R


"""Symmetry file I/O.

This module provides functions to read and write symmetry operation files.
"""


def read_symmetry_file(filename: str) -> Tuple[List[SymmetryOperator], bool]:
    """Read symmetry operations from file.

    File format:
        nsymm = N
        --- 1 ---
        rot[0,0] rot[0,1] rot[0,2]
        rot[1,0] rot[1,1] rot[1,2]
        rot[2,0] rot[2,1] rot[2,2]
        t[0] t[1] t[2] [T/F for time reversal]
        ...

    Args:
        filename: Path to symmetry file.

    Returns:
        Tuple of (symmetry_operators, flag_global_trsymm).
    """
    symmetries = []
    flag_global_trsymm = True

    with open(filename, 'r') as f:
        lines = f.readlines()

    line_idx = 0

    # Find nsymm
    nsymm = 0
    while line_idx < len(lines):
        line = lines[line_idx].strip().lower()
        if line.startswith('nsymm'):
            parts = line.split('=')
            if len(parts) >= 2:
                nsymm = int(parts[1].strip())
            break
        # Check for global time reversal
        if 'time-reversal' in line or 'global' in line:
            if 'false' in line:
                flag_global_trsymm = False
        line_idx += 1

    line_idx += 1

    # Read symmetry operations
    for _ in range(nsymm):
        # Skip to '---' line
        while line_idx < len(lines):
            if '---' in lines[line_idx]:
                break
            line_idx += 1
        line_idx += 1

        # Read rotation matrix
        rotation = np.zeros((3, 3))
        for i in range(3):
            parts = lines[line_idx].split()
            rotation[i] = [float(parts[0]), float(parts[1]), float(parts[2])]
            line_idx += 1

        # Read translation and optional time reversal
        parts = lines[line_idx].split()
        translation = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
        time_reversal = -1
        if len(parts) >= 4:
            tr_str = parts[3].upper()
            if tr_str == 'T':
                time_reversal = 1
            elif tr_str == 'F':
                time_reversal = 0
        line_idx += 1

        symmetries.append(SymmetryOperator(
            rotation=rotation,
            translation=translation,
            time_reversal=time_reversal
        ))

    return symmetries, flag_global_trsymm


def write_symmetry_file(filename: str, symmetries: List[SymmetryOperator],
                        flag_global_trsymm: bool = True,
                        space_group_info: Optional[str] = None):
    """Write symmetry operations to file.

    Args:
        filename: Output file path.
        symmetries: List of symmetry operators.
        flag_global_trsymm: Global time reversal symmetry flag.
        space_group_info: Optional space group information string.
    """
    with open(filename, 'w') as f:
        if space_group_info:
            f.write(f"space group information:\n{space_group_info}\n")

        f.write(f"global time-reversal symmetry = {'True' if flag_global_trsymm else 'False'}\n")
        f.write(f"nsymm = {len(symmetries)}\n")

        for i, symm in enumerate(symmetries):
            f.write(f"--- {i + 1} ---\n")
            for j in range(3):
                f.write(f"{int(round(symm.rotation[j, 0])):2d} "
                       f"{int(round(symm.rotation[j, 1])):2d} "
                       f"{int(round(symm.rotation[j, 2])):2d}\n")
            tr_str = ""
            if not flag_global_trsymm and symm.time_reversal >= 0:
                tr_str = " T" if symm.time_reversal == 1 else " F"
            f.write(f"{symm.translation[0]:.6f} "
                   f"{symm.translation[1]:.6f} "
                   f"{symm.translation[2]:.6f}{tr_str}\n")


def format_symmetry_description(symm: SymmetryOperator, lattice: np.ndarray,
                                index: int = 0) -> str:
    """Format a symmetry operation as a human-readable string.

    Args:
        symm: Symmetry operator.
        lattice: Lattice vectors [3, 3].
        index: Symmetry index for labeling.

    Returns:
        Formatted description string.
    """
    from .symmetrize._rotate import rotation_to_axis_angle

    axis, angle, is_inversion = rotation_to_axis_angle(symm.rotation, lattice)
    angle_deg = np.degrees(angle)

    if abs(angle) < 0.1 and symm.time_reversal != 1:
        trans_norm = np.linalg.norm(symm.translation)
        if trans_norm < 0.01:
            if is_inversion:
                return f"symm{index + 1:4d}: Inversion"
            else:
                return f"symm{index + 1:4d}: Identity"

    desc = f"symm{index + 1:4d}:{angle_deg:6.1f} deg rot around "
    desc += f"({axis[0]:7.4f},{axis[1]:7.4f},{axis[2]:7.4f})"

    trans_norm = np.linalg.norm(symm.translation)
    if trans_norm > 0.01:
        desc += f" with trans ({symm.translation[0]:7.4f} "
        desc += f"{symm.translation[1]:7.4f} {symm.translation[2]:7.4f})"

    if is_inversion:
        if abs(abs(angle_deg) - 180) < 1.0:
            desc += " with inv (Mirror)"
        else:
            desc += " with inv"

    if symm.time_reversal == 1:
        desc += " with TRS"

    return desc
