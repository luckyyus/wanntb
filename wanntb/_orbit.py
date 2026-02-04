import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict

from .constant import ORBITAL_NAME_FULL_SHELL, ORBITAL_NAME_TO_LM, EPS5, ORBITAL_NAME_MAP


def orbital_info(projections: List[Dict], real_lattice: NDArray,
                 atom_pos: NDArray, atom_names: List[str], atom_counts: NDArray,
                 is_soc=True, order='uudd') -> Tuple[NDArray, NDArray, NDArray]:
    """Derive Wannier orbital information from input configuration.

    Args:
        config: Parsed input configuration.

    Returns:
        List of WannOrb objects describing all Wannier orbitals.
    """
    orb_pos = []
    orb_lmsr = []
    orb_laxis = []

    # Build mapping from element name to site indices (Case-insensitive)
    element_sites = {}
    idx = 0
    for i, name in enumerate(atom_names):
        element_sites[name.lower()] = list(range(idx, idx + atom_counts[i]))
        idx += atom_counts[i]

    # Process each projection group
    for pj in projections:
        # Parse local axes if specified

        # Complete orthogonal axes
        if 'zaxis' in pj.keys():
            zaxis, xaxis, yaxis = _complete_axes(pj['zaxis'], pj['xaxis'], pj['yaxis'])
            local_axes = np.array([xaxis, yaxis, zaxis])
        else:
            local_axes = np.eye(3, dtype=np.float64)
        # Parse radial quantum number
        radial = pj['r'] if 'r' in pj.keys() else 1
        sites = []
        # Get sites for this projection
        if pj['element'].startswith('f=') or pj['element'].startswith('c='):
            # Coordinate specification
            coord_type = pj['element'][0]
            coord_str = pj['element'][2:]
            site = _parse_coordinate(coord_str)
            if coord_type == 'c':
                # Cartesian to direct
                site = site @ np.linalg.inv(real_lattice)
            sites = [site]
        else:
            # Element name (Case-insensitive)
            elem_key = pj['element'].lower()
            if elem_key in element_sites:
                site_indices = element_sites[elem_key]
                sites = [atom_pos[i] for i in site_indices]
            else:
                print(f"Warning: Element '{pj['element']}' not found in structure")
                continue

        # Process each site
        for site in sites:
            # Process each orbital
            for orb_name in pj['orb_names']:
                orbs = _expand_orbital_name(orb_name)
                for l, mr in orbs:
                    # Create orbital (spin up)
                    orb_pos.append(site)
                    orb_lmsr.append([l,mr, 0, radial])
                    orb_laxis.append(local_axes.copy())
                    # For SOC with 'udud; orbital order
                    if is_soc and order=='udud':
                        orb_pos.append(site)
                        orb_lmsr.append([l, mr, 1, radial])
                        orb_laxis.append(local_axes.copy())

    # For VASP with SOC, spin-down orbitals come after all spin-up
    if is_soc and order != 'udud':
        n_half = len(orb_pos)
        orb_pos.extend(orb_pos)
        for i in range(n_half):
            orb_lmsr.append([orb_lmsr[i][0],orb_lmsr[i][1], 1, orb_lmsr[i][3]])
        orb_laxis.extend(orb_laxis)
    orb_pos = np.array(orb_pos, dtype=np.float64)
    orb_lmsr = np.array(orb_lmsr, dtype=np.uint8)
    orb_laxis = np.array(orb_laxis, dtype=np.float64)
    return orb_pos, orb_lmsr, orb_laxis

def _parse_axis_string(s: str) -> NDArray:
    """Parse axis string like '1,0,0' to numpy array."""
    if not s or s == '0,0,0':
        return np.zeros(3)
    parts = s.replace(' ', '').split(',')
    if len(parts) >= 3:
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])])
    return np.zeros(3)


def _parse_coordinate(s: str) -> NDArray:
    """Parse coordinate string to numpy array."""
    parts = s.replace(' ', '').split(',')
    if len(parts) >= 3:
        return np.array([float(parts[0]), float(parts[1]), float(parts[2])])
    raise ValueError(f"Invalid coordinate string: {s}")


def _complete_axes(zaxis: NDArray, xaxis: NDArray,
                  yaxis: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """Complete orthogonal axis system from partially specified axes."""
    z_norm = np.linalg.norm(zaxis)
    x_norm = np.linalg.norm(xaxis)
    y_norm = np.linalg.norm(yaxis)

    # Count how many axes are specified
    n_specified = sum([z_norm > EPS5, x_norm > EPS5, y_norm > EPS5])

    if n_specified == 0:
        # Default: identity
        return np.array([0., 0., 1.]), np.array([1., 0., 0.]), np.array([0., 1., 0.])

    if n_specified == 3:
        # All specified, just normalize
        return (zaxis / z_norm, xaxis / x_norm, yaxis / y_norm)

    if n_specified == 2:
        if z_norm < EPS5:
            xaxis = xaxis / x_norm
            yaxis = yaxis / y_norm
            zaxis = np.cross(xaxis, yaxis)
        elif x_norm < EPS5:
            yaxis = yaxis / y_norm
            zaxis = zaxis / z_norm
            xaxis = np.cross(yaxis, zaxis)
        else:  # y not specified
            xaxis = xaxis / x_norm
            zaxis = zaxis / z_norm
            yaxis = np.cross(zaxis, xaxis)
    else:  # n_specified == 1
        if z_norm > EPS5:
            zaxis = zaxis / z_norm
            # Choose x perpendicular to z
            if abs(zaxis[0]) < 0.9:
                xaxis = np.cross(zaxis, np.array([1., 0., 0.]))
            else:
                xaxis = np.cross(zaxis, np.array([0., 1., 0.]))
            xaxis = xaxis / np.linalg.norm(xaxis)
            yaxis = np.cross(zaxis, xaxis)
        elif x_norm > EPS5:
            xaxis = xaxis / x_norm
            zaxis = np.array([0., 0., 1.])
            yaxis = np.cross(zaxis, xaxis)
            yaxis = yaxis / np.linalg.norm(yaxis)
        else:
            yaxis = yaxis / y_norm
            zaxis = np.array([0., 0., 1.])
            xaxis = np.cross(yaxis, zaxis)
            xaxis = xaxis / np.linalg.norm(xaxis)

    return zaxis, xaxis, yaxis


def _expand_orbital_name(name: str) -> List[Tuple[int, int]]:
    """Expand orbital name to list of (l, m) tuples.

    Handles both single orbitals (e.g., 'dxy') and full shells (e.g., 'd').
    """
    name = name.lower().strip()

    # Check for full shell
    if name in ORBITAL_NAME_FULL_SHELL.keys():
        return ORBITAL_NAME_FULL_SHELL[name]

    # Check for specific orbital
    if name in ORBITAL_NAME_TO_LM.keys():
        return [ORBITAL_NAME_TO_LM[name]]

    if name in ORBITAL_NAME_MAP.keys():
        return [ORBITAL_NAME_TO_LM[ORBITAL_NAME_MAP[name]]]

    raise ValueError(f"Unknown orbital name: {name}")


