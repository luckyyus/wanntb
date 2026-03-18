import numpy as np
from numpy.typing import NDArray
from typing import Tuple
from numba import njit, types
from numba.typed import typeddict


from ..constant import S_, TwoPi, V2, EPS6, MAX_L

from ..utility import normalize_vector


@njit(nogil=True)
def L_matrix(l: int) -> NDArray:
    """
    生成角动量算符L_x, L_y, L_z的矩阵表示（用于球谐函数基）。
    参数:
        l: 角动量量子数。
    返回:
        元组 (Lx, Ly, Lz)，每个是(2l+1)x(2l+1)的复数矩阵。
    """
    dim = 2 * l + 1
    Lmat = np.zeros((3, dim, dim), dtype=np.complex128)
    Lz = np.zeros((dim, dim), dtype=np.complex128)
    Lp = np.zeros((dim, dim), dtype=np.complex128)  # 升算符L+
    Lm = np.zeros((dim, dim), dtype=np.complex128)  # 降算符L-
    
    # 填充Lz和升降算符
    for m in range(-l, l+1):
        idx = m + l  # 索引从0开始
        Lz[idx, idx] = m
        if m < l:
            Lp[idx+1, idx] = np.sqrt((l - m) * (l + m + 1))  # L+|lm>
            Lm[idx, idx+1] = Lp[idx+1, idx]  # L-|lm>
    
    # Lx = (L+ + L-)/2, Ly = (L+ - L-)/(2j)
    Lmat[0] = (Lp + Lm) / 2
    Lmat[1] = (Lp - Lm) / 2.0j
    Lmat[2] = Lz
    
    return Lmat

@njit(nogil=True)
def Y2R_R2Y(l: int) -> Tuple[NDArray, NDArray]:
    """
    生成球谐函数(Ylm)和实球谐函数(Real Spherical Harmonics)之间的变换矩阵。
    
    参数:
        l: 角动量量子数 (0=s, 1=p, 2=d, 3=f)
    
    返回:
        R2Y: (shape: (2l+1) x (2l+1))
        R2Y:  (shape: (2l+1) x (2l+1))
    """
    dim = int(2 * l + 1)
    
    # 初始化变换矩阵
    Y2R = np.zeros((dim, dim), dtype=np.complex128)
    
    if l == 0:  # s轨道
        Y2R[0, 0] = 1.0
    elif l == 1:  # p轨道
        # pz = |1,0>
        Y2R[0, 1] = 1.0  # pz对应m=0
        
        # px = (|1,-1> - |1,1>)/sqrt(2)
        Y2R[1, 0] = 1.0/V2   # m=-1分量
        Y2R[1, 2] = -1.0/V2  # m=+1分量
        
        # py = i(|1,-1> + |1,1>)/sqrt(2)
        Y2R[2, 0] = 1j/V2    # m=-1分量
        Y2R[2, 2] = 1j/V2    # m=+1分量
        
    elif l == 2:  # d轨道
        # dz2 = |2,0>
        Y2R[0, 2] = 1.0
        
        # dzx = (|2,-1> - |2,1>)/sqrt(2)
        Y2R[1, 1] = 1.0/V2
        Y2R[1, 3] = -1.0/V2
        
        # dyz = i(|2,-1> + |2,1>)/sqrt(2)
        Y2R[2, 1] = 1j/V2
        Y2R[2, 3] = 1j/V2
        
        # dx2-y2 = (|2,-2> + |2,2>)/sqrt(2)
        Y2R[3, 0] = 1.0/V2
        Y2R[3, 4] = 1.0/V2
        
        # dxy = i(|2,-2> - |2,2>)/sqrt(2)
        Y2R[4, 0] = 1j/V2
        Y2R[4, 4] = -1j/V2
        
    elif l == 3:  # f轨道
        # fz3 = |3,0>
        Y2R[0, 3] = 1.0
        
        # fxz2 = (|3,-1> - |3,1>)/sqrt(2)
        Y2R[1, 2] = 1.0/V2
        Y2R[1, 4] = -1.0/V2
        
        # fyz2 = i(|3,-1> + |3,1>)/sqrt(2)
        Y2R[2, 2] = 1j/V2
        Y2R[2, 4] = 1j/V2
        
        # fz(x2-y2) = (|3,-2> + |3,2>)/sqrt(2)
        Y2R[3, 1] = 1.0/V2
        Y2R[3, 5] = 1.0/V2
        
        # fxyz = i(|3,-2> - |3,2>)/sqrt(2)
        Y2R[4, 1] = 1j/V2
        Y2R[4, 5] = -1j/V2
        
        # fx3-3xy2 = (|3,-3> - |3,3>)/sqrt(2)
        Y2R[5, 0] = 1.0/V2
        Y2R[5, 6] = -1.0/V2
        
        # fy(3x2-y2) = i(|3,-3> + |3,3>)/sqrt(2)
        Y2R[6, 0] = 1j/V2
        Y2R[6, 6] = 1j/V2
    
    else:
        # 对于其他l值，使用单位矩阵（简化处理）
        Y2R = np.eye(dim, dtype=np.complex128)
    
    # 计算逆矩阵：Y2C = C2Y^{-1}
    R2Y = np.linalg.inv(Y2R)
    
    return Y2R, R2Y

@njit(nogil=True)
def rotate_Ylm(l: int, axis: NDArray|tuple[float, float, float], alpha: float, inversion=False) -> NDArray:
    """
    生成球谐函数基下的旋转矩阵。
    参数:
        l: 角动量量子数。
        axis: 旋转轴。
        alpha: 旋转角度。
        inversion: 是否反演。
    返回:
        (2l+1)x(2l+1)旋转矩阵。
    """
    Lx, Ly, Lz = L_matrix(l)
    # 旋转生成器: n·L
    L_dot_n = axis[0] * Lx + axis[1] * Ly + axis[2] * Lz
    # 旋转矩阵: exp(-i * alpha * n·L)

    # rot_r = expm(-1j * alpha * L_dot_n)
    exp_n = alpha * L_dot_n
    eigenvalues, eigenvectors = np.linalg.eigh(exp_n)

    # exp(M) = V @ diag(exp(-i * eigenvalues)) @ V^{-1}
    exp_eig = np.diag(np.exp(-1j * eigenvalues))
    rot_r = eigenvectors @ exp_eig @ np.conj(eigenvectors.T)
    if inversion:
        # 反演处理：根据l的奇偶性调整符号
        if l % 2 == 1:
            rot_r = -rot_r
    
    return rot_r

@njit(nogil=True)
def rotate_real_Ylm(l: int, axis: NDArray|tuple[float, float, float], alpha: float, inversion=False) -> NDArray:
        """
        生成实球谐函数基下的旋转矩阵。
        参数:
            l: 角动量量子数。
            axis: 旋转轴。
            alpha: 旋转角度。
            inversion: 是否反演。
        返回:
            (2l+1)x(2l+1)旋转矩阵。
        """
        y2r, r2y = Y2R_R2Y(l)
        rot_y = rotate_Ylm(l, axis, alpha, inversion)
        rot_r = y2r @ rot_y @ r2y  # 是对的！
        return rot_r

@njit(nogil=True)
def get_all_L_rotation_matrix(axis, angle, is_inversion):
    max_dim = 2 * MAX_L + 1
    l_rot = np.zeros((MAX_L + 1, max_dim, max_dim), dtype=np.complex128)
    for l in range(MAX_L + 1):
        dim = 2 * l + 1
        l_rot[l, 0:dim, 0:dim] = rotate_real_Ylm(l, axis, angle, is_inversion)
    return l_rot


# # Pre-compute transformation matrices for efficiency (not used)
# _CACHED_Y2R = np.zeros((MAX_L + 1, 2 * MAX_L + 1, 2 * MAX_L + 1), dtype=np.complex128)
# _CACHED_R2Y = np.zeros((MAX_L + 1, 2 * MAX_L + 1, 2 * MAX_L + 1), dtype=np.complex128)
# _CACHED_L = np.zeros(MAX_L, dtype=np.bool)
#
# @njit(nogil=True)
# def Y2R_R2Y_cached(l: int, _cached_y2r, _cached_r2y, _cached_l) -> Tuple[NDArray, NDArray]:
#     """Get cached Ylm/real transformation matrices."""
#     dim = 2 * l + 1
#     if not _cached_l[l]:
#         _cached_y2r[l,0:dim,0:dim], _cached_r2y[l,0:dim,0:dim] = Y2R_R2Y(l)
#         _cached_l[l] = True
#     return _cached_y2r[l,0:dim,0:dim], _cached_r2y[l,0:dim,0:dim]

@njit(nogil=True)
def combine_rotation_with_local_axis(rotation_cart: NDArray,
                                     source_axis: NDArray,
                                     target_axis: NDArray) -> NDArray:
    """Combine global rotation with local axis transformation.

    When orbitals have local coordinate axes, the effective rotation
    seen by the orbital is: R_local = A_target^{-1} @ R_global @ A_source
    where A is the local axis transformation matrix.

    Args:
        rotation_cart: Global rotation in Cartesian coordinates [3, 3].
        source_axis: Source orbital local axes [3, 3].
        target_axis: Target orbital local axes [3, 3].

    Returns:
        Effective rotation in local coordinates [3, 3].
    """
    # A_source: rows are local x, y, z axes
    # The rotation that takes global coords to local is A^T
    rot_local = target_axis @ rotation_cart @ source_axis.T
    return rot_local


@njit(nogil=True)
def rotate_spinor(axis: NDArray, alpha: float, inversion=False) -> NDArray:
    """
    生成SU(2)旋量旋转矩阵。
    参数:
        axis: 3元素数组，旋转轴（单位向量）。
        alpha: 旋转角度。
        inv: 是否包含反演（默认为False）。
    返回:
        2x2复数矩阵，表示旋转。
    """
    norm = np.linalg.norm(axis)
    if norm > 1e-10:
        axis = axis / norm
    else:
        # Zero axis means no rotation
        return np.eye(2, dtype=np.complex128)

    # 旋转轴点乘Pauli矩阵
    sigma_n = axis[0] * S_[0] + axis[1] * S_[1] + axis[2] * S_[2]
    # 计算旋转矩阵: exp(-i * alpha * (n·sigma) / 2)
    # rot_spin = expm(-1j * alpha * sigma_n / 2)
    exp_n =  alpha * sigma_n / 2
    eigenvalues, eigenvectors = np.linalg.eigh(exp_n)

    # exp(M) = V @ diag(exp(eigenvalues)) @ V^{-1}
    exp_eig = np.diag(np.exp(-1j * eigenvalues))
    rot_spin = eigenvectors @ exp_eig @ np.conj(eigenvectors.T)

    if inversion:
        rot_spin *= -1j  # inv时乘以-i
    
    return rot_spin


# def get_rotation_matrix(orb_pos: NDArray, orb_lmsr: NDArray, orb_laxis: NDArray,
#                         rotation: NDArray, translation: NDArray, time_reversal: NDArray,
#                         symm_index_in_double_group: int,
#                         lattice: NDArray, kpt: NDArray,
#                         is_soc: bool = False, is_laxis: bool = False
#                         ) -> NDArray:
#     """Compute symmetry operator matrix in Wannier orbital basis. (not used)
#
#     This matrix represents how Wannier orbitals at a given k-point transform
#     under a symmetry operation.
#
#     Args:
#         orb_info: List of WannOrb objects describing all orbitals.
#         symm: Symmetry operator {R|t}.
#         symm_index_in_double_group: Index in double group. For single-valued
#             group elements (0 to nsymm-1), index >= 0. For double-valued group
#             elements (additional elements due to spinor 4π periodicity),
#             index < 0 (specifically, -(original_index + 1)).
#         lattice: Lattice vectors [3, 3].
#         kpt: k-point in reciprocal coordinates [3,].
#         is_soc: If True, include spinor rotation.
#         is_laxis: If True, account for local orbital axes.
#
#     Returns:
#         Symmetry operator matrix [norb, norb] as complex array.
#     """
#     norb = len(orb_pos)
#     sym_op = np.zeros((norb, norb), dtype=np.complex128)
#
#     # Get rotation in Cartesian coordinates for axis-angle computation
#     rot_cart = rotation_in_cart(rotation, lattice)
#     axis, angle, is_inversion = rotation_to_axis_angle(rotation, lattice)
#
#     # For double group, add 2π to angle
#     if is_soc and symm_index_in_double_group < 0:
#         angle += TwoPi
#
#     # Get k' = R @ k (or -R @ k for time reversal)
#     rot_k = np.linalg.inv(rotation).T  # Rotation in reciprocal space
#     kpt_rotated = rot_k @ kpt
#     if time_reversal == 1:
#         kpt_rotated = -kpt_rotated
#
#     # Compute spinor rotation (same for all orbitals)
#     if is_soc:
#         s_rot = rotate_spinor(axis, angle)
#     else:
#         s_rot = np.eye(2, dtype=np.complex128)
#
#     # Pre-compute orbital rotation matrices for each l value for one symmetry operator
#     # orb_rot = nb.typed.Dict.empty(key_type=nb.types.uint8, value_type=nb.types.float64[:, :])
#     orb_rot = {}
#     for l in range(4):
#         orb_rot[l] = rotate_real_Ylm(l, axis, angle, is_inversion)
#
#     # Build the symmetry operator matrix
#     for io in range(norb):
#         for jo in range(norb):
#             # Check if angular momentum matches
#             if orb_lmsr[io, 0] != orb_lmsr[jo, 0]:
#             # if orb_info[io].l != orb_info[jo].l:
#                 continue
#
#             # Transform site j under symmetry
#             tau_j_transformed = rotation @ orb_pos[jo] + translation
#
#             # Find which orbital site this maps to (with lattice translation)
#             idx, rvec = find_equivalent_atom(tau_j_transformed, orb_pos, lattice)
#
#             # tau' should equal tau_i (modulo lattice vector)
#             if idx < 0 or not np.allclose(orb_pos[idx], orb_pos[io], atol=EPS5):
#                 continue
#
#             # Get orbital indices
#             l = orb_lmsr[jo, 0]
#             # ml is 0-indexed
#             mr_i = orb_lmsr[io, 1]
#             mr_j = orb_lmsr[jo, 1]
#             # ms is 0-up, 1-down
#             ms_i = orb_lmsr[io, 2]
#             ms_j = orb_lmsr[jo, 2]
#
#             # Handle local axes if needed
#             if is_laxis:
#                 # Combine rotation with local axis transformation
#                 axis_j = orb_laxis[jo]
#                 axis_i = orb_laxis[io]
#                 rot_combined = combine_rotation_with_local_axis(
#                     rot_cart, axis_j, axis_i
#                 )
#                 # Recompute axis-angle for combined rotation
#                 la_axis, la_angle, la_inv = rotation_to_axis_angle(
#                     rot_combined,
#                     np.eye(3)  # Already in Cartesian
#                 )
#                 orb_rot_local = rotate_real_Ylm(l, la_axis, la_angle, la_inv)
#                 orb_elem = orb_rot_local[mr_i, mr_j]
#             else:
#                 orb_elem = orb_rot[l][mr_i, mr_j]
#
#             # Spinor element
#             spin_elem = s_rot[ms_i, ms_j]
#
#             # Phase factor: exp(-2πi k' · (R_vec - t))
#             # where R_vec is the lattice translation needed
#             phase_arg = np.dot(kpt_rotated, rvec - translation)
#             phase = np.exp(-1j * TwoPi * phase_arg)
#
#             # Total matrix element
#             sym_op[io, jo] = phase * orb_elem * spin_elem
#
#     return sym_op


@njit(nogil=True)
def rotation_in_cart(rotation_direct: NDArray, lattice: NDArray) -> NDArray:
    """Convert rotation from direct to Cartesian coordinates.

    R_cart = L^T @ R_direct @ L^{-T}

    where L is the lattice matrix with lattice vectors as rows.

    Args:
        rotation_direct: Rotation matrix in direct coordinates [3, 3].
        lattice: Lattice vectors as row vectors [3, 3].

    Returns:
        Rotation matrix in Cartesian coordinates [3, 3].
    """
    lattice_t = lattice.T
    lattice_t_inv = np.linalg.inv(lattice_t)
    return lattice_t @ rotation_direct @ lattice_t_inv


@njit(nogil=True)
def rotation_in_reciprocal(rotation_direct: NDArray) -> NDArray:
    """Convert rotation matrix or symmetry operator matrix from direct to reciprocal space.

    In reciprocal space: k' = R^{-T} k = (R^{-1})^T k

    For orthogonal rotations: R^{-T} = R, so this is trivial.
    For general rotations in direct coords: the reciprocal rotation is R^{-T}.

    Args:
        rotation_direct: Rotation matrix in direct coordinates [3, 3] or symmetry operator matrix [num_wann, num_wann].

    Returns:
        Rotation matrix for k-vectors [3, 3] or [num_wann, num_wann].
    """
    return np.linalg.inv(rotation_direct).T


@njit(nogil=True)
def rotation_to_axis_angle(rotation: NDArray, lattice: NDArray) -> Tuple[NDArray, float, bool]:
    """Convert rotation matrix to axis-angle representation.

    Args:
        rotation: 3x3 rotation matrix in direct coordinates.
        lattice: Lattice vectors [3, 3] for Cartesian conversion.

    Returns:
        Tuple of (axis, angle, is_inversion) where:
            axis: Unit rotation axis in Cartesian coordinates [3,].
            angle: Rotation angle in radians.
            is_inversion: True if operation includes spatial inversion.
    """
    # Convert to Cartesian coordinates
    # R_cart = lattice^T @ R_direct @ lattice^{-T}
    rot_cart = rotation_in_cart(rotation, lattice)

    # Compute determinant to check for inversion
    det = np.linalg.det(rot_cart)
    is_inversion = det < 0

    # Convert improper rotation to proper rotation
    rot_proper = rot_cart * np.sign(det)

    # Check for identity
    if np.allclose(rot_proper, np.eye(3), atol=EPS6):
        return np.array([0., 0., 1.]), 0.0, is_inversion

    # Check for inversion (now identity after sign flip)
    if is_inversion and np.allclose(rot_proper, np.eye(3), atol=EPS6):
        return np.array([0., 0., 1.]), 0.0, True

    # Compute trace and angle
    trace = np.trace(rot_proper) - 1
    if trace > 2:
        trace = 2
    elif trace < -2:
        trace = -2
    angle = np.arccos(trace / 2.0)

    # Handle 180-degree rotation specially
    if np.abs(angle - np.pi) < EPS6:
        # Find eigenvector with eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(rot_proper)
        idx = np.argmin(np.abs(eigvals - 1.0))
        axis = np.real(eigvecs[:, idx])
        angle = np.pi
    elif np.abs(angle) < EPS6:
        axis = np.array([0., 0., 1.])
        angle = 0.0
    else:
        # General case: axis from antisymmetric part
        axis = np.array([
            rot_proper[2, 1] - rot_proper[1, 2],
            rot_proper[0, 2] - rot_proper[2, 0],
            rot_proper[1, 0] - rot_proper[0, 1]
        ])
        axis = normalize_vector(axis)

        # Check sign of angle
        test_val = axis[1] * axis[0] * (1 - np.cos(angle)) + axis[2] * np.sin(angle)
        if np.abs(test_val - rot_proper[1, 0]) > 1e-5:
            angle = -angle

    # Ensure axis points in canonical direction
    if axis[2] < -EPS6:
        axis = -axis
        angle = -angle
    elif np.abs(axis[2]) < EPS6 and axis[0] < -EPS6:
        axis = -axis
        angle = -angle
    elif np.abs(axis[2]) < EPS6 and np.abs(axis[0]) < EPS6 and axis[1] < -EPS6:
        axis = -axis
        angle = -angle

    # Ensure angle in (-π, π]
    if angle < -np.pi + EPS6:
        angle += TwoPi

    return axis, angle, is_inversion

