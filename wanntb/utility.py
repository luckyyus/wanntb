import numpy as np
from numba import njit, prange
from typing import Optional

from .constant import TwoPi, EPS4, EPS5, EPS6, S_, Berry_Task


def get_list_index(item, li):
    for index, value in enumerate(li):
        if value == item:
            return index
    return -1


@njit(nogil=True)
def _matrix_add_by_index(target, xlist, ylist, add):
    nx = xlist.shape[0]
    ny = ylist.shape[0]
    assert add.shape[0] == nx and add.shape[1] == ny, 'add.shape == (len(xlist), len(ylist))'
    for ix in range(nx):
        for iy in range(ny):
            target[xlist[ix], ylist[iy]] += add[ix, iy]
    pass


@njit(nogil=True)
def _get_submatrix_by_index(target, xlist, ylist):
    nx = int(len(xlist))
    ny = int(len(ylist))
    sub = np.zeros((nx, ny), dtype=np.complex128)
    for ix in range(nx):
        for iy in range(ny):
            sub[ix, iy] = target[xlist[ix], ylist[iy]]
    return sub


@njit(nogil=True)
def _get_ham_k2d(num_wann, ham_R, r_vec, k_2d, efermi):
    # ham_k = np.zeros((num_wann, num_wann), dtype=complex128)
    rdotk = r_vec[:, 0:2] @ k_2d * TwoPi
    phase_fac = np.cos(rdotk) + 1j * np.sin(rdotk)
    ham_k = phase_fac @ ham_R
    # for ir in range(n_rpts):
    #     # 只有Rz为0的H_R是需要考虑的
    #     # if R_vec[ir, 2] == 0:
    #     rdotk = TwoPi * np.dot(r_vec[ir, 0:2], k_2d)
    #     ham_k += ham_R[ir, :, :] * complex(np.cos(rdotk), np.sin(rdotk)) / n_degen[ir]
    return ham_k - np.eye(num_wann, dtype=np.complex128) * efermi


@njit(parallel=True, nogil=True)
def get_dos_e_kpar(num_wann, ham_R, R_vec, ef, n_e, e_list, nkpts, kpts):
    dos_k = np.zeros((n_e, nkpts), dtype=float)
    for ik in prange(nkpts):
        ham_k = _get_ham_k2d(num_wann, ham_R, R_vec, kpts[ik, :], ef)
        for ie in range(n_e):
            gR = np.linalg.inv(np.eye(num_wann, dtype=np.complex128) * (e_list[ie] + 1j * EPS4) - ham_k)
            dos_k[ie, ik] = np.trace(gR).imag
    return - np.sum(dos_k, axis=1) / np.pi / nkpts


@njit(nogil=True)
def hermiization_R(mat, R_vec):
    """
    对格点坐标下的算符矩阵厄米化
    """
    _mat = np.copy(mat)
    shape = _mat.shape
    if len(shape) == 3:
        nrpt = shape[0]
    else:
        nrpt = shape[0]
        ndim = shape[1]
    lhermi = np.zeros(nrpt, dtype=np.bool_)
    for ir in range(nrpt):
        if lhermi[ir]:
            continue
        for jr in range(ir-1, nrpt):
            dr = R_vec[ir] + R_vec[jr]

            if dr[0] == 0 and dr[1] == 0 and dr[2] == 0:
                if len(shape) == 3:
                    _real = (_mat[ir] + _mat[jr].T).real / 2
                    _imag = (_mat[ir] - _mat[jr].T).imag / 2
                    _mat[ir] = _real + 1j * _imag
                    _mat[jr] = (_real - 1j * _imag).T
                else:
                    for k in range(ndim):
                        _real = (_mat[ir, k] + _mat[jr, k].T).real / 2
                        _imag = (_mat[ir, k] - _mat[jr, k].T).imag / 2
                        _mat[ir, k] = _real + 1j * _imag
                        _mat[jr, k] = (_real - 1j * _imag).T
                lhermi[ir] = True
                lhermi[jr] = True
                break
    return _mat


@njit(nogil=True)
def unitary_trans(mat, uu, inverse=False):
    if inverse:
        return uu @ mat @ uu.conj().T
    else:
        return uu.conj().T @ mat @ uu

@njit(nogil=True)
def unitary_trans_sub(mat, uu1, uu2):
    return uu1.conj().T @ mat @ uu2


@njit(nogil=True)
def fourier_phase_R_to_k(R_vec, kpt):
    """
    计算从R表象到k表象的相因子, [n_Rpt]
    @param R_vec: [n_Rpt, 3]
    @param kpt: [3]
    """
    rdotk = R_vec @ kpt * TwoPi
    return np.cos(rdotk) + np.sin(rdotk) *1j


@njit(nogil=True)
def fourier_R_to_k(mat_R, R_cartT, phase_fac, iout=[0]):
    """
    把一个厄米矩阵从R空间转变到k空间
    @param mat_R: R空间的厄米矩阵，单位 eV
    @param R_vec_cart_T: R格点的xyz坐标，单位 angst.，数组排列转置
    @param phase_fac: 各个R的相因子
    @param iout: 输出选项tuple，0代表mat_k, 1~3代表xyz三个方向的dmat/dk
    """
    num_wann = mat_R.shape[1]
    output = np.zeros((4, num_wann, num_wann), dtype=np.complex128)
    # output[0] = np.sum(mat_R * phase_fac, axis=2)
    # if 1 in iout:
    #     output[1] = np.sum(mat_R * R_cartT[0] * phase_fac, axis=2) * 1j
    # if 2 in iout:
    #     output[2] = np.sum(mat_R * R_cartT[0] * phase_fac, axis=2) * 1j
    # if 3 in iout:
    #     output[3] = np.sum(mat_R * R_cartT[0] * phase_fac, axis=2) * 1j
    for i in range(num_wann):
        for j in range(num_wann):
            mat_Rij = mat_R[i, j]
            output[0, i, j] = np.sum(mat_Rij * phase_fac)
            if 1 in iout:
                output[1, i, j] = np.sum(mat_Rij * R_cartT[0] * phase_fac) * 1j
            if 2 in iout:
                output[2, i, j] = np.sum(mat_Rij * R_cartT[1] * phase_fac) * 1j
            if 3 in iout:
                output[3, i, j] = np.sum(mat_Rij * R_cartT[2] * phase_fac) * 1j
    return output


@njit(nogil=True)
def fourier_R_to_k_vec3(vec_R, phase_fac):
    num_wann = vec_R.shape[1]
    # oo_true = np.sum(vec_R * phase_fac, axis=3)
    oo_true = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for k in range(3):
        for i in range(num_wann):
            for j in range(num_wann):
                vec_Rkij = vec_R[k, i, j]
                oo_true[k, i, j] = np.sum(vec_Rkij * phase_fac)
    return oo_true


@njit(nogil=True)
def fourier_R_to_k_curl(vec_R, phase_fac, R_cartT):
    num_wann = vec_R.shape[1]
    oo_curl = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for i in range(num_wann):
        for j in range(num_wann):
            vec_ij_0 = vec_R[0, i, j]
            vec_ij_1 = vec_R[1, i, j]
            vec_ij_2 = vec_R[2, i, j]
            oo_curl[0, i, j] = np.sum((R_cartT[1] * vec_ij_2
                                       - R_cartT[2] * vec_ij_1) * phase_fac) * 1j
            oo_curl[1, i, j] = np.sum((R_cartT[2] * vec_ij_0
                                       - R_cartT[0] * vec_ij_2) * phase_fac) * 1j
            oo_curl[2, i, j] = np.sum((R_cartT[0] * vec_ij_1
                                       - R_cartT[1] * vec_ij_0) * phase_fac) * 1j
    return oo_curl


@njit(nogil=True)
def get_deltaU(ham_R, R_vec, R_cartT, num_wann, kpt, uu, q_frac, q, order=2, lbloch=False):
    # duu[idim, iproj, iband]
    duu = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    if order == 4:
        for i in range(3):
            h1, eig1, uu1 = _ham_k_system(ham_R, R_vec, R_cartT, kpt - 2 * q_frac[i])
            h2, eig2, uu2 = _ham_k_system(ham_R, R_vec, R_cartT, kpt - q_frac[i])
            h3, eig3, uu3 = _ham_k_system(ham_R, R_vec, R_cartT, kpt + q_frac[i])
            h4, eig4, uu4 = _ham_k_system(ham_R, R_vec, R_cartT, kpt + 2 * q_frac[i])
            duu[i] = (uu1 - 8 * uu2 + 8 * uu3 - uu4) / (12 * q)
    else:
        for i in range(3):
            h1, eig1, uu1 = _ham_k_system(ham_R, R_vec, R_cartT, kpt - q_frac[i])
            h2, eig2, uu2 = _ham_k_system(ham_R, R_vec, R_cartT, kpt + q_frac[i])
            duu[i] = (uu2 - uu1) / (2 * q)
    if lbloch: # in the bloch space (else in the WF space)
        for i in range(3):
            duu[i] = uu.conj().T @ duu[i]
    return duu


@njit(nogil=True)
def get_eig_da(eig, ham_da, uu, eig_diff=1e-4):
    num_wann = eig.shape[0]
    eig_da = np.zeros(num_wann, dtype=np.float64)
    ham_bar_da = unitary_trans(ham_da, uu)
    i = 0
    while i < num_wann:
        diff = eig[i + 1] - eig[i] if i + 1 < num_wann else 1.0
        if diff < eig_diff:
            degen_min = i
            degen_max = i + 1
            while degen_max + 1 < num_wann:
                diff = eig[degen_max + 1] - eig[degen_min]
                if diff < eig_diff:
                    degen_max += 1
                else:
                    break
            eig_da[degen_min:degen_max] \
                = np.linalg.eigvalsh(ham_bar_da[degen_min:degen_max,degen_min:degen_max])
            i = degen_max
        else:
            eig_da[i] = ham_bar_da[i, i].real
        i += 1
    return eig_da


@njit(nogil=True)
def _ham_k_da_system(ham_R, R_vec, R_cartT, kpt, direction):
    """
    计算一个k点的哈密顿量、本征值、征值随k的某个方向的导数和本征态。
    @param ham_R: R空间的紧束缚哈密顿量
    @param R_vec: R格点坐标
    @param R_cartT: R格点的xyz坐标，数组排列转置
    @param direction: 方向，123分别代表xyz
    @param kpt: k点坐标，倒格矢表象
    @return: k空间哈密顿量，本征值，本征值随k的导数，本征态
    """
    fac = fourier_phase_R_to_k(R_vec, kpt)
    out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[direction])
    ham_k, ham_k_da = out[0], out[direction]
    eig, uu = np.linalg.eigh(ham_k)
    eig_da = get_eig_da(eig, ham_k_da, uu)
    return ham_k, eig, eig_da, uu

@njit(nogil=True)
def _ham_k_system(ham_R, R_vec, R_cartT, kpt):
    """
    计算一个k点的哈密顿量、本征值和本征态。
    @param ham_R: R空间的紧束缚哈密顿量
    @param R_vec: R格点坐标
    @param R_cartT: R格点的xyz坐标，数组排列转置
    @param kpt: k点坐标，倒格矢表象
    @return: k空间哈密顿量、本征值、本征态
    """
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_k = fourier_R_to_k(ham_R, R_cartT, fac, iout=[5])[0]
    eig, uu = np.linalg.eigh(ham_k)
    return ham_k, eig, uu


@njit(parallel=True, nogil=True)
def get_eig_for_kpts_kpar(ham_R, R_vec, R_cartT, num_wann, kpts):
    nkpts = kpts.shape[0]
    eigs = np.zeros((nkpts, num_wann), dtype=float)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        fac = fourier_phase_R_to_k(R_vec, kpt)
        ham_k = fourier_R_to_k(ham_R, R_cartT, fac)[0]
        eig, uu = np.linalg.eigh(ham_k)
        eigs[ik, :] = eig
    return eigs


@njit('float64(float64, float64, float64)', nogil=True)
def A_n(eig_n, ef, eta):
    de = eig_n - ef
    return eta / (de * de + eta * eta / 4)


@njit(nogil=True)
def A_vec(eig, ef, eta):
    de = eig - ef
    return eta / (de * de + eta * eta / 4)


@njit(nogil=True)
def occ_fermi(eig, ef, eta):
    return 1.0 / (np.exp((eig - ef)/eta) + 1)


@njit(nogil=True)
def dos_fermi(eig, ef, eta):
    """dos for eigenvalues with Fermi-Dirac distribution:
    N(n) = - partial f_n/ partial e (e=e_f)
    """
    fac = (eig - ef) / eta
    return 1.0 / (np.exp(fac) + 1) / (1 + np.exp(-fac)) / eta

@njit(nogil=True)
def get_delta_E(eig, ef, eta):
    de = eig - ef
    return  eta / (de * de + eta * eta / 4) / TwoPi

@njit(nogil=True)
def spin_w(gamma, num_wann, udud_order=False):
    n_orbit = num_wann//2
    if udud_order:
        sw = np.kron(np.eye(n_orbit, dtype=np.complex128), S_[gamma])
    else:
        sw = np.kron(S_[gamma], np.eye(n_orbit, dtype=np.complex128))
    return sw


@njit(nogil=True)
def inv_e_d_r(eig, num_wann):
    """
    inv_e_d[m, n] = 1 / (e_n - e_m)
    """
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    for n_ in range(num_wann):
        for m_ in range(num_wann):
            if m_ == n_:
                continue
            e_d = eig[n_] - eig[m_]
            inv_e_d[m_, n_] = 1.0 / e_d if abs(e_d) > 1e-8 else 0.0
    return inv_e_d

@njit(nogil=True)
def inv_e_d_c(eig, num_wann, eta=1e-6):
    """
    inv_e_d[m, n] = 1 / (e_n - e_m + i eta)
    """
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.complex128)
    e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            # if m_ == n_:
            #     continue
            e_d[m_, n_] = eig[n_] - eig[m_]
            inv_e_d[m_, n_] = 1.0 / (e_d[m_, n_] + 1j * eta) # if abs(e_d) > 1e-8 else 0.0
    return inv_e_d, e_d


@njit(nogil=True)
def inv_e_d_2(eig, num_wann, eta=1e-6):
    """
    inv_e_d_2[m, n] = 1 / ((e_n - e_m)^2 + eta^2))
    """
    inv_e_d_2 = np.zeros((num_wann, num_wann), dtype=np.float64)
    e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            e_d[m_, n_] = eig[n_] - eig[m_]
            inv_e_d_2[m_, n_] = 1.0 / (e_d[m_, n_] * e_d[m_, n_] + eta * eta) # if abs(e_d) > 1e-8 else 0.0
    return inv_e_d_2


# @njit(parallel=True)
# def sz_n(uu, num_wann: int):
#     """
#     计算 < un | Sz | un > 无量纲
#     @param uu:
#     @param num_wann:
#     @return:
#     """
#     half = num_wann // 2
#     sz = np.zeros(num_wann, dtype=float)
#     for n_ in prange(num_wann):
#         up = np.dot(uu[0:half, n_].conj(), uu[0:half, n_]).real
#         dn = np.dot(uu[half:num_wann, n_].conj(), uu[half:num_wann, n_]).real
#         sz[n_] = up - dn
#     return sz

@njit
def guess(x):
    return 1.0 / (2.0 + np.exp(-x) + np.exp(+x))


@njit(parallel=True, nogil=True)
def get_carrier_kpar(ham_R, R_vec, R_vec_cart_T,
                     num_wann, direction, kpts, q_frac, q, ef, eta):
    nkpts = kpts.shape[0]
    list_o_k = np.zeros(nkpts, dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        ham_k, eig, eig_da, uu = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T,
                                                  num_wann, kpt, direction)
        # k + q
        ham_q, eig_q, eig_q_da, uu_q = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T,
                                                        num_wann, kpt + q_frac, direction)
        eig_dd_inv = (q / (eig_q_da - eig_da - 1j * eta/q)).real
        n_eig_ef = A_vec(eig, ef, eta)
        list_o_k[ik] = np.sum(n_eig_ef * eig_dd_inv * eig_da * eig_da)
    return np.sum(list_o_k) / (nkpts * TwoPi * TwoPi)


def get_itasks(tasks):
    _tasks = tasks.split('+')
    itasks = []
    for task in _tasks:
        if task in Berry_Task.keys():
            itasks.append(Berry_Task[task]['itask'])
        else:
            print('%s is not in Berry_Task' % task)
    itasks = np.sort(itasks)
    # print('itasks:', itasks)
    begin_idx = {}
    count = 0
    for it in itasks:
        begin_idx[it] = count
        count += 3
    return itasks, begin_idx, count


@njit(nogil=True)
def vectors_equal(v1: np.ndarray, v2: np.ndarray, tol: float = EPS5) -> bool:
    """Check if two vectors are equal within tolerance.

    Args:
        v1: First vector.
        v2: Second vector.
        tol: Tolerance for comparison.

    Returns:
        True if vectors are equal within tolerance.
    """
    return np.all(np.abs(v1 - v2) < tol)

@njit(nogil=True)
def vector_distance(v1: np.ndarray, v2: np.ndarray,
                   lattice: Optional[np.ndarray] = None) -> float:
    """Compute distance between two points.

    Args:
        v1: First point (direct coordinates if lattice given).
        v2: Second point.
        lattice: Lattice vectors [3, 3] for Cartesian conversion.

    Returns:
        Distance between points.
    """
    diff = v1 - v2
    if lattice is not None:
        diff = diff @ lattice  # Convert to Cartesian
    return np.linalg.norm(diff)


@njit(nogil=True)
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length.

    Args:
        v: Input vector.

    Returns:
        Normalized vector, or zero vector if input is zero.
    """
    norm = np.linalg.norm(v)
    if norm < EPS6:
        return v
    return v / norm


@njit(nogil=True)
def kpoints_equivalent(k1: np.ndarray, k2: np.ndarray, tol: float = EPS5) -> bool:
    """Check if two k-points are equivalent (differ by reciprocal lattice vector).

    Args:
        k1: First k-point in reciprocal coordinates.
        k2: Second k-point.
        tol: Tolerance for fractional part comparison.

    Returns:
        True if k-points are equivalent.
    """
    diff = k1 - k2
    frac_diff = diff - np.round(diff)
    return np.all(np.abs(frac_diff) < tol)


@njit(cache=True, nogil=True)
def find_R_vec(rv: np.ndarray, rvec_pool: np.ndarray) -> int:
    """Binary search for R-vector (integer) in a sorted pool (Lexicographical)."""
    low = 0
    high = rvec_pool.shape[0] - 1
    while low <= high:
        mid = (low + high) // 2
        diff = rv - rvec_pool[mid]

        # Lexicographical comparison with tolerance
        is_smaller = False
        is_equal = True

        for i in range(3): # the first index is the major one
            if diff[i] < 0:
                is_smaller = True
                is_equal = False
                break
            elif diff[i] > 0:
                is_equal = False
                break

        if is_equal:
            return mid
        elif is_smaller:
            high = mid - 1
        else:
            low = mid + 1
    return -1