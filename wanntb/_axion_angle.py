import numpy as np
from numba import njit, prange
from ._berry import _berry_Ah_k, _get_omega_gmat, I_A, I_B
from .utility import occ_fermi

@njit(nogil=True)
def _calc_axion_kernel_k(Ah_k, num_wann, f):
    """
    计算单个 k 点的总能带轴子核 L(k)
    优化点：通过 np.ascontiguousarray 消除性能警告
    公式: Tr[ \epsilon^{ijk} (A_i \partial_j A_k - i * (2/3) * A_i A_j A_k) ]
    """
    # 1. 提取berry connection 矩阵并确保内存连续
    Ax = np.ascontiguousarray(Ah_k[0, :num_wann, :num_wann])
    Ay = np.ascontiguousarray(Ah_k[1, :num_wann, :num_wann])
    Az = np.ascontiguousarray(Ah_k[2, :num_wann, :num_wann])

    # 2. 计算非阿贝尔 Berry 曲率矩阵
    omega_mat = _get_omega_gmat(Ah_k, Ah_k, f, num_wann)

    # 3. 计算第一项: Tr[ sum(A_i * Omega_i) ]
    # term1 = 0.0j
    # for i in range(3):
    #     # 提取曲率分量并确保内存连续
    #     Om_i = np.ascontiguousarray(omega_mat[i, :num_wann, :num_wann])
    #     if i == 0:
    #         term1 += np.trace(Ax @ Om_i)
    #     elif i == 1:
    #         term1 += np.trace(Ay @ Om_i)
    #     else:
    #         term1 += np.trace(Az @ Om_i)

    term1 = 0.0j
    for i in range(3):
        A_comp = np.ascontiguousarray(Ah_k[i, :num_wann, :num_wann])
        Om_comp = np.ascontiguousarray(omega_mat[i, :num_wann, :num_wann])
        term1 += np.trace(A_comp @ Om_comp)

    # # 4. 计算第二项: i * Tr[ Ax * (Ay * Az - Az * Ay) ]
    comm_yz = Ay @ Az - Az @ Ay
    term2 = 1.0j * np.trace(Ax @ comm_yz)

    return term1 + term2


@njit(parallel=True, nogil=True)
def axion_fermi(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpts, efs, eta, ss_R_in=None):
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    results_ef = np.zeros(n_ef, dtype=np.complex128)
    itasks = np.array([0], dtype=np.int32)

    # 解决 None 类型推断问题
    if ss_R_in is None:
        ss_R = np.zeros((1, 1, 1, 1), dtype=np.complex128)
    else:
        ss_R = ss_R_in

    for ik in prange(nkpts):
        kpt = kpts[ik]
        eig, _, Ah_bk, _ = _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT,
                                       num_wann, eta, kpt, 2, ss_R, None)

        for i in range(n_ef):
            ef = efs[i]
            f = occ_fermi(eig, ef, eta)
            kernel = _calc_axion_kernel_k(Ah_bk, num_wann, f)
            results_ef[i] += kernel

    # 归一化：结果实部除以 (-4 * pi * nkpts)
    return results_ef.real / (-4.0 * np.pi * nkpts)