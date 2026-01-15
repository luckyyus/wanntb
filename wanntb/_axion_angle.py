import numpy as np
from numba import njit
from ._berry import _get_omega_gmat


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


