import numpy as np
from numba import njit, prange

from wanntb.constant import Eta_4
from wanntb.utility import _get_ham_k2d, _matrix_add_by_index, _get_submatrix_by_index


@njit(nogil=True)
def _surface_GR(energy, n_dim, h0, t, mu=0.0, n_iter=25):
    """
    计算表面格林函数
    :param energy: 入射能量
    :param n_dim: 电极维度
    :param h0: 单层哈密顿矩阵
    :param t: 跃迁哈密顿矩阵
    :param mu: 化学势
    :param n_iter: 迭代次数
    :return: [n_dim, n_dim] complex
    """
    a0 = t
    b0 = np.ascontiguousarray(t.T.conjugate())
    e0 = h0 - np.eye(n_dim) * (1j * Eta_4 - mu)
    es = e0.copy()
    for i in range(n_iter):
        # if self.max_z == 1:  # 目前就做了最近邻，其他的先空着
        gR0 = np.linalg.inv(-e0 + np.eye(n_dim) * energy)
        es += a0.dot(gR0).dot(b0)
        e0 += a0.dot(gR0).dot(b0) + b0.dot(gR0).dot(a0)
        a1 = a0.dot(gR0).dot(a0)
        b1 = b0.dot(gR0).dot(b0)
        a0 = a1
        b0 = b1
    gR0 = np.linalg.inv(-es + np.eye(n_dim) * energy)
    return gR0


@njit(parallel=True, nogil=True)
def transmission_k2d_kpar(num_wann, ham_R, R_vec, efermi,
                          li_lc, li_rc, sRl, sRr, gm_l, gm_r, e_list,
                          kpts):
    n_e = e_list.shape[0]
    nkpts = kpts.shape[0]
    # 每个k点的透射系数数组
    trans_k = np.zeros((n_e, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        ham_k = _get_ham_k2d(num_wann, ham_R, R_vec, kpts[ik, :], efermi)
        ham_k2 = np.zeros((num_wann, num_wann), dtype=np.complex128)
        for ie in range(n_e):
            ham_k2[:, :] = ham_k.copy()
            _matrix_add_by_index(ham_k2, li_lc, li_lc, sRl[ie, :, :])
            _matrix_add_by_index(ham_k2, li_rc, li_rc, sRr[ie, :, :])
            # 总的推迟格林函数和超越格林函数
            gR = np.linalg.inv(np.eye(num_wann, dtype=np.complex128) * e_list[ie] - ham_k2)
            gA = gR.T.conjugate()
            gR_lr = _get_submatrix_by_index(gR, li_lc, li_rc)
            gA_rl = _get_submatrix_by_index(gA, li_rc, li_lc)
            tt = np.dot(np.dot(np.dot(gm_l[ie, :, :], gR_lr), gm_r[ie, :, :]), gA_rl)
            trans_k[ie, ik] = np.trace(tt).real

    return np.sum(trans_k, axis=1) / nkpts


@njit(parallel=True, nogil=True)
def get_self_energies_epar(l_h0, l_t, l_dim,
                           r_h0, r_t, r_dim,
                           n_e, e_list,
                           mu_l, mu_r, v_lc, v_rc, n_iter=25):
    n_lc = v_lc.shape[1]
    n_rc = v_rc.shape[1]
    sRl = np.zeros((n_e, n_lc, n_lc), dtype=np.complex128)
    sRr = np.zeros((n_e, n_rc, n_rc), dtype=np.complex128)
    gm_l = np.zeros((n_e, n_lc, n_lc), dtype=np.complex128)
    gm_r = np.zeros((n_e, n_rc, n_rc), dtype=np.complex128)
    for ie in prange(n_e):
        gsRl = _surface_GR(e_list[ie], l_dim, l_h0, l_t, mu_l, n_iter)
        gsRr = _surface_GR(e_list[ie], r_dim, r_h0, r_t, mu_r, n_iter)
        sRl[ie, :, :] = np.ascontiguousarray(v_lc.T.conjugate()).dot(gsRl).dot(v_lc)
        sRr[ie, :, :] = np.ascontiguousarray(v_rc.T.conjugate()).dot(gsRr).dot(v_rc)
        gm_l[ie, :, :] = (sRl[ie, :, :] - sRl[ie, :, :].T.conjugate()) * 1j
        gm_r[ie, :, :] = (sRr[ie, :, :] - sRr[ie, :, :].T.conjugate()) * 1j
    return sRl, sRr, gm_l, gm_r
