import numpy as np
from numba import njit, prange
from .utility import A_n, _ham_k_da_system


@njit(nogil=True)
def _get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q, num_wann: int, ef: float, eta: float, q: float):
    half = num_wann // 2
    alpha_k = np.zeros(num_wann, dtype=np.float64)
    alpha_qvd_k = np.zeros(num_wann, dtype=np.float64)
    qvs_k = np.zeros(num_wann, dtype=np.float64)
    alpha_k_n = np.zeros(num_wann, dtype=np.float64)
    alpha_qvd_k_n = np.zeros(num_wann, dtype=np.float64)
    qvs_k_n = np.zeros(num_wann, dtype=np.float64)
    for n_ in prange(num_wann):
        if abs(eig[n_] - ef) > eta * 16.0:
            continue
        # un_up = np.ascontiguousarray(uu[0:half, n_])
        un_dn = np.ascontiguousarray(uu[half:num_wann, n_])
        # sz = np.dot(un_up.conj(), un_up).real - np.dot(un_dn.conj(), un_dn).real
        An = A_n(eig[n_], ef, eta)
        alpha_k_n[:] = 0.0
        alpha_qvd_k_n[:] = 0.0
        qvs_k_n[:] = 0.0
        for m_ in prange(num_wann):
            um_up = np.ascontiguousarray(uu_q[0:half, m_])
            sp2_q = abs(np.dot(um_up.conj(), un_dn * e_s))**2
            Am = A_n(eig_q[m_], ef, eta)
            # ww in units eV^-2
            ww = An * Am
            # alpha_k in units 1
            alpha_k_n[m_] = sp2_q * ww
            # qvd in units eV angst.
            alpha_qvd_k_n[m_] = sp2_q * ww * (eig_q_da[m_] - eig_da[n_])
            # qvs in units eV angst.
            if n_ != m_:
                d_eig_mn = eig_q[m_] - eig[n_]
                if abs(d_eig_mn) < 1E-7:
                    qvs_k_n[m_] = - sp2_q * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                             / (d_eig_mn - 1E-7j)).real
                else:
                    qvs_k_n[m_] = - sp2_q * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                         / d_eig_mn)
            else:
                # qvs_k_n[m_] = eta * q * ww * eig_da[n_] * eig_da[n_] * sz
                # qvs_k_n[m_] = - sp2_q * (eig_q_da[n_] - eig_da[n_]) * An / (eig_q[n_] - eig[n_])
                #             = - sp2_q * (eig_q_da[n_] - eig_da[n_]) * An / ( q * eig_da[n_])
                #             = - sp2_q * (1 + q * eig_dda[n_]/eig_da[n_] -1) * An / q
                #             = - sp2_q * (eig_dda[n_]/eig_da[n_]) * An
                if abs(eig_da[n_] * q) < 1E-7:
                    qvs_k_n[m_] = - sp2_q * ((eig_q_da[n_] * Am - eig_da[n_] * An)
                                             / ( q * eig_da[n_] - 1E-7j)).real
                else:
                    qvs_k_n[m_] = - sp2_q * (eig_q_da[n_] / eig_da[n_] * Am - An) / q
        alpha_k[n_] = np.sum(alpha_k_n)
        alpha_qvd_k[n_] = np.sum(alpha_qvd_k_n)
        qvs_k[n_] = np.sum(qvs_k_n)
    return np.sum(alpha_k), np.sum(alpha_qvd_k), np.sum(qvs_k)


@njit(nogil=True)
def _get_alpha_beta_inter_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q, num_wann: int, ef: float, eta: float):
    half = num_wann // 2
    alpha_k = np.zeros(num_wann, dtype=np.float64)
    alpha_qvd_k = np.zeros(num_wann, dtype=np.float64)
    qvs_k = np.zeros(num_wann, dtype=np.float64)
    alpha_k_n = np.zeros(num_wann, dtype=np.float64)
    alpha_qvd_k_n = np.zeros(num_wann, dtype=np.float64)
    qvs_k_n = np.zeros(num_wann, dtype=np.float64)
    for n_ in prange(num_wann):
        if abs(eig[n_] - ef) > eta * 16.0:
            continue
        # un_up = np.ascontiguousarray(uu[0:half, n_])
        un_dn = np.ascontiguousarray(uu[half:num_wann, n_])
        # sz = np.dot(un_up.conj(), un_up).real - np.dot(un_dn.conj(), un_dn).real
        An = A_n(eig[n_], ef, eta)
        alpha_k_n[:] = 0.0
        alpha_qvd_k_n[:] = 0.0
        qvs_k_n[:] = 0.0
        for m_ in prange(num_wann):
            if n_ == m_:
                continue
            um_up = np.ascontiguousarray(uu_q[0:half, m_])
            sp2_q = abs(np.dot(um_up.conj(), un_dn * e_s)) ** 2
            Am = A_n(eig_q[m_], ef, eta)
            # ww in units eV^-2
            ww = An * Am
            # alpha_k in units 1
            alpha_k_n[m_] = sp2_q * ww
            # qvd in units eV angst.
            alpha_qvd_k_n[m_] = sp2_q * ww * (eig_q_da[m_] - eig_da[n_])
            # qvs in units eV angst.
            d_eig_mn = eig_q[m_] - eig[n_]
            if abs(d_eig_mn) < 1E-7:
                qvs_k_n[m_] = - sp2_q * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                         / (d_eig_mn - 1E-7j)).real
            else:
                qvs_k_n[m_] = - sp2_q * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                         / d_eig_mn)

        alpha_k[n_] = np.sum(alpha_k_n)
        alpha_qvd_k[n_] = np.sum(alpha_qvd_k_n)
        qvs_k[n_] = np.sum(qvs_k_n)
    return np.sum(alpha_k), np.sum(alpha_qvd_k), np.sum(qvs_k)


@njit(parallel=True, nogil=True)
def get_alpha_beta_kpar(ham_R, R_vec, R_vec_cart_T,
                        num_wann, direction, e_s, kpts, q_frac, q, ef, eta, adpt_kpts, adpt_qvs=20):
    nkpts = kpts.shape[0]
    nadpt = adpt_kpts.shape[0]
    ladpt = True if nadpt > 1 else False
    o_k = np.zeros((nkpts, 3), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        ham_k, eig, eig_da, uu = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, kpt, direction)
        # k + q
        ham_q, eig_q, eig_q_da, uu_q = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, kpt + q_frac, direction)
        o_k[ik, :] = _get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q,
                                       num_wann, ef, eta, q)
        if ladpt and np.abs(o_k[ik, 2]) > adpt_qvs:
            _adpt_kpts = kpt + adpt_kpts
            o_adpt = np.zeros((nadpt, 3), dtype=np.float64)
            for iik in prange(nadpt):
                _kpt = _adpt_kpts[iik]
                ham_k, eig, eig_da, uu = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, _kpt, direction)
                # k + q
                ham_q, eig_q, eig_q_da, uu_q = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, _kpt + q_frac, direction)
                o_adpt[iik, :] = _get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q,
                                               num_wann, ef, eta, q)
            o_k[ik, :] = np.sum(o_adpt, axis=0) / nadpt
    return np.sum(o_k, axis=0) / nkpts


@njit(parallel=True, nogil=True)
def get_alpha_beta_efs_kpar(ham_R, R_vec, R_vec_cart_T,
                        num_wann, direction, e_s, kpts, q_frac, q, efs, eta, adpt_kpts, adpt_qvs=20):
    n_ef = efs.shape[0]
    nkpts = kpts.shape[0]
    nadpt = adpt_kpts.shape[0]
    ladpt = True if nadpt > 1 else False
    o_ek = np.zeros((nkpts, n_ef, 3), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        ham_k, eig, eig_da, uu = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, kpt, direction)
        # k + q
        ham_q, eig_q, eig_q_da, uu_q = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, kpt + q_frac, direction)
        _ladpt_k = False
        for ie in range(n_ef):
            ef = efs[ie]
            o_ek[ik, ie, :] = _get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q,
                                       num_wann, ef, eta, q)
            if abs(o_ek[ik, ie, 2]) > adpt_qvs:
                _ladpt_k = True
        if ladpt and _ladpt_k:
            _adpt_kpts = kpt + adpt_kpts
            o_adpt = np.zeros((nadpt, n_ef, 3), dtype=np.float64)
            for iik in prange(nadpt):
                _kpt = _adpt_kpts[iik]
                ham_k, eig, eig_da, uu = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, _kpt, direction)
                # k + q
                ham_q, eig_q, eig_q_da, uu_q = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, _kpt + q_frac, direction)
                for ie in range(n_ef):
                    ef = efs[ie]
                    o_adpt[iik, ie, :] = _get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q,
                                               num_wann, ef, eta, q)
            o_ek[ik, :, :] = np.sum(o_adpt, axis=0) / nadpt
    return np.sum(o_ek, axis=0) / nkpts


@njit(parallel=True, nogil=True)
def get_alpha_beta_kpar_kpath(ham_R, R_vec, R_vec_cart_T,
                              num_wann, direction, e_s, kpts, q_frac, q, ef, eta):
    nkpts = kpts.shape[0]
    o_k = np.zeros((nkpts, 6), dtype=float)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        ham_k, eig, eig_da, uu = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, kpt, direction)
        # k + q
        ham_q, eig_q, eig_q_da, uu_q = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, kpt + q_frac, direction)
        o_k[ik, 0:3] = _get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q,
                                         num_wann, ef, eta, q)
        o_k[ik, 3:6] = _get_alpha_beta_inter_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q,
                                               num_wann, ef, eta)
    return o_k