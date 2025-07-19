import numpy as np
from numba import njit, prange
from .constant import TwoPi, Hbar_, Mu_B_, Berry_Task
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi, \
    fourier_R_to_k_curl, unitary_trans_sub, _ham_k_system, get_inv_e_d

I_A = np.array([1, 2, 0], dtype=np.int32)
I_B = np.array([2, 0, 1], dtype=np.int32)

# @njit(nogil=True)
# def get_berry_curvature_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, eta):
#     fac = fourier_phase_R_to_k(R_vec, kpt)
#     ham_out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[1, 2, 3])
#     A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
#     eig, uu = np.linalg.eigh(ham_out[0])
#     e_d = np.zeros((num_wann,num_wann), dtype=np.float64)
#     inv_e_d = np.zeros((num_wann,num_wann), dtype=np.float64)
#     for m_ in range(num_wann):
#         for n_ in range(num_wann):
#             if m_ == n_:
#                 continue
#             e_d[m_, n_] = eig[m_] - eig[n_]
#             inv_e_d[m_, n_] = -1.0 / (e_d[m_, n_]*e_d[m_, n_] + eta * eta)
#             # inv_e_d[m_, n_] = 1.0 / (eig[m_] - eig[n_]) if abs(eig[m_] - eig[n_]) > 1e-7 else 0.0
#     Ah_k = np.zeros((3,num_wann,num_wann), dtype=np.complex128)
#     # A_bar^W_a[3, num_wann, num_wann] in units angst.
#
#     for i in range(3):
#         # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
#         # A^H_a = A_bar^H_a + i D^H_a = i<psi_m| del_a psi_n>
#         Ah_k[i] = 1j * e_d * unitary_trans(A_bar_k[i], uu) - unitary_trans(ham_out[i+1], uu)
#     omega_k = np.zeros((3, num_wann), dtype=np.float64)
#     for n_ in range(num_wann):
#         omega_k[0, n_] = np.sum((Ah_k[1, n_, :] * Ah_k[2, :, n_] * inv_e_d[n_, :]).imag)
#         omega_k[1, n_] = np.sum((Ah_k[2, n_, :] * Ah_k[0, :, n_] * inv_e_d[n_, :]).imag)
#         omega_k[2, n_] = np.sum((Ah_k[0, n_, :] * Ah_k[1, :, n_] * inv_e_d[n_, :]).imag)
#     omega_k *= -2.0
#     return omega_k, eig


@njit(nogil=True)
def _get_Ah_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d = get_inv_e_d(eig, num_wann)
    Ah_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    for i in range(3):
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_mna / (e_n - e_m)}
        # A^H_a = A_bar^H_a + i D^H_a = i<psi_m| del_a psi_n>
        Ah_k[i] = unitary_trans(A_bar_k[i], uu) + 1j * unitary_trans(ham_out[i + 1], uu) * inv_e_d
    return Ah_k, eig, uu

@njit(nogil=True)
def _get_Ah_ab_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, subwf):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d = get_inv_e_d(eig, num_wann)
    Ah_ak = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Ah_bk = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    for i in range(3):
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_mna / (e_n - e_m)}
        # A^H_a = A_bar^H_a + i D^H_a = i<psi_m| del_a psi_n>
        Ah_ak[i] = (unitary_trans_sub(A_bar_k[i, subwf, :], uu[subwf, :], uu)
                    + 1j * unitary_trans_sub(ham_out[i + 1, subwf, :], uu[subwf, :], uu) * inv_e_d)
        Ah_ak[i] = (Ah_ak[i] + Ah_ak[i].T.conj()) * 0.5
        Ah_bk[i] = unitary_trans(A_bar_k[i], uu) + 1j * unitary_trans(ham_out[i + 1], uu) * inv_e_d
    return Ah_ak, Ah_bk, eig, uu

@njit(nogil=True)
def _get_js_Ahb_k(ham_R, r_mat_R, ss_R, R_vec, R_cartT,
                           num_wann, kpt, xyz, subwf):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    sw = fourier_R_to_k_vec3(ss_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d = get_inv_e_d(eig, num_wann)
    js_a = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Ah_a = unitary_trans(A_bar_k[I_A[xyz]], uu) + 1j * unitary_trans(ham_out[I_A[xyz] + 1], uu) * inv_e_d
    Ah_b = unitary_trans(A_bar_k[I_B[xyz]], uu) + 1j * unitary_trans(ham_out[I_B[xyz] + 1], uu) * inv_e_d
    for i in range(3):
        mat_S = unitary_trans(sw[i], uu) if subwf is None else unitary_trans_sub(sw[i, subwf, :], uu[subwf, :], uu)
        mat_B = mat_S @ Ah_a
        js_a[i] = (mat_B + mat_B.T.conj()) * 0.5
    return js_a, Ah_b, eig

@njit(nogil=True)
def _get_vh_jsd_inv2_eig_k(ham_R, r_mat_R, ss_R, R_vec, R_cartT,
                           num_wann, kpt, eta, xyz, subwf):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    sw = fourier_R_to_k_vec3(ss_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    # inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    inv2 = np.zeros((num_wann, num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ != n_:
                e_d[m_, n_] = eig[m_] - eig[n_]
                # inv_e_d[m_, n_] = - 1.0 / e_d[m_, n_] if abs(e_d[m_, n_]) > 1e-8 else 0.0
                inv2[n_, m_] = 1.0 / (e_d[m_, n_] * e_d[m_, n_] + eta * eta)
    js_a = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    # Dh_a = ham_a * inv_e_d
    vh_a = 1j * e_d * unitary_trans(A_bar_k[I_A[xyz]], uu) + unitary_trans(ham_out[I_A[xyz] + 1], uu)
    vh_b = 1j * e_d * unitary_trans(A_bar_k[I_B[xyz]], uu) + unitary_trans(ham_out[I_B[xyz] + 1], uu)
    for i in range(3):
        mat_S = unitary_trans(sw[i], uu) if subwf is None else unitary_trans_sub(sw[i, subwf, :], uu[subwf, :], uu)
        mat_B = mat_S @ vh_a
        js_a[i] = (mat_B + mat_B.T.conj()) * 0.5
    return vh_b, js_a, inv2, eig


@njit(nogil=True)
def _get_Ah_bar_Dh_eig_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d = get_inv_e_d(eig, num_wann)
    Ah_bar_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Dh_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for i in range(3):
        Ah_bar_k[i] = unitary_trans(A_bar_k[i], uu)
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_mna / (e_n - e_m)}
        Dh_k[i] = unitary_trans(ham_out[i + 1], uu) * inv_e_d
    return Ah_bar_k, Dh_k, eig


@njit(nogil=True)
def _get_Aw_bar_jw_eig_uu_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    ow_k = fourier_R_to_k_curl(r_mat_R, fac, R_cartT)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d = get_inv_e_d(eig, num_wann)
    jw_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for i in range(3):
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        jw_k[i] = unitary_trans(unitary_trans(ham_out[i + 1], uu) * inv_e_d * 1j, uu, inverse=True)
    return A_bar_k, ow_k, jw_k, eig, uu


@njit(nogil=True)
def _get_f_omega(Ah_k, f, num_wann):
    fo_k = np.zeros((3, num_wann), dtype=np.float64)
    g = 1.0 - f
    for i in range(3):
        for n_ in range(num_wann):
            fo_k[i, n_] = np.sum(g * (Ah_k[I_A[i], n_, :] * Ah_k[I_B[i], :, n_]).imag)
    fo_k *= -2.0 * f
    return fo_k

@njit(nogil=True)
def _get_f_omega_sub(Ah_ak, Ah_bk, f, num_wann):
    fo_k = np.zeros((3, num_wann), dtype=np.float64)
    g = 1.0 - f
    for i in range(3):
        for n_ in range(num_wann):
            fo_k[i, n_] = np.sum(g * (Ah_ak[I_A[i], n_, :] * Ah_bk[I_B[i], :, n_]).imag)
    fo_k *= -2.0 * f
    return fo_k

@njit(nogil=True)
def _get_f_omega_mat(Ah_ak, Ah_bk, f, num_wann):
    fo_k = np.zeros((3, num_wann, num_wann), dtype=np.float64)
    g = 1.0 - f
    for i in range(3):
        for m_ in range(num_wann):
            for n_ in range(num_wann):
                fo_k[i, m_, n_] = np.sum(g * (Ah_ak[I_A[i], m_, :] * Ah_bk[I_B[i], :, n_]).imag)
    fo_k *= -2.0 * f #to 陈宝风：你检查一下这样得到的矩阵的厄米性，如果不行就这个f不乘
    return fo_k

@njit(nogil=True)
def _get_f_spin_omega(vh_b, js2, inv2, f, num_wann):
    fso_k = np.zeros(   (3, num_wann), dtype=np.float64)
    g = 1.0 - f
    for i in range(3):
        for n_ in range(num_wann):
            fso_k[i, n_] = np.sum((g * js2[i, n_, :] * vh_b[:, n_] * inv2[n_, :]).imag)
    fso_k *= -2.0 * f
    return fso_k

@njit(nogil=True)
def _get_f_spins_omega(js_a, Ah_b, f, num_wann):
    fso_k = np.zeros((3, num_wann), dtype=np.float64)
    g = 1.0 - f
    for i in range(3):
        for n_ in range(num_wann):
            fso_k[i, n_] = np.sum(g * (js_a[i, n_, :] * Ah_b[:, n_]).imag)
    fso_k *= -2.0 * f
    return fso_k

@njit(nogil=True)
def _get_f_omega_new(Ah_bar_k, Dh_k, f, num_wann):
    g = 1.0 - f
    # Diag[Im(A_bar^H_alpha.A_bar^H_beta)]
    o_bar = np.zeros((3, num_wann), dtype=np.float64)
    # Diag[Re(D^H_alpha.A_bar^H_beta - D^H_beta.A_bar^H_alpha)]
    o_i = np.zeros((3, num_wann), dtype=np.float64)
    # Diag[Im(D^H_alpha.D^H_beta)]
    o_d = np.zeros((3, num_wann), dtype=np.float64)
    for i in range(3):
        for n_ in range(num_wann):
            o_bar[i, n_] = np.sum(g * (Ah_bar_k[I_A[i], n_, :] * Ah_bar_k[I_B[i], :, n_]).imag)
            o_i[i, n_] = np.sum(g *
                                ((Dh_k[I_A[i], n_, :] * Ah_bar_k[I_B[i], :, n_]).real
                                 - (Dh_k[I_B[i], n_, :] * Ah_bar_k[I_A[i], :, n_]).real))
            o_d[i, n_] = np.sum(g * (Dh_k[I_A[i], n_, :] * Dh_k[I_B[i], :, n_]).imag)
    fo_k = (o_bar + o_i - o_d) * f * -2.0
    return fo_k


@njit(nogil=True)
def _get_f_omega_w(A_bar_k, ow_k, jw_k, f, uu, num_wann):
    fw = unitary_trans(np.asarray(np.diag(f), dtype=np.complex128), uu, inverse=True)
    gw = np.eye(num_wann, dtype=np.complex128) - fw
    fo_k = np.zeros((3, num_wann), dtype=np.float64)
    for i in range(3):
        # Re(f.A_bar_alpha.A_bar_beta)
        o_bar = fw @ ow_k[i]
        o_d = fw @ A_bar_k[I_A[i]] @ gw @ jw_k[I_B[i]]
        o_d += fw @ jw_k[I_A[i]] @ gw @ A_bar_k[I_B[i]]
        o_d += fw @ jw_k[I_A[i]] @ gw @ jw_k[I_B[i]]
        fo_k_a = o_bar + o_d * 2.0 * 1j
        fo_k[i] = np.diag(unitary_trans(fo_k_a, uu)).real
    return fo_k

@njit(nogil=True)
def _get_f_omega_interpolation(f, duu, num_wann):
    g = 1.0 - f
    fo_k = np.zeros((3, num_wann), dtype=np.float64)
    for i in range(3):
        for n_ in range(num_wann):
            fo_k[i, n_] = np.sum(g * (duu[I_A[i], :, n_].conj() * duu[I_B[i], :, n_]).imag)
    fo_k *= -2.0 * f
    return fo_k

@njit(nogil=True)
def _get_berrycurv_f_eig_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, ef, eta, mode, subwf, q_frac=None, q=1e-6):
    if mode == 1:
        Ah_bar_k, Dh_k, eig = _get_Ah_bar_Dh_eig_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt)
        f = occ_fermi(eig, ef, eta)
        of_k = _get_f_omega_new(Ah_bar_k, Dh_k, f, num_wann)
    elif mode == 2:
        A_bar_k, ow_k, jw_k, eig, uu = _get_Aw_bar_jw_eig_uu_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt)
        f = occ_fermi(eig, ef, eta)
        of_k = _get_f_omega_w(A_bar_k, ow_k, jw_k, f, uu, num_wann)
    elif mode == 3:
        ham_k, eig, uu = _ham_k_system(ham_R, R_vec, R_cartT, kpt)
        duu = get_deltaU_h(ham_R, R_vec, R_cartT, num_wann, kpt, uu, q_frac, q, order=2)
        f = occ_fermi(eig, ef, eta)
        of_k = _get_f_omega_interpolation(f, duu, num_wann)
    else:
        if subwf is None:
            Ah_k, eig, uu = _get_Ah_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt)
            f = occ_fermi(eig, ef, eta)
            of_k = _get_f_omega(Ah_k, f, num_wann)
        else:
            Ah_ak, Ah_bk, eig, uu = _get_Ah_ab_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, subwf)
            f = occ_fermi(eig, ef, eta)
            of_k = _get_f_omega_sub(Ah_ak, Ah_bk, f, num_wann)
    return of_k, eig


@njit(nogil=True)
def _get_berrycurv_f_efs_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, efs, eta, mode, subwf):
    n_ef = efs.shape[0]
    if mode == 1:
        Ah_bar_k, Dh_k, eig = _get_Ah_bar_Dh_eig_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt)
    else:
        if subwf is None:
            Ah_k, eig, uu = _get_Ah_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt)
        else:
            Ah_ak, Ah_bk, eig, uu = _get_Ah_ab_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, subwf)
    ofg_k = np.zeros((n_ef, 3, num_wann), dtype=np.float64)
    for i in range(n_ef):
        ef = efs[i]
        f = occ_fermi(eig, ef, eta)
        if mode == 1:
            ofg_k[i, :, :] = _get_f_omega_new(Ah_bar_k, Dh_k, f, num_wann)
        else:
            if subwf is None:
                ofg_k[i, :, :] = _get_f_omega(Ah_k, f, num_wann)
            else:
                ofg_k[i, :, :] = _get_f_omega_sub(Ah_ak, Ah_bk, f, num_wann)
    return np.sum(ofg_k, axis=2)


@njit(nogil=True)
def _get_shc_f_efs_k(ham_R, r_mat_R, R_vec, ss_R, R_cartT,
                     num_wann, kpt, efs, eta, xyz, subwf):
    n_ef = efs.shape[0]
    v_b, jsd, inv2, eig = _get_vh_jsd_inv2_eig_k(ham_R, r_mat_R, ss_R, R_vec, R_cartT, num_wann, kpt,
                                                 eta, xyz, subwf)
    ofg_k = np.zeros((n_ef, 3, num_wann), dtype=np.float64)
    for i in range(n_ef):
        ef = efs[i]
        f = occ_fermi(eig, ef, eta)
        ofg_k[i, :, :] = _get_f_spin_omega(v_b, jsd, inv2, f, num_wann)
    return np.sum(ofg_k, axis=2)


@njit(parallel=True, nogil=True)
def get_berrycurv_kpar_kpath(ham_R, r_mat_R, R_vec, R_cartT,
                             num_wann, kpts, ef, eta, mode, subwf=None, q_frac=None, q=1e-6):
    nkpts = kpts.shape[0]
    list_o_k = np.zeros((3, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        oo, eig = _get_berrycurv_f_eig_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, ef, eta,
                                         mode=mode, subwf=subwf, q_frac=q_frac, q=q)
        list_o_k[:, ik] = np.sum(oo, axis=1)
    return list_o_k.T


@njit(parallel=True, nogil=True)
def get_ahc_kpar_fermi(ham_R, r_mat_R, R_vec, R_cartT,
                       num_wann, kpts, efs, eta, mode=0, subwf=None):
    fac = - TwoPi
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    list_o_ef_k = np.zeros((n_ef, 3, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        list_o_ef_k[:, :, ik] = _get_berrycurv_f_efs_k(
                ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, efs, eta, mode, subwf)
    return np.sum(list_o_ef_k, axis=2) / nkpts * fac


@njit(parallel=True, nogil=True)
def get_shc_kpar_fermi(ham_R, r_mat_R, R_vec, ss_R, R_cartT,
                       num_wann, kpts, efs, eta, xyz, subwf=None):
    fac = - TwoPi
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    # sw = spin_w(gamma, num_wann, udud_order)
    list_o_ef_k = np.zeros((n_ef, 3, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        list_o_ef_k[:, :, ik] = _get_shc_f_efs_k(ham_R, r_mat_R, R_vec, ss_R, R_cartT,
                                              num_wann, kpt, efs, eta, xyz, subwf)
    return np.sum(list_o_ef_k, axis=2) / nkpts * fac


@njit(parallel=True, nogil=True)
def get_morb_berry_kpar_kpath(ham_R, r_mat_R, R_vec, R_cartT,
                              num_wann, kpts, ef, eta, xyz):
    fac = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    morb_k = np.zeros(nkpts, dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        oo, eig = _get_berrycurv_f_eig_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, ef, eta, mode=0)
        morb_k[ik] = np.sum(oo[xyz, :] * (ef - eig))
    return morb_k * fac


@njit(parallel=True, nogil=True)
def get_morb_berry_kpar(ham_R, r_mat_R, R_vec, R_cartT,
                        num_wann, kpts, ef, eta, xyz):
    fac = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    morb_k = np.zeros(nkpts, dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        oo, eig = _get_berrycurv_f_eig_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, ef, eta, mode=0)
        morb_k[ik] = np.sum(oo[xyz, :] * (ef - eig))
    return np.sum(morb_k) * fac / nkpts


@njit(nogil=True)
def get_deltaU_w(ham_R, R_vec, R_cartT, num_wann, kpt, q_frac, q, order=2):
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
    return duu


@njit(nogil=True)
def get_deltaU_h(ham_R, R_vec, R_cartT, num_wann, kpt, uu, q_frac, q, order=2):
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
    for i in range(3):
        duu[i] = uu.conj().T @ duu[i]
    return duu

@njit(nogil=True)
def _get_morb1_morb2_k(eig, num_wann, ef, duu, fo_k, f, xyz):
    g = 1.0 - f
    # f.(ef - eig).Omega
    morb2 = fo_k[xyz] * (ef - eig)
    # -0.5.f.eig.Omega
    morb1_2 = fo_k[xyz] * 0.5 * eig
    # f.<duu_n_a|H|duu_n_b> = f.<duu_n_a|uu_m>e_n<uu_m|duu_n_b>
    morb1_1 = np.zeros(num_wann, dtype=np.float64)
    for n_ in range(num_wann):
        morb1_1[n_] = np.sum((duu[I_A[xyz], :, n_].conj() * duu[I_B[xyz], :, n_]).imag * eig * g)
    morb1 = morb1_2 + morb1_1 * f
    return morb1, morb2

@njit(nogil=True)
def _get_morb_k(eig, num_wann, ef, Ah_ak, Ah_bk, fo_k, f, xyz):
    g = 1.0 - f
    # f.(ef - eig).Omega
    morb2 = fo_k[xyz] * (ef - eig)
    # -0.5.f.eig.Omega
    morb1_2 = fo_k[xyz] * 0.5 * eig
    # f.<duu_n_a|H|duu_n_b> = f.<duu_n_a|uu_m>e_n<uu_m|duu_n_b>
    morb1_1 = np.zeros(num_wann, dtype=np.float64)
    for n_ in range(num_wann):
        morb1_1[n_] = np.sum(eig * g * (Ah_ak[I_A[xyz], n_, :] * Ah_bk[I_B[xyz], :, n_]).imag)
    morb1 = morb1_2 + morb1_1 * f
    return morb1, morb2

@njit(parallel=True, nogil=True)
def get_totmorb_kpar_kpath(ham_R, r_mat_R, R_vec, R_cartT,
                           num_wann, kpts, q_frac, q, ef, eta, xyz):
    fac = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    # 0: morb1; 1: morb2; 2: totmorb
    morb_k = np.zeros((nkpts, 3), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        Ah_k, eig, uu = _get_Ah_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt)
        # ham_k, eig, uu = _ham_k_system(ham_R, R_vec, R_cartT, kpt)
        # duu = get_deltaU_h(ham_R, R_vec, R_cartT, num_wann, kpt, uu, q_frac, q, order=2)
        f = occ_fermi(eig, ef, eta)
        # fo_k[idim, iband]
        of_k = _get_f_omega(Ah_k, f, num_wann)
        # of_k = _get_f_omega_interpolation(f, duu, num_wann)
        morb1, morb2 = _get_morb_k(eig, num_wann, ef, Ah_k, Ah_k, of_k, f, xyz)
        morb_k[ik, 0] = np.sum(morb1)
        morb_k[ik, 1] = np.sum(morb2)
    morb_k[:, 2] = morb_k[:, 0] + morb_k[:, 1]
    return morb_k * fac


@njit(nogil=True)
def _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt, xyz, ss_R, subwf):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d = get_inv_e_d(eig, num_wann)
    # for subwf
    Ah_a = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Ah_b = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    js_a = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    if 10 in itasks: # shc
        sw = fourier_R_to_k_vec3(ss_R, fac)
    for i in range(3):
        Ah_b[i] = unitary_trans(A_bar_k[i], uu) + 1j * unitary_trans(ham_out[i + 1], uu) * inv_e_d
        if 0 in itasks or 20 in itasks:  # ahc && morb
            if subwf is not None:
                Ah_a[i] = (unitary_trans_sub(A_bar_k[i, subwf, :], uu[subwf, :], uu)
                            + 1j * unitary_trans_sub(ham_out[i + 1, subwf, :], uu[subwf, :], uu) * inv_e_d)
                Ah_a[i] = (Ah_a[i] + Ah_a[i].T.conj()) * 0.5
            else:
                Ah_a[i] = Ah_b[i]
        if 10 in itasks: # shc
            mat_S = unitary_trans(sw[i], uu) if subwf is None \
                else unitary_trans_sub(sw[i, subwf, :], uu[i, subwf, :], uu)
            # j^spin_a
            mat_B = mat_S @ Ah_b[I_A[xyz]]
            js_a[i] = (mat_B + mat_B.T.conj()) * 0.5
    return eig, Ah_a, Ah_b, js_a

@njit(parallel=True, nogil=True)
def berry_kpath(itasks, ham_R, r_mat_R, R_vec, R_cartT,
                num_wann, kpts, ef, eta, xyz=2, ss_R=None, subwf=None):
    """
    itasks:
    0: ahc
    10: shc
    20: morb
    """
    fac1 = TwoPi
    fac2 = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    # sigma_x, sigma_y, sigma_z
    ahc_ks = np.zeros((nkpts, 3), dtype=np.float64) if 0 in itasks else None
    # sigma^x_ab, sigma^y_ab sigma^z_ab
    shc_ks = np.zeros((nkpts, 3), dtype=np.float64) if 10 in itasks else None
    # morb1, morb2, morb
    morb_ks = np.zeros((nkpts, 3), dtype=np.float64) if 20 in itasks else None
    for ik in prange(nkpts):
        kpt = kpts[ik]
        eig, Ah_bk, Ah_ak, js = _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt,
                                            xyz, ss_R, subwf=subwf)

        f = occ_fermi(eig, ef, eta)
        if 0 in itasks:
            of_k = _get_f_omega(Ah_bk, f, num_wann) if subwf is None \
                else _get_f_omega_sub(Ah_ak, Ah_bk, f, num_wann)
            ahc_ks[ik, :] = np.sum(of_k, axis=1)
        if 10 in itasks:
            osf_k = _get_f_spins_omega(js, Ah_bk[I_B[xyz]], f, num_wann)
            shc_ks[ik, :] = np.sum(osf_k, axis=1)
        if 20 in itasks:
            of_k = _get_f_omega(Ah_bk, f, num_wann) if subwf is None \
                else _get_f_omega_sub(Ah_ak, Ah_bk, f, num_wann)
            morb1, morb2 = _get_morb_k(eig, num_wann, ef, Ah_ak, Ah_bk, of_k, f, xyz)
            morb_ks[ik, 0] = np.sum(morb1)
            morb_ks[ik, 1] = np.sum(morb2)
    count = itasks.shape[0] * 3
    out = np.zeros((nkpts, count), dtype=np.float64)
    count = 0
    if 0 in itasks:
        out[:, count:count+3] = ahc_ks * fac1
        count += 3
    if 10 in itasks:
        out[:, count:count + 3] = shc_ks * fac1
        count += 3
    if 20 in itasks:
        morb_ks[:, 2] = morb_ks[:, 0] + morb_ks[:, 1]
        out[:, count:count + 3] = morb_ks * fac2
        count += 3
    return out

@njit(parallel=True, nogil=True)
def berry_fermi(itasks, ham_R, r_mat_R, R_vec, R_cartT,
                num_wann, kpts, efs, eta, xyz=2, ss_R=None, subwf=None):
    """
        itasks:
        0: ahc
        10: shc
        20: morb
    """
    fac1 = -TwoPi
    fac2 = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    # sigma_x, sigma_y, sigma_z
    if 0 in itasks:
        ahc_ks = np.zeros((n_ef, 3, nkpts), dtype=np.float64)
    if 10 in itasks:
        # sigma^x_ab, sigma^y_ab sigma^z_ab
        shc_ks = np.zeros((n_ef, 3, nkpts), dtype=np.float64)
    if 20 in itasks:
        # morb1, morb2, morb
        morb_ks = np.zeros((n_ef, 2, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        eig, Ah_ak, Ah_bk, js_ak = _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, kpt,
                                               xyz, ss_R, subwf)
        for i in range(n_ef):
            ef = efs[i]
            f = occ_fermi(eig, ef, eta)
            if 0 in itasks:
                of_k = _get_f_omega(Ah_bk, f, num_wann) if subwf is None \
                    else _get_f_omega_sub(Ah_ak, Ah_bk, f, num_wann)
                ahc_ks[i, :, ik] = np.sum(of_k, axis=1)
            if 10 in itasks:
                osf_k = _get_f_spins_omega(js_ak, Ah_bk[I_B[xyz]], f, num_wann)
                shc_ks[i, :, ik] = np.sum(osf_k, axis=1)
            if 20 in itasks:
                of_k = _get_f_omega(Ah_bk, f, num_wann) if subwf is None \
                    else _get_f_omega_sub(Ah_ak, Ah_bk, f, num_wann)
                morb1, morb2 = _get_morb_k(eig, num_wann, ef, Ah_ak, Ah_bk, of_k, f, xyz)
                morb_ks[i, 0, ik] = np.sum(morb1)
                morb_ks[i, 1, ik] = np.sum(morb2)
    count = itasks.shape[0] * 3
    out = np.zeros((n_ef, count), dtype=np.float64)
    count = 0
    if 0 in itasks:
        out[:, count:count+3] = fac1 * np.sum(ahc_ks, axis=2)
        count += 3
    if 10 in itasks:
        out[:, count:count+3] = fac1 * np.sum(shc_ks, axis=2)
        count += 3
    if 20 in itasks:
        out[:, count:count+2] = fac2 * np.sum(morb_ks, axis=2)
        out[:, count+2] = out[:, count] + out[:, count+1]
        count += 3
    return out / nkpts