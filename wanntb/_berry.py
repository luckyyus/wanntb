import numpy as np
from numba import njit, prange
from .constant import TwoPi, Hbar_, Mu_B_
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi


@njit(nogil=True)
def get_berry_curvature_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, eta):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    e_d = np.zeros((num_wann,num_wann), dtype=np.float64)
    inv_e_d = np.zeros((num_wann,num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d[m_, n_] = eig[m_] - eig[n_]
            inv_e_d[m_, n_] = 1.0 / (e_d[m_, n_]*e_d[m_, n_] + eta * eta)
            # inv_e_d[m_, n_] = 1.0 / (eig[m_] - eig[n_]) if abs(eig[m_] - eig[n_]) > 1e-7 else 0.0
    Ah_k = np.zeros((3,num_wann,num_wann), dtype=np.complex128)
    # A_bar^W_a[3, num_wann, num_wann] in units angst.

    for i in range(3):
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        # A^H_a = A_bar^H_a + i D^H_a = i<psi_m| del_a psi_n>
        Ah_k[i] = 1j * e_d * unitary_trans(A_bar_k[i], uu) - unitary_trans(ham_out[i+1], uu)
    omega_k = np.zeros((3, num_wann), dtype=np.float64)
    for n_ in range(num_wann):
        omega_k[0, n_] = np.sum((Ah_k[1, n_, :] * Ah_k[2, :, n_] * inv_e_d[n_, :]).imag)
        omega_k[1, n_] = np.sum((Ah_k[2, n_, :] * Ah_k[0, :, n_] * inv_e_d[n_, :]).imag)
        omega_k[2, n_] = np.sum((Ah_k[0, n_, :] * Ah_k[1, :, n_] * inv_e_d[n_, :]).imag)
    omega_k *= -2.0
    return omega_k, eig


@njit(nogil=True)
def _get_Ah_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d = eig[m_] - eig[n_]
            inv_e_d[m_, n_] = 1.0 / e_d if abs(e_d) > 1e-8 else 0.0
    Ah_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    for i in range(3):
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        # A^H_a = A_bar^H_a + i D^H_a = i<psi_m| del_a psi_n>
        Ah_k[i] = unitary_trans(A_bar_k[i], uu) + 1j * unitary_trans(ham_out[i + 1], uu) * inv_e_d
    return Ah_k, eig


def _get_Ah_bar_Dh_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d = eig[m_] - eig[n_]
            inv_e_d[m_, n_] = 1.0 / e_d if abs(e_d) > 1e-8 else 0.0
    Ah_bar_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Dh_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for i in range(3):
        Ah_bar_k[i] = unitary_trans(A_bar_k[i], uu)
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        Dh_k[i] = unitary_trans(ham_out[i + 1], uu) * inv_e_d
    return Ah_bar_k, Dh_k, eig


@njit(nogil=True)
def _get_f_omega(Ah_k, f, num_wann):
    fo_k = np.zeros((3, num_wann), dtype=np.float64)
    g = 1.0 - f
    for n_ in range(num_wann):
        fo_k[0, n_] = np.sum((Ah_k[1, n_, :] * g * Ah_k[2, :, n_]).imag)
        fo_k[1, n_] = np.sum((Ah_k[2, n_, :] * g * Ah_k[0, :, n_]).imag)
        fo_k[2, n_] = np.sum((Ah_k[0, n_, :] * g * Ah_k[1, :, n_]).imag)
    fo_k *= -2.0 * f
    return fo_k


@njit(nogil=True)
def _get_berry_curv_f_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta):
    Ah_k, eig = _get_Ah_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt)
    f = occ_fermi(eig, ef, eta)
    of_k = _get_f_omega(Ah_k, f, num_wann)
    return of_k, eig


@njit(nogil=True)
def _get_berry_curv_f_efs_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, efs, eta):
    n_ef = efs.shape[0]
    Ah_k, eig = _get_Ah_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt)
    ofg_k = np.zeros((3, n_ef, num_wann), dtype=np.float64)
    for i in range(n_ef):
        ef = efs[i]
        f = occ_fermi(eig, ef, eta)
        ofg_k[:, i, :] = _get_f_omega(Ah_k, f, num_wann)
    return np.sum(ofg_k, axis=2)


@njit(nogil=True)
def _get_f_omega_new(Ah_bar_k, Dh_k, f, num_wann):
    i_A = [1, 2, 0]
    i_B = [2, 0, 1]
    g = 1.0 - f
    # Diag[Im(A(H)_bar_alpha.A(H)_bar_beta)]
    o_bar = np.zeros((3, num_wann), dtype=np.float64)
    # omega_i = np.zeros((3, num_wann), dtype=np.float64)
    # Diag[Im(D(H)_bar_alpha.D(H)_bar_beta)]
    o_d = np.zeros((3, num_wann), dtype=np.float64)
    for i in range(3):
        for n_ in range(num_wann):
            o_bar[i, n_] = np.sum((Ah_bar_k[i_A[i], n_, :] * Ah_bar_k[i_B[i], :, n_]).imag)
            # omega_i[i, n_] = np.sum(g *
            #                           (Dh_k[i_A[i], n_, :] * Ah_bar_k[i_B[i], :, n_]
            #                            - Dh_k[i_B[i], n_, :] * Ah_bar_k[i_A[i], :, n_]).imag)
            o_d[i, n_] = np.sum((Dh_k[i_A[i], n_, :] * g * Dh_k[i_B[i], :, n_]).imag)
    omega = (o_bar - o_d) * f
    omega *= -2.0
    return omega


@njit(nogil=True)
def _get_berry_curv_f_eig_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta):
    Ah_bar_k, Dh_k, eig = _get_Ah_bar_Dh_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt)
    f = occ_fermi(eig, ef, eta)
    omega = _get_f_omega_new(Ah_bar_k, Dh_k, f, num_wann)
    return omega, eig


@njit(nogil=True)
def _get_berry_curv_f_efs_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, efs, eta):
    n_ef = efs.shape[0]
    Ah_bar_k, Dh_k, eig = _get_Ah_bar_Dh_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt)
    ofg_k = np.zeros((3, n_ef, num_wann), dtype=np.float64)
    for i in range(n_ef):
        ef = efs[i]
        f = occ_fermi(eig, ef, eta)
        ofg_k[:, i, :] = _get_f_omega_new(Ah_bar_k, Dh_k, f, num_wann)
    return np.sum(ofg_k, axis=2)


@njit(parallel=True, nogil=True)
def get_berry_curv_kpar_kpath(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                              num_wann, kpts, ef, eta, lnew):
    # fac = - 1.0
    nkpts = kpts.shape[0]
    list_o_k = np.zeros((3, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        if lnew:
            oo, eig = _get_berry_curv_f_eig_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        else:
            oo, eig = _get_berry_curv_f_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        list_o_k[:, ik] = np.sum(oo, axis=1)
    return list_o_k.T


@njit(parallel=True, nogil=True)
def get_ahc_kpar_fermi(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                       num_wann, kpts, efs, eta, lnew):
    fac = - TwoPi
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    list_o_ef_k = np.zeros((3, n_ef, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        if lnew:
            list_o_ef_k[:, :, ik] = _get_berry_curv_f_eig_k_new(
                ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, efs, eta)
        else:
            list_o_ef_k[:, :, ik] = _get_berry_curv_f_efs_k(
                ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, efs, eta)
    return (np.sum(list_o_ef_k, axis=2) / nkpts * fac).T


@njit(parallel=True, nogil=True)
def get_morb_berry_kpar_kpath(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                              num_wann, kpts, ef, eta, direction, lnew):
    fac = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    morb_k = np.zeros(nkpts, dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        if lnew:
            oo, eig = _get_berry_curv_f_eig_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        else:
            oo, eig = _get_berry_curv_f_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        morb_k[ik] = np.sum(oo[direction-1, :] * (ef - eig))
    return morb_k * fac


@njit(parallel=True, nogil=True)
def get_morb_berry_kpar(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                        num_wann, kpts, ef, eta, direction, lnew):
    fac = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    morb_k = np.zeros(nkpts, dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        if lnew:
            oo, eig = _get_berry_curv_f_eig_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        else:
            oo, eig = _get_berry_curv_f_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        morb_k[ik] = np.sum(oo[direction - 1, :] * (ef - eig))
    return np.sum(morb_k) * fac / nkpts
