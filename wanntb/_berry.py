import numpy as np
from numba import njit, prange
from .constant import TwoPi, Hbar_, Mu_B_
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi, \
    fourier_R_to_k_curl, spin_w, get_eig_da, unitary_trans_sub

I_A = np.array([1, 2, 0], dtype=np.int32)
I_B = np.array([2, 0, 1], dtype=np.int32)

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
            inv_e_d[m_, n_] = -1.0 / (e_d[m_, n_]*e_d[m_, n_] + eta * eta)
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
            inv_e_d[m_, n_] = - 1.0 / e_d if abs(e_d) > 1e-8 else 0.0
    Ah_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    for i in range(3):
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        # A^H_a = A_bar^H_a + i D^H_a = i<psi_m| del_a psi_n>
        Ah_k[i] = unitary_trans(A_bar_k[i], uu) + 1j * unitary_trans(ham_out[i + 1], uu) * inv_e_d
    return Ah_k, eig


@njit(nogil=True)
def _get_vh_jsd_inv2_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, eta, alpha_beta, sw, subwf=None):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    inv2 = np.zeros((num_wann, num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ != n_:
                e_d[m_, n_] = eig[m_] - eig[n_]
                inv_e_d[m_, n_] = - 1.0 / e_d[m_, n_] if abs(e_d[m_, n_]) > 1e-8 else 0.0
                inv2[n_, m_] = 1.0 / (e_d[m_, n_] * e_d[m_, n_] + eta * eta)
    # ham_a = unitary_trans(ham_out[I_A[alpha_beta] + 1], uu)
    # Dh_a = ham_a * inv_e_d
    vh_a = 1j * e_d * unitary_trans(A_bar_k[I_A[alpha_beta]], uu) + unitary_trans(ham_out[I_A[alpha_beta] + 1], uu)
    vh_b = 1j *e_d * unitary_trans(A_bar_k[I_B[alpha_beta]], uu) + unitary_trans(ham_out[I_B[alpha_beta] + 1], uu)
    mat_S = unitary_trans(sw, uu) if subwf is None else unitary_trans_sub(sw[subwf, :], uu[subwf, :], uu)
    # mat_K = mat_S @ Dh_a - 1j * unitary_trans(sw @ A_bar_k[I_A[alpha_beta]], uu)
    # mat_L = unitary_trans(sw @ ham_out[0], uu) @ Dh_a - 1j * unitary_trans(sw @ ham_out[0] @ A_bar_k[I_A[alpha_beta]], uu)
    # mat_B = mat_S * eig_da + mat_K * eig - mat_L
    mat_B = mat_S @ vh_a
    jsd_gamma = mat_B + mat_B.T.conj()
    return vh_b, jsd_gamma, inv2, eig


@njit(nogil=True)
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
            inv_e_d[m_, n_] = - 1.0 / e_d if abs(e_d) > 1e-8 else 0.0
    Ah_bar_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Dh_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for i in range(3):
        Ah_bar_k[i] = unitary_trans(A_bar_k[i], uu)
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        Dh_k[i] = unitary_trans(ham_out[i + 1], uu) * inv_e_d
    return Ah_bar_k, Dh_k, eig


@njit(nogil=True)
def _get_Aw_bar_jw_eig_uu_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    ow_k = fourier_R_to_k_curl(r_mat_R, fac, R_vec_cart_T)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d = eig[m_] - eig[n_]
            inv_e_d[m_, n_] = - 1.0 / e_d if abs(e_d) > 1e-8 else 0.0
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
            fo_k[i, n_] = np.sum((Ah_k[I_A[i], n_, :] * g * Ah_k[I_B[i], :, n_]).imag)
    fo_k *= -2.0 * f
    return fo_k


@njit(nogil=True)
def _get_f_omega_s(vh_b, js2, inv2, f, num_wann):
    fo_k = np.zeros(num_wann, dtype=np.float64)
    g = 1.0 - f
    for n_ in range(num_wann):
        fo_k[n_] = np.sum((g * js2[n_, :] * vh_b[:, n_] * inv2[n_, :]).imag)
    fo_k *= -2.0 * f
    return fo_k

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
            o_bar[i, n_] = np.sum((Ah_bar_k[I_A[i], n_, :] * Ah_bar_k[I_B[i], :, n_]).imag)
            o_i[i, n_] = np.sum(g *
                                ((Dh_k[I_A[i], n_, :] * Ah_bar_k[I_B[i], :, n_]).real
                                 - (Dh_k[I_B[i], n_, :] * Ah_bar_k[I_A[i], :, n_]).real))
            o_d[i, n_] = np.sum((Dh_k[I_A[i], n_, :] * g * Dh_k[I_B[i], :, n_]).imag)
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
def _get_berrycurv_f_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta):
    Ah_k, eig, = _get_Ah_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt)
    f = occ_fermi(eig, ef, eta)
    of_k = _get_f_omega(Ah_k, f, num_wann)
    return of_k, eig


@njit(nogil=True)
def _get_berrycurv_f_efs_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, efs, eta):
    n_ef = efs.shape[0]
    Ah_k, eig, = _get_Ah_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt)
    ofg_k = np.zeros((3, n_ef, num_wann), dtype=np.float64)
    for i in range(n_ef):
        ef = efs[i]
        f = occ_fermi(eig, ef, eta)
        ofg_k[:, i, :] = _get_f_omega(Ah_k, f, num_wann)
    return np.sum(ofg_k, axis=2)


@njit(nogil=True)
def _get_shc_f_efs_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, efs, eta, alpha_beta, sw, subwf=None):
    n_ef = efs.shape[0]
    v_b, jsd, inv2, eig = _get_vh_jsd_inv2_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt,
                                                  eta, alpha_beta, sw, subwf=subwf)
    ofg_k = np.zeros((n_ef, num_wann), dtype=np.float64)
    for i in range(n_ef):
        ef = efs[i]
        f = occ_fermi(eig, ef, eta)
        ofg_k[i, :] = _get_f_omega_s(v_b, jsd, inv2, f, num_wann)
    return np.sum(ofg_k, axis=1)


@njit(nogil=True)
def _get_berrycurv_f_eig_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta):
    Ah_bar_k, Dh_k, eig = _get_Ah_bar_Dh_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt)
    f = occ_fermi(eig, ef, eta)
    omega = _get_f_omega_new(Ah_bar_k, Dh_k, f, num_wann)
    return omega, eig


@njit(nogil=True)
def _get_berrycurv_f_efs_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, efs, eta):
    n_ef = efs.shape[0]
    Ah_bar_k, Dh_k, eig = _get_Ah_bar_Dh_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt)
    ofg_k = np.zeros((3, n_ef, num_wann), dtype=np.float64)
    for i in range(n_ef):
        ef = efs[i]
        f = occ_fermi(eig, ef, eta)
        ofg_k[:, i, :] = _get_f_omega_new(Ah_bar_k, Dh_k, f, num_wann)
    return np.sum(ofg_k, axis=2)


@njit(nogil=True)
def _get_berrycurv_f_eig_k_w(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta):
    A_bar_k, ow_k, jw_k, eig, uu = _get_Aw_bar_jw_eig_uu_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt)
    f = occ_fermi(eig, ef, eta)
    omega = _get_f_omega_w(A_bar_k, ow_k, jw_k, f, uu, num_wann)
    return omega, eig


@njit(parallel=True, nogil=True)
def get_berrycurv_kpar_kpath(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                             num_wann, kpts, ef, eta):
    # fac = - 1.0
    nkpts = kpts.shape[0]
    list_o_k = np.zeros((3, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        # oo, eig = _get_berrycurv_f_eig_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        # oo, eig = _get_berrycurv_f_eig_k_w(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        oo, eig = _get_berrycurv_f_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        list_o_k[:, ik] = np.sum(oo, axis=1)
    return list_o_k.T


@njit(parallel=True, nogil=True)
def get_ahc_kpar_fermi(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                       num_wann, kpts, efs, eta):
    fac = - TwoPi
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    list_o_ef_k = np.zeros((3, n_ef, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        # list_o_ef_k[:, :, ik] = _get_berrycurv_f_efs_k_new(
        #         ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, efs, eta)
        list_o_ef_k[:, :, ik] = _get_berrycurv_f_efs_k(
                ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, efs, eta)
    return (np.sum(list_o_ef_k, axis=2) / nkpts * fac).T


def get_shc_kpar_fermi(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                       num_wann, kpts, efs, eta, alpha_beta, gamma, udud_order, subwf):
    fac = - TwoPi
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    sw = spin_w(gamma, num_wann, udud_order)
    list_o_ef_k = np.zeros((n_ef, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        list_o_ef_k[:, ik] = _get_shc_f_efs_k(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                                              num_wann, kpt, efs, eta, alpha_beta, sw, subwf=subwf)
    return np.sum(list_o_ef_k, axis=1) / nkpts * fac


@njit(parallel=True, nogil=True)
def get_morb_berry_kpar_kpath(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                              num_wann, kpts, ef, eta, direction):
    fac = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    morb_k = np.zeros(nkpts, dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        # oo, eig = _get_berrycurv_f_eig_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        # oo, eig = _get_berrycurv_f_eig_k_w(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        oo, eig = _get_berrycurv_f_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        morb_k[ik] = np.sum(oo[direction-1, :] * (ef - eig))
    return morb_k * fac


@njit(parallel=True, nogil=True)
def get_morb_berry_kpar(ham_R, r_mat_R, R_vec, R_vec_cart_T,
                        num_wann, kpts, ef, eta, direction):
    fac = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    morb_k = np.zeros(nkpts, dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        # oo, eig = _get_berrycurv_f_eig_k_new(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        oo, eig = _get_berrycurv_f_eig_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, ef, eta)
        morb_k[ik] = np.sum(oo[direction - 1, :] * (ef - eig))
    return np.sum(morb_k) * fac / nkpts
