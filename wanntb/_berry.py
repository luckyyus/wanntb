import numpy as np
from numba import njit, prange
from .constant import TwoPi, Hbar_, Mu_B_
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi, \
    unitary_trans_sub, inv_e_d_c

I_A = np.array([1, 2, 0], dtype=np.int32)
I_B = np.array([2, 0, 1], dtype=np.int32)

@njit(nogil=True)
def _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, xyz, ss_R, subwf):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d, e_d = inv_e_d_c(eig, num_wann, eta)
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
                else unitary_trans_sub(sw[i, subwf, :], uu[subwf, :], uu)
            va = - 1.0j * e_d * unitary_trans(A_bar_k[I_A[xyz]], uu) + unitary_trans(ham_out[I_A[xyz] + 1], uu)
            # j^spin_a
            mat_B = mat_S @ va
            js_a[i] = (mat_B + mat_B.T.conj()) * 0.5 * inv_e_d * -1.0j
    return eig, Ah_a, Ah_b, js_a


@njit(nogil=True)
def _get_f_omega(Ah_ak, Ah_bk, f, num_wann):
    fo_k = np.zeros((3, num_wann), dtype=np.float64)
    g = 1.0 - f
    for i in range(3):
        for n_ in range(num_wann):
            fo_k[i, n_] = np.sum(g * (Ah_ak[I_A[i], n_, :] * Ah_bk[I_B[i], :, n_]).imag)
    fo_k *= -2.0 * f
    return fo_k

@njit(nogil=True)
def _get_omega_gmat(Ah_ak, Ah_bk, f, num_wann):
    g = 1.0 - f
    fo_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)

    for i in range(3):
        for m_ in range(num_wann):
            for n_ in range(num_wann):
                # omega_z,mn = i \sum { A_x,ml . A_y,ln - A_x,ln . A_y, ml}
                # omega_z,nm = i \sum { A_x,nl . A_y,lm - A_x,lm . A_y, nl}
                #            = i \sum { A*_x,ln . A*_y,ml - A*_x,nl . A*_y, lm}
                #            = (-i \sum { A_x,ln . A_y,ml - A_x,nl . A_y, lm})* = omega_z,mn*
                fo_k[i, m_, n_] = np.sum(g * (Ah_ak[I_A[i], m_, :] * Ah_bk[I_B[i], :, n_]
                                              - Ah_ak[I_A[i], :, n_] * Ah_bk[I_B[i], m_, :]))
    fo_k *= 1.0j
    return fo_k

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

@njit(nogil=True)
def _get_morb_gmat_k(eig, num_wann, ef, Ah_ak, Ah_bk, o_mat, f, xyz):
    g = 1.0 - f
    morb2 = o_mat[xyz] * (ef - eig)
    morb1_2 = o_mat[xyz] * 0.5 * eig
    morb1_1 = np.zeros((num_wann, num_wann), dtype=np.complex128)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            morb1_1[m_, n_] = np.sum(eig * g * (Ah_ak[I_A[xyz], m_, :] * Ah_bk[I_B[xyz], :, n_]
                                              - Ah_ak[I_A[xyz], :, n_] * Ah_bk[I_B[xyz], m_, :]))
    morb1 = morb1_2 + morb1_1 * 0.5j
    return morb1, morb2


@njit(parallel=True, nogil=True)
def berry_kpath(itasks, ham_R, r_mat_R, R_vec, R_cartT,
                num_wann, kpts, ef, eta, xyz=2, ss_R=None, subwf=None):
    """
    itasks:
    0: ahc
    10: shc
    20: morb
    """
    fac1 = -TwoPi
    fac2 = 1e-8 / Hbar_ / Mu_B_
    nkpts = kpts.shape[0]
    if 0 in itasks:
        # sigma_x, sigma_y, sigma_z
        ahc_ks = np.zeros((nkpts, 3), dtype=np.float64)
    if 10 in itasks:
        # sigma^x_ab, sigma^y_ab sigma^z_ab
        shc_ks = np.zeros((nkpts, 3), dtype=np.float64)
    if 20 in itasks:
        # morb1, morb2, morb
        morb_ks = np.zeros((nkpts, 3), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        eig, Ah_bk, Ah_ak, js = _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, xyz, ss_R,
                                            subwf)

        f = occ_fermi(eig, ef, eta)
        if 0 in itasks:
            of_k = _get_f_omega(Ah_ak, Ah_bk, f, num_wann)
            ahc_ks[ik, :] = np.sum(of_k, axis=1)
        if 10 in itasks:
            osf_k = _get_f_spins_omega(js, Ah_bk[I_B[xyz]], f, num_wann)
            shc_ks[ik, :] = np.sum(osf_k, axis=1)
        if 20 in itasks:
            of_k = _get_f_omega(Ah_ak, Ah_bk, f, num_wann)
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
    if 0 in itasks:
        # sigma_x, sigma_y, sigma_z
        ahc_ks = np.zeros((n_ef, 3, nkpts), dtype=np.float64)
    if 10 in itasks:
        # sigma^x_ab, sigma^y_ab sigma^z_ab
        shc_ks = np.zeros((n_ef, 3, nkpts), dtype=np.float64)
    if 20 in itasks:
        # morb1, morb2, morb
        morb_ks = np.zeros((n_ef, 2, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        eig, Ah_ak, Ah_bk, js_ak = _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, xyz, ss_R,
                                               subwf)
        for i in range(n_ef):
            ef = efs[i]
            f = occ_fermi(eig, ef, eta)
            if 0 in itasks:
                of_k = _get_f_omega(Ah_ak, Ah_bk, f, num_wann)
                ahc_ks[i, :, ik] = np.sum(of_k, axis=1)
            if 10 in itasks:
                osf_k = _get_f_spins_omega(js_ak, Ah_bk[I_B[xyz]], f, num_wann)
                shc_ks[i, :, ik] = np.sum(osf_k, axis=1)
            if 20 in itasks:
                of_k = _get_f_omega(Ah_ak, Ah_bk, f, num_wann)
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