import numpy as np
from numba import njit, prange
from .constant import TwoPi, Hbar_, Mu_B_
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi, \
    unitary_trans_sub, inv_e_d_c

I_A = np.array([1, 2, 0], dtype=np.int32)
I_B = np.array([2, 0, 1], dtype=np.int32)

@njit(nogil=True)
def _get_Ah_ab_S_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, ss_R=None, subwf=None, subwf2=None):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d, e_d = inv_e_d_c(eig, num_wann, eta)
    Ah_ak = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Ah_bk = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    S_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    if ss_R is not None:
        sw = fourier_R_to_k_vec3(ss_R, fac)
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    for i in range(3):
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_mna / (e_n - e_m)}
        # A^H_a = A_bar^H_a + i D^H_a = i<psi_m| del_a psi_n>
        Ah_bk[i] = unitary_trans(A_bar_k[i], uu) + 1j * unitary_trans(ham_out[i + 1], uu) * inv_e_d
        if subwf is None:
            Ah_ak[i] = Ah_bk[i].conj()
        else:
            Ah_ak[i] = (unitary_trans_sub(A_bar_k[i, subwf, :], uu[subwf, :], uu)
                        + 1j * unitary_trans_sub(ham_out[i + 1, subwf, :], uu[subwf, :], uu) * inv_e_d.conj())
            Ah_ak[i] = (Ah_ak[i] + Ah_ak[i].T.conj()) * 0.5
        if ss_R is not None:
            mat_S = unitary_trans(sw[i], uu) if subwf2 is None \
                else unitary_trans_sub(sw[i, subwf2, :], uu[subwf2, :], uu)
            S_k[i] = mat_S if subwf2 is None else 0.5 * (mat_S + mat_S.T.conj())
    return eig, uu, Ah_ak, Ah_bk, S_k


@njit(nogil=True)
def _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, xyz, ss_R, subwf):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d, e_d = inv_e_d_c(eig, num_wann, eta)

    Ah_a = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Ah_b = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    js_a = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    if 10 in itasks: # shc
        sw = fourier_R_to_k_vec3(ss_R, fac)
    for i in range(3):
        # $A_{nm} = \langle u_n | i \nabla_k | u_m \rangle$
        Ah_b[i] = unitary_trans(A_bar_k[i], uu) + 1j * unitary_trans(ham_out[i + 1], uu) * inv_e_d

        if 0 in itasks or 20 in itasks:  # ahc && morb
            if subwf is not None:
                Ah_a[i] = (unitary_trans_sub(A_bar_k[i, subwf, :], uu[subwf, :], uu)
                            + 1j * unitary_trans_sub(ham_out[i + 1, subwf, :], uu[subwf, :], uu) * inv_e_d.conj())
                Ah_a[i] = (Ah_a[i] + Ah_a[i].T.conj()) * 0.5
            else:
                Ah_a[i] = Ah_b[i].conj()
        if 10 in itasks: # shc
            mat_S = unitary_trans(sw[i], uu) if subwf is None \
                else unitary_trans_sub(sw[i, subwf, :], uu[subwf, :], uu)
            va = - 1.0j * e_d * unitary_trans(A_bar_k[I_A[xyz]], uu) + unitary_trans(ham_out[I_A[xyz] + 1], uu)
            # j^spin_a
            mat_B = mat_S @ va
            js_a[i] = (mat_B + mat_B.T.conj()) * -0.5j * inv_e_d.conj()
    return eig, uu, Ah_a, Ah_b, js_a


@njit(nogil=True)
def _get_f_omega(Ah_ak, Ah_bk, f, num_wann):
    fo_k = np.zeros((3, num_wann), dtype=np.float64)
    g = 1.0 - f
    for i in range(3):
        for n_ in range(num_wann):
            # \Omega = Im <A_x | A_y>
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
def _get_morb_gmat_k(eig, num_wann, ef, Ah_ak, Ah_bk, omega_mat, f, xyz):
    g = 1.0 - f
    morb2 = omega_mat[xyz] * (ef - eig)
    morb1_2 = omega_mat[xyz] * 0.5 * eig
    morb1_1 = np.zeros((num_wann, num_wann), dtype=np.complex128)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            morb1_1[m_, n_] = np.sum(eig * g * (Ah_ak[I_A[xyz], m_, :] * Ah_bk[I_B[xyz], :, n_]
                                              - Ah_ak[I_A[xyz], :, n_] * Ah_bk[I_B[xyz], m_, :]))
    morb1 = morb1_2 + morb1_1 * 0.5j
    return morb1, morb2


# @njit(nogil=True)
# def _get_sigma_diag(ss_R, R_vec, kpt, uu, num_wann, subwf=None):
#     """ 计算 x, y, z 三个方向自旋算符的对角元 """
#     fac = fourier_phase_R_to_k(R_vec, kpt)
#     sw = fourier_R_to_k_vec3(ss_R, fac)
#     sigma_diag = np.zeros((3, num_wann), dtype=np.float64)
#     for i in range(3):
#         s_bar = unitary_trans(sw[i], uu) if subwf is None \
#             else unitary_trans_sub(sw[i, subwf, :], uu[subwf, :], uu)
#         for n_ in range(num_wann):
#             sigma_diag[i, n_] = s_bar[n_, n_].real
#     return sigma_diag


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
        eig, _, Ah_bk, Ah_ak, js = _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, xyz, ss_R,
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
        eig, _, Ah_ak, Ah_bk, js_ak = _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, xyz, ss_R,
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


@njit(parallel=True, nogil=True)
def intra_shc_fermi(ham_R, r_mat_R, R_vec, R_cartT, ss_R, num_wann, kpts, efs, eta, xyz=2, subwf=None):
    """
    计算 Intraband SHC
    """
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    shc_k = np.zeros((n_ef, 3, nkpts), dtype=np.float64)

    for ik in prange(nkpts):
        kpt = kpts[ik]
        eig, uu, Ah_a, Ah_b, _S = _get_Ah_ab_S_k(ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt,
                                                ss_R=ss_R, subwf=None, subwf2=subwf)

        for i in range(n_ef):
            ef = efs[i]
            f = occ_fermi(eig, ef, eta)
            omega = _get_f_omega(Ah_a, Ah_b, f, num_wann)

            for s_i in range(3):
                shc_k[i, s_i, ik] = np.sum(omega[xyz] * np.diag(_S[s_i]).real)

    return np.sum(shc_k, axis=2) / nkpts

# @njit(parallel=True, nogil=True)
# def intra_shc_fermi(ham_R, r_mat_R, R_vec, R_cartT, ss_R, num_wann, kpts, efs, eta, xyz=2, subwf=None):
#     """
#     计算 Intraband SHC
#     """
#     nkpts = kpts.shape[0]
#     n_ef = efs.shape[0]
#     shc_k = np.zeros((n_ef, 3, nkpts), dtype=np.float64)
#
#     itasks = np.array([0], dtype=np.int32)
#
#     for ik in prange(nkpts):
#         kpt = kpts[ik]
#         eig, uu, _, Ah_b, _ = _berry_Ah_k(itasks, ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, xyz, ss_R, subwf)
#
#         # 在_berry_Ah_k 输出了uu 下面代码废掉
#         # fac = fourier_phase_R_to_k(R_vec, kpt)
#         # ham_k = fourier_R_to_k(ham_R, R_cartT, fac, iout=[0])[0]
#         # _, uu = np.linalg.eigh(ham_k)
#
#         sigma_diags = _get_sigma_diag(ss_R, R_vec, kpt, uu, num_wann)
#
#         for s_i in range(3):
#             sigma = sigma_diags[s_i]
#
#             # 构造 Ah_W，$Ah_w_{nm} = \sigma_{nn} * A_{nm}$
#             Ah_w = np.zeros_like(Ah_b)
#             for i in range(3):
#                 for m_ in range(num_wann):
#                     for n_ in range(num_wann):
#                         Ah_w[i, m_, n_] = 1/2 * (sigma[m_] + sigma[n_]) * Ah_b[i, m_, n_]
#                         # Ah_w[i, m_, n_] = sigma[m_] * Ah_b[i, m_, n_]
#
#             for ie in range(n_ef):
#                 ef = efs[ie]
#                 f = occ_fermi(eig, ef, eta)
#
#                 omega_val = _get_f_omega(Ah_w, Ah_b, f, num_wann)
#
#                 shc_k[ie, s_i, ik] = np.sum(omega_val[xyz])
#
#     return np.sum(shc_k, axis=2) / nkpts

# --------------------------------------------------------------------------
# Functions Merged from _OHE.py
# --------------------------------------------------------------------------

@njit(nogil=True)
def get_morb_mat(ef, ham_R, r_mat_R, R_vec, _R_cartT, num_wann, kpt):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, _R_cartT, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    e_d = inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    f = occ_fermi(eig, ef, eta=1e-8)
    g = np.diag((1 - f)).astype(np.complex128)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d[m_, n_] = eig[m_] - eig[n_]
            e_d1 = eig[m_] - eig[n_]
            inv_e_d[m_, n_] = - 1.0 / e_d1 if abs(e_d1) > 1e-8 else 0.0
    Abar_h_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Dbar_h_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Dh_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    vh_bar_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    vh_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    tmp = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Ah_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    fo_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for i in range(3):
        Abar_h_k[i] = unitary_trans(A_bar_k[i], uu)
        vh_bar_k[i] = unitary_trans(ham_out[i + 1], uu)
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        Dbar_h_k[i] = vh_bar_k[i] * inv_e_d
        Ah_k[i] = Abar_h_k[i] + 1j * Dbar_h_k[i]
        Dh_k[i] = Dbar_h_k[i] - 1j * Abar_h_k[i]
        vh_k[i] = vh_bar_k[i] + 1j * e_d * Abar_h_k[i]
        tmp[i] = g @ Dh_k[i]

    deltaU = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for i in range(num_wann):
        for j in range(num_wann):
            for ii in range(3):
                deltaU[ii, :, i] += tmp[ii, j, i] * uu[:, j]
    g = 1 - f
    for i in range(3):
        for m_ in range(num_wann):
            for n_ in range(num_wann):
                fo_k[i, m_, n_] = np.sum(g * (Ah_k[I_A[i], m_, :] * Ah_k[I_B[i], :, n_]
                                              - Ah_k[I_A[i], :, n_] * Ah_k[I_B[i], m_, :]))
    fo_k *= 1j

    morb1_1 = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    morb1_2 = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    eig_mat = np.zeros((num_wann, num_wann), dtype=np.float64)
    for i in range(num_wann):
        for j in range(num_wann):
            eig_mat[i, j] = 0.25 * (eig[i] + eig[j])
    for i in range(3):
        morb1_2[i] = eig_mat * fo_k[i]

    operator = 0.5 * ham_out[0]
    for i in range(num_wann):
        for j in range(num_wann):
            morb1_1[0, i, j] = deltaU[1, :, i].conj().T @ operator @ deltaU[2, :, j] - deltaU[2, :,
                                                                                       i].conj().T @ operator @ deltaU[
                                                                                                                1, :, j]
            morb1_1[1, i, j] = deltaU[2, :, i].conj().T @ operator @ deltaU[0, :, j] - deltaU[0, :,
                                                                                       i].conj().T @ operator @ deltaU[
                                                                                                                2, :, j]
            morb1_1[2, i, j] = deltaU[0, :, i].conj().T @ operator @ deltaU[1, :, j] - deltaU[1, :,
                                                                                       i].conj().T @ operator @ deltaU[
                                                                                                                0, :, j]
    morb1_1 *= -1j
    morb1 = morb1_1 + morb1_2

    return morb1, vh_k, eig, f


@njit(nogil=True)
def get_OBC_kpath(ham_R, r_mat_R, R_vec, _R_cartT, num_wann, kpt, ef, dir):
    factor = 0.262
    morb, vh_k, eig, f = get_morb_mat(ef, ham_R, r_mat_R, R_vec, _R_cartT, num_wann, kpt)
    j_mat = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    OBC = np.zeros((3, num_wann), dtype=np.complex128)
    g = (1 - f).astype(np.complex128)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d1 = eig[m_] - eig[n_]
            inv_e_d[m_, n_] = - 1.0 / e_d1 if abs(e_d1) > 1e-8 else 0.0
    for i in range(3):
        mat = morb[dir] @ vh_k[i]
        j_mat[i] = (mat + mat.conj().T) * 0.5 * factor * inv_e_d * inv_e_d
    for i in range(3):
        for n_ in range(num_wann):
            OBC[i, n_] = np.sum(g * j_mat[I_A[i], n_, :] * vh_k[I_B[i], :, n_])
        #  OBC[i,n_] = np.sum(j_mat[I_A[i],n_,:]*vh_k[I_B[i],:,n_])
    return OBC.imag, f


@njit(parallel=True, nogil=True)
def get_OHE_kpar_kmesh(ham_R, r_mat_R, R_vec, _R_cartT, num_wann, kpts, ef, dir):
    nkpts = kpts.shape[0]
    # num_ef = ef_list.shape[0]
    OBC_f = np.zeros((3, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        OBC, f = get_OBC_kpath(ham_R, r_mat_R, R_vec, _R_cartT, num_wann, kpt, ef, dir)
        for i in range(3):
            OBC_f[i, ik] = np.sum(f * OBC[i])
    return 2 * np.sum(OBC_f, axis=1) / nkpts


@njit(parallel=True, nogil=True)
def get_OHE_kpar_kmesh_fermi(ham_R, r_mat_R, R_vec, _R_cartT, num_wann, kpts, ef_list, dir):
    nkpts = kpts.shape[0]
    num_ef = ef_list.shape[0]
    OBC_f = np.zeros((num_ef, 3, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        for j in range(num_ef):
            ef = ef_list[j]
            OBC, f = get_OBC_kpath(ham_R, r_mat_R, R_vec, _R_cartT, num_wann, kpt, ef, dir)
            for i in range(3):
                OBC_f[j, i, ik] = np.sum(f * OBC[i])
    return 2 * np.sum(OBC_f, axis=2) / nkpts

