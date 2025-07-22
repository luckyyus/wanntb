import numpy as np
from numba import njit, prange

from ._berry import _get_berrycurv_f_eig_k, _get_Ah_k, _get_morb_k
from .constant import TwoPi, Hbar_, Mu_B_
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi, \
    fourier_R_to_k_curl, unitary_trans_sub, _ham_k_system, inv_e_d_r, get_deltaU

I_A = np.array([1, 2, 0], dtype=np.int32)
I_B = np.array([2, 0, 1], dtype=np.int32)

@njit(nogil=True)
def get_berry_curvature_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, num_wann, kpt, eta):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    e_d = np.zeros((num_wann,num_wann), dtype=np.float64)
    inv2 = np.zeros((num_wann,num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d[m_, n_] = eig[m_] - eig[n_]
            inv2[m_, n_] = -1.0 / (e_d[m_, n_]*e_d[m_, n_] + eta * eta)
    Ah_k = np.zeros((3,num_wann,num_wann), dtype=np.complex128)
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    for i in range(3):
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        # A^H_a = A_bar^H_a + i D^H_a = i<psi_m| del_a psi_n>
        Ah_k[i] = 1j * e_d * unitary_trans(A_bar_k[i], uu) - unitary_trans(ham_out[i+1], uu)
    omega_k = np.zeros((3, num_wann), dtype=np.float64)
    for n_ in range(num_wann):
        omega_k[0, n_] = np.sum((Ah_k[1, n_, :] * Ah_k[2, :, n_] * inv2[n_, :]).imag)
        omega_k[1, n_] = np.sum((Ah_k[2, n_, :] * Ah_k[0, :, n_] * inv2[n_, :]).imag)
        omega_k[2, n_] = np.sum((Ah_k[0, n_, :] * Ah_k[1, :, n_] * inv2[n_, :]).imag)
    omega_k *= -2.0
    return omega_k, eig


@njit(nogil=True)
def _get_f_omega(Ah_k, f, num_wann):
    fo_k = np.zeros((3, num_wann), dtype=np.float64)
    g = 1.0 - f
    for i in range(3):
        for n_ in range(num_wann):
            fo_k[i, n_] = np.sum(g * (Ah_k[I_A[i], n_, :] * Ah_k[I_B[i], :, n_]).imag)
    fo_k *= -2.0 * f
    return fo_k


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
        of_k = _get_f_omega(Ah_k, Ah_k, f, num_wann)
        # of_k = _get_f_omega_interpolation(f, duu, num_wann)
        morb1, morb2 = _get_morb_k(eig, num_wann, ef, Ah_k, Ah_k, of_k, f, xyz)
        morb_k[ik, 0] = np.sum(morb1)
        morb_k[ik, 1] = np.sum(morb2)
    morb_k[:, 2] = morb_k[:, 0] + morb_k[:, 1]
    return morb_k * fac
