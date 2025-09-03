import numpy as np
from numba import njit, prange
from .constant import TwoPi
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi, \
    unitary_trans_sub, inv_e_d_c, inv_e_d_2, get_delta_E 

@njit(nogil=True)
def _berry_Ah_k_edelstein(ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, se_xyz, ss_R, subwf):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, R_cartT, fac, iout=[1, 2, 3])
    # A_bar^W_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    inv_e_d, e_d = inv_e_d_c(eig, num_wann, eta)
    # for subwf
    se_a = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    va_a = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    
    sw = fourier_R_to_k_vec3(ss_R, fac)
    va_a[se_xyz] = -1.0j * e_d * unitary_trans(A_bar_k[se_xyz], uu) + unitary_trans(ham_out[se_xyz + 1], uu)
    for i in range(3): 
        mat_S = unitary_trans(sw[i], uu) if subwf is None \
            else unitary_trans_sub(sw[i, subwf, :], uu[subwf, :], uu)            
        se_a[i] = mat_S 
    return eig, se_a, va_a

@njit(nogil=True)
def _get_f_spins_edelstein_omega(se_a, va_a, f, eig, ef, num_wann, se_xyz, eta):
    """
    inter 与 intra 两部分：
    - inter: fse_inter[3, num_wann]
    - intra: fse_intra[3, num_wann]
    """
    # fse_inter = np.zeros((3, num_wann), dtype=np.float64)
    # fse_intra = np.zeros((3, num_wann), dtype=np.float64)

    fse_inter = np.zeros((3), dtype=np.complex128)
    fse_intra = np.zeros((3), dtype=np.complex128)
    f_ij =  np.zeros((num_wann, num_wann), dtype=np.complex128)
    inv_e_d2 = inv_e_d_2(eig, num_wann, eta)
    df_de = get_delta_E(eig, ef, eta)  # = 1/eta *-∂f/∂E
    
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            f_ij[m_,n_] = f[m_] - f[n_]

    for i in range(3): 
        fse_inter[i] =  1j * np.sum(np.diag((f_ij * inv_e_d2 * se_a[i]) @ va_a[se_xyz] ))   
        fse_intra[i] =  np.sum(df_de * np.diag(se_a[i]) * np.diag(va_a[se_xyz]))

    return fse_inter, fse_intra

@njit(parallel=True, nogil=True)
def edelstein_fermi(ham_R, r_mat_R, R_vec, R_cartT,
                    num_wann, kpts, efs, eta, se_xyz=0, ss_R=None, subwf=None):
    """
    输出形状：
      inter_out[n_ef, 3]，intra_out[n_ef, 3]，方向依次为 x,y,z
    """
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    inter_ks = np.zeros((n_ef, 3, nkpts), dtype=np.float64)
    intra_ks = np.zeros((n_ef, 3, nkpts), dtype=np.float64)

    for ik in prange(nkpts):
        kpt = kpts[ik]
        eig, se_ak, va_ak = _berry_Ah_k_edelstein(ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, se_xyz, ss_R, subwf)
        for i in range(n_ef):
            ef = efs[i]
            f = occ_fermi(eig, ef, eta)            
            fse_inter, fse_intra = _get_f_spins_edelstein_omega(se_ak, va_ak, f, eig, ef, num_wann, se_xyz, eta)
            inter_ks[i, :, ik] = fse_inter.real
            intra_ks[i, :, ik] = fse_intra.real

    inter_out = np.sum(inter_ks, axis=2) / nkpts
    intra_out = np.sum(intra_ks, axis=2) / nkpts
    return inter_out, intra_out
