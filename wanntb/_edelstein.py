import numpy as np
from numba import njit, prange
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi, \
    unitary_trans_sub, inv_e_d_c, inv_e_d_2, get_delta_E, dos_fermi 

@njit(nogil=True)
def _berry_vh_ssh_k_edelstein(ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, ss_R, subwf):
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
    for i in range(3): 
        va_a[i] = -1.0j * e_d * unitary_trans(A_bar_k[i], uu) + unitary_trans(ham_out[i + 1], uu)
        mat_S = unitary_trans(sw[i], uu) if subwf is None \
            else unitary_trans_sub(sw[i, subwf, :], uu[subwf, :], uu)            
        se_a[i] = (mat_S + mat_S.T.conj())* 0.5
    return eig, se_a, va_a

# @njit(nogil=True)
# def _get_f_spins_edelstein_omega9(se_a, va_a, f, eig, ef, num_wann, eta):
#     """Return inter and intra with 9 components (xx,xy,xz,yx,yy,yz,zx,zy,zz).
#     """
#     # mapping: index1 selects s component, index2 selects v component
#     index1 = (0,0,0,1,1,1,2,2,2)
#     index2 = (0,1,2,0,1,2,0,1,2)
#     ncomp = 9
#     fse_inter = np.zeros((ncomp,), dtype=np.complex128)
#     fse_intra = np.zeros((ncomp,), dtype=np.complex128)
#
#     # f_ij = f[m] - f[n]
#     f_ij = np.zeros((num_wann, num_wann), dtype=np.complex128)
#     for m_ in range(num_wann):
#         for n_ in range(num_wann):
#             if m_ == n_:
#                 continue
#             f_ij[m_, n_] = f[n_] - f[m_]
#
#     # inv_e_d2 used as the elementwise factor (== 1/((e_n-e_m)^2 + eta^2))
#     inv_e_d2 = inv_e_d_2(eig, num_wann, eta)
#     df_de = get_delta_E(eig, ef, eta)
#
#     for ic in range(ncomp):
#         a = index1[ic]
#         b = index2[ic]
#         fse_inter[ic] = 1j * np.sum(np.diag((f_ij * inv_e_d2 * se_a[a]) @ va_a[b]))
#         fse_intra[ic] = np.sum(df_de * (np.diag(se_a[a]) * np.diag(va_a[b])) / eta)
#
#     return fse_inter, fse_intra


@njit(nogil=True)
def _get_f_spins_edelstein_omega9(se_a, va_a, f, eig, ef, num_wann, eta, eta_intra):
    """Return inter and intra with 9 components (xx,xy,xz,yx,yy,yz,zx,zy,zz)."""
    # mapping: index1 selects s component, index2 selects v component
    index1 = (0,0,0,1,1,1,2,2,2)
    index2 = (0,1,2,0,1,2,0,1,2)
    ncomp = 9
    fse_inter = np.zeros((ncomp,), dtype=np.complex128)
    fse_intra = np.zeros((ncomp,), dtype=np.complex128)
    f_ij = np.zeros((num_wann, num_wann), dtype=np.complex128)
    inv_e_d2 = inv_e_d_2(eig, num_wann, eta)
    df_de = get_delta_E(eig, ef, eta_intra)  # lorentzian ~ -df/dE
    # df_de = dos_fermi(eig, ef, eta)

    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            f_ij[m_, n_] = f[n_] - f[m_]
    # compute inter: sum_{m,n} (f_m - f_n) * inv_e_d2[m,n] * s[a]_{m,n} * v[b]_{n,m}
    for ic in range(ncomp):
        a = index1[ic]
        b = index2[ic]
        s_mat = se_a[a]
        v_mat = va_a[b]
        total = 0.0 + 0.0j
        for m_ in range(num_wann):
            for n_ in range(num_wann):
                if m_ == n_:
                    continue
                total += f_ij[m_, n_] * inv_e_d2[m_, n_] * s_mat[m_, n_] * v_mat[n_, m_]
        fse_inter[ic] = 1j * total
    # compute intra: sum_n (1/eta) * df_de[n] * s[a]_{n,n} * v[b]_{n,n}
    for ic in range(ncomp):
        a = index1[ic]
        b = index2[ic]
        # take diagonals
        total = 0.0 + 0.0j
        for n_ in range(num_wann):
            total += (1.0/eta_intra) * df_de[n_] * se_a[a][n_, n_] * va_a[b][n_, n_]
        fse_intra[ic] = total
    return fse_inter, fse_intra


@njit(parallel=True, nogil=True)
def edelstein_fermi(ham_R, r_mat_R, R_vec, R_cartT,
                    num_wann, kpts, efs, eta, eta_intra, ss_R=None, subwf=None):
    """
    输出形状：
      inter_out[n_ef, 9]，intra_out[n_ef, 9]，分量顺序为 xx,xy,xz,yx,yy,yz,zx,zy,zz
    """
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    inter_ks = np.zeros((n_ef, 9, nkpts), dtype=np.float64)
    intra_ks = np.zeros((n_ef, 9, nkpts), dtype=np.float64)

    for ik in prange(nkpts):
        kpt = kpts[ik]
        eig, se_ak, va_ak = _berry_vh_ssh_k_edelstein(ham_R, r_mat_R, R_vec, R_cartT, num_wann, eta, kpt, ss_R, subwf)
        for i in range(n_ef):
            ef = efs[i]
            f = occ_fermi(eig, ef, eta)
            fse_inter, fse_intra = _get_f_spins_edelstein_omega9(se_ak, va_ak, f, eig, ef, num_wann, eta, eta_intra)
            inter_ks[i, :, ik] = 2 * fse_inter.real
            intra_ks[i, :, ik] = 2 * fse_intra.real

    inter_out = np.sum(inter_ks, axis=2) / nkpts
    intra_out = np.sum(intra_ks, axis=2) / nkpts
    return inter_out, intra_out
