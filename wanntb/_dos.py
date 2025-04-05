import numpy as np
from numba import njit, prange
from .utility import fourier_phase_R_to_k, fourier_R_to_k, _ham_k_system, occ_fermi


@njit(parallel=True, nogil=True)
def get_occ_kpar(ham_R, R_vec, R_vec_cart_T, kpts, efs, eta):
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    occ_ef_k = np.zeros((n_ef, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        phase_fac = fourier_phase_R_to_k(R_vec, kpt)
        ham_k = fourier_R_to_k(ham_R, R_vec_cart_T, phase_fac, iout=[5])[0]
        eig = np.linalg.eigvalsh(ham_k)
        for i in range(n_ef):
            ef = efs[i]
            occ = occ_fermi(eig, ef, eta)
            occ_ef_k[i, ik] = np.sum(occ)
    return np.sum(occ_ef_k, axis=1) / nkpts


@njit(parallel=True, nogil=True)
def get_occ_proj_kpar(ham_R, R_vec, R_vec_cart_T, num_wann, kpts, efs, eta):
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    occ_ef_k = np.zeros((n_ef, num_wann, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        # uu[iproj, iband]
        ham_k, eig, uu = _ham_k_system(ham_R, R_vec, R_vec_cart_T, kpt)
        # proj[iproj, iband]
        proj = np.abs(uu)**2
        for i in range(n_ef):
            ef = efs[i]
            occ = occ_fermi(eig, ef, eta)
            # occ_proj[iproj, iband]
            occ_proj = proj * occ
            occ_ef_k[i, :, ik] = np.sum(occ_proj, axis=1)
    return np.sum(occ_ef_k, axis=2) / nkpts