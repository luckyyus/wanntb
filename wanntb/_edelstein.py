import numpy as np
from numba import njit, prange

from ._berry import _get_v_S_k, _get_morb_gmat_k, _get_omega_gmat
from .constant import TwoPi
from .utility import occ_fermi, \
    A_vec

# mapping: I_1 is response (s) component, I_2 is source (v) component
I_1 = np.array((0, 0, 0, 1, 1, 1, 2, 2, 2), dtype=np.int16)
I_2 = np.array((0, 1, 2, 0, 1, 2, 0, 1, 2), dtype=np.int16)


@njit(nogil=True)
def _get_f_spins_edelstein_omega9(eig, inv2, s_k, vh, f, ef, num_wann, eta_intra):
    """Return inter and intra with 9 components (xx,xy,xz,yx,yy,yz,zx,zy,zz)."""

    fe_inter = np.zeros(9, dtype=np.float64)
    fe_intra = np.zeros(9, dtype=np.float64)
    f_ij = np.zeros((num_wann, num_wann), dtype=np.float64)

    dos = A_vec(eig, ef, eta_intra) / TwoPi  # lorentzian ~ -df/dE

    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            f_ij[m_, n_] = f[n_] - f[m_]

    for ic in range(9):
        s_mat = s_k[I_1[ic]]
        v_mat = vh[I_2[ic]]
        # compute inter: -2i sum_{m,n} (f_m - f_n) * inv_e_d2[m,n] * s[a]_{m,n} * v[b]_{n,m}
        inter = 0.0 + 0.0j
        for n_ in range(num_wann):
            inter += np.sum(f_ij[:, n_] * inv2[:, n_] * s_mat[n_, :] * v_mat[:, n_])
        fe_inter[ic] = -2.0 * inter.imag

        # compute intra: sum_n (1/eta) * df_de[n] * s[a]_{n,n} * v[b]_{n,n}
        intra = np.sum((1.0 / eta_intra) * dos * np.diag(s_mat) * np.diag(v_mat))
        fe_intra[ic] = intra.real
    return fe_inter, fe_intra


def _get_f_morb_edelstein_omega3(eig, inv2, morb_gmat_k, vh, f, ef, num_wann, eta_intra):
    fe_inter = np.zeros(3, dtype=np.float64)
    fe_intra = np.zeros(3, dtype=np.float64)

    dos = A_vec(eig, ef, eta_intra) / TwoPi  # lorentzian ~ -df/dE

    for ic in range(3):
        v_mat = vh[ic]
        # compute inter: -2i sum_{m,n} (f_m - f_n) * inv_e_d2[m,n] * s[a]_{m,n} * v[b]_{n,m}
        inter = 0.0 + 0.0j
        for n_ in range(num_wann):
            inter += np.sum(f[n_] * inv2[:, n_] * morb_gmat_k[n_, :] * v_mat[:, n_])
        fe_inter[ic] = -2.0 * inter.imag

        # compute intra: sum_n (1/eta) * df_de[n] * s[a]_{n,n} * v[b]_{n,n}
        intra = np.sum((1.0 / eta_intra) * dos * np.diag(morb_gmat_k) * np.diag(v_mat))
        fe_intra[ic] = intra.real
    return fe_inter, fe_intra


@njit(parallel=True, nogil=True)
def edelstein_fermi(s_or_l, ham_R, r_mat_R, R_vec, R_cartT,
                    num_wann, kpts, efs, eta, eta_intra, xyz=2, ss_R=None, subwf=None):
    """
    输出形状：
      inter_out[n_ef, 9]，intra_out[n_ef, 9]
      SREE 分量顺序为 xx, xy, xz, yx, yy, yz, zx, zy, zz
      OREE 分量顺序为 qx1, qy1, qz1, qx2, qy2, qz2, qx, qy, qz 其中q为l的分量，1为局域，2为巡游
    """
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    _shape = (n_ef, 9, nkpts)
    inter_ks = np.zeros(_shape, dtype=np.float64)
    intra_ks = np.zeros(_shape, dtype=np.float64)

    for ik in prange(nkpts):
        kpt = kpts[ik]
        if s_or_l: # spin REE
            eig, inv_e_d2, ah_k, v_k, s_k = _get_v_S_k(ham_R, r_mat_R, R_vec, R_cartT,
                                                    num_wann, eta, kpt,
                                                    ss_R=ss_R, subwf_S=subwf)
            for i in range(n_ef):
                ef = efs[i]
                f = occ_fermi(eig, ef, eta)
                fse_inter, fse_intra = _get_f_spins_edelstein_omega9(eig, inv_e_d2, s_k, v_k,
                                                                     f, ef, num_wann, eta_intra)
                inter_ks[i, :, ik] = fse_inter
                intra_ks[i, :, ik] = fse_intra
        else: # orbital REE
            eig, inv_e_d2, ah_k, v_k, ah_ak = _get_v_S_k(ham_R, r_mat_R, R_vec, R_cartT,
                                                         num_wann, eta, kpt,
                                                         subwf_A=subwf)
            for i in range(n_ef):
                ef = efs[i]
                f = occ_fermi(eig, ef, eta)
                omega_g = _get_omega_gmat(ah_ak, ah_k, f, num_wann)
                morb1, morb2 = _get_morb_gmat_k(eig, num_wann, ef, ah_ak, ah_k, omega_g, f, xyz)
                foe1_inter, foe1_intra = _get_f_morb_edelstein_omega3(eig, inv_e_d2, morb1, v_k,
                                                                      f, ef, num_wann, eta_intra)
                foe2_inter, foe2_intra = _get_f_morb_edelstein_omega3(eig, inv_e_d2, morb2, v_k,
                                                                      f, ef, num_wann, eta_intra)
                inter_ks[i, 0:3, ik] = foe1_inter
                inter_ks[i, 3:6, ik] = foe2_inter
                intra_ks[i, 0:3, ik] = foe1_intra
                intra_ks[i, 3:6, ik] = foe2_intra

    inter_out = np.sum(inter_ks, axis=2) / nkpts
    intra_out = np.sum(intra_ks, axis=2) / nkpts
    if not s_or_l: #orbital REE
        inter_out[:, 6:9] = inter_out[:, 0:3] + inter_out[:, 3:6]
        intra_out[:, 6:9] = intra_out[:, 0:3] + intra_out[:, 3:6]
    return inter_out, intra_out
