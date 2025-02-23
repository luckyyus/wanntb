import numpy as np
from math import pi, exp, sin, cos
import yaml
import pkgutil
from numba import njit, prange, complex128, cuda


TwoPi = 2.0 * pi

H_ = 4.1356676969E-15  # Planck constant h in eV*s
Hbar_ = H_ / TwoPi # hbar in eV.s

Orbitals: dict = yaml.safe_load(pkgutil.get_data(__package__, 'orbitals.yml').decode('utf-8'))

Eta = 1.0E-4

Cart = np.array([[1.0, 0.0, 0.0],
                 [0.0, 1.0, 0.0],
                 [0.0, 0.0, 1.0]], dtype=float)


def get_list_index(item, li):
    for index, value in enumerate(li):
        if value == item:
            return index
    return -1


@njit
def fermi(e, kbT):
    return 1.0 / (exp(e/kbT) + 1.0)


@njit
def _surface_GR(energy, n_dim, h0, t, mu=0.0, n_iter=25):
    """
    计算表面格林函数
    :param energy: 入射能量
    :param n_dim: 电极维度
    :param h0: 单层哈密顿矩阵
    :param t: 跃迁哈密顿矩阵
    :param mu: 化学势
    :param n_iter: 迭代次数
    :return: [n_dim, n_dim] complex
    """
    a0 = t
    b0 = np.ascontiguousarray(t.T.conjugate())
    e0 = h0 - np.eye(n_dim) * (1j * Eta - mu)
    es = e0.copy()
    for i in range(n_iter):
        # if self.max_z == 1:  # 目前就做了最近邻，其他的先空着
        gR0 = np.linalg.inv(-e0 + np.eye(n_dim) * energy)
        es += a0.dot(gR0).dot(b0)
        e0 += a0.dot(gR0).dot(b0) + b0.dot(gR0).dot(a0)
        a1 = a0.dot(gR0).dot(a0)
        b1 = b0.dot(gR0).dot(b0)
        a0 = a1
        b0 = b1
    gR0 = np.linalg.inv(-es + np.eye(n_dim) * energy)
    return gR0


@njit
def _matrix_add_by_index(target, xlist, ylist, add):
    nx = xlist.shape[0]
    ny = ylist.shape[0]
    assert add.shape[0] == nx and add.shape[1] == ny, 'add.shape == (len(xlist), len(ylist))'
    for ix in range(nx):
        for iy in range(ny):
            target[xlist[ix], ylist[iy]] += add[ix, iy]
    pass


@njit
def _get_submatrix_by_index(target, xlist, ylist):
    nx = int(len(xlist))
    ny = int(len(ylist))
    sub = np.zeros((nx, ny), dtype=complex128)
    for ix in range(nx):
        for iy in range(ny):
            sub[ix, iy] = target[xlist[ix], ylist[iy]]
    return sub


# 不用了，废弃
@njit
def transmission_jit(num_wann, ham, li_lc, li_rc,
                     energy, sigmaRl, sigmaRr, gamma_l, gamma_r):
    _matrix_add_by_index(ham, li_lc, li_lc, sigmaRl)
    _matrix_add_by_index(ham, li_rc, li_rc, sigmaRr)
    # 总的推迟格林函数和超越格林函数
    gR = np.linalg.inv(np.eye(num_wann, dtype=complex128) * energy - ham)
    gA = gR.T.conjugate()
    gR_lr = _get_submatrix_by_index(gR, li_lc, li_rc)
    gA_rl = _get_submatrix_by_index(gA, li_rc, li_lc)
    # tr(gamma_l.gR_lr.gamma_r.gA_rl)
    return np.trace(np.dot(np.dot(np.dot(gamma_l, gR_lr), gamma_r), gA_rl)).real


@njit(nogil=True)
def _get_ham_k2d(num_wann, ham_R, r_vec, n_degen, k_2d, efermi):
    # ham_k = np.zeros((num_wann, num_wann), dtype=complex128)
    rdotk = r_vec[:, 0:2] @ k_2d * TwoPi
    phase_fac = (np.cos(rdotk) + 1j * np.sin(rdotk)) / n_degen
    ham_k = phase_fac @ ham_R
    # for ir in range(n_rpts):
    #     # 只有Rz为0的H_R是需要考虑的
    #     # if R_vec[ir, 2] == 0:
    #     rdotk = TwoPi * np.dot(r_vec[ir, 0:2], k_2d)
    #     ham_k += ham_R[ir, :, :] * complex(np.cos(rdotk), np.sin(rdotk)) / n_degen[ir]
    return ham_k - np.eye(num_wann, dtype=complex128) * efermi


@njit(parallel=True, nogil=True)
def transmission_k(num_wann, ham_R, R_vec, n_degen, efermi,
                   li_lc, li_rc, sRl, sRr, gm_l, gm_r, n_e, e_list, nkpt,
                   kpts):
    # 每个k点的透射系数数组
    trans_k = np.zeros((n_e, nkpt), dtype=float)
    for ik in prange(nkpt):
        ham_k = _get_ham_k2d(num_wann, ham_R, R_vec, n_degen, kpts[ik, :], efermi)
        ham_k2 = np.zeros((num_wann, num_wann), dtype=complex128)
        for ie in range(n_e):
            ham_k2[:, :] = ham_k.copy()
            _matrix_add_by_index(ham_k2, li_lc, li_lc, sRl[ie, :, :])
            _matrix_add_by_index(ham_k2, li_rc, li_rc, sRr[ie, :, :])
            # 总的推迟格林函数和超越格林函数
            gR = np.linalg.inv(np.eye(num_wann, dtype=complex128) * e_list[ie] - ham_k2)
            gA = gR.T.conjugate()
            gR_lr = _get_submatrix_by_index(gR, li_lc, li_rc)
            gA_rl = _get_submatrix_by_index(gA, li_rc, li_lc)
            tt = np.dot(np.dot(np.dot(gm_l[ie, :, :], gR_lr), gm_r[ie, :, :]), gA_rl)
            trans_k[ie, ik] = np.trace(tt).real

    return np.sum(trans_k, axis=1) / nkpt


@njit(parallel=True, nogil=True)
def get_dos_e(num_wann, ham_R, R_vec, n_degen, efermi, n_e, e_list, nkpt, kpts):
    dos_k = np.zeros((n_e, nkpt), dtype=float)
    for ik in prange(nkpt):
        ham_k = _get_ham_k2d(num_wann, ham_R, R_vec, n_degen, kpts[ik, :], efermi)
        for ie in range(n_e):
            gR = np.linalg.inv(np.eye(num_wann, dtype=complex128) * (e_list[ie] + 1j * Eta) - ham_k)
            dos_k[ie, ik] = np.trace(gR).imag
    return - np.sum(dos_k, axis=1) / pi / nkpt


@njit(parallel=True, nogil=True)
def get_self_energies(l_h0, l_t, l_dim,
                      r_h0, r_t, r_dim,
                      n_e, e_list,
                      mu_l, mu_r, v_lc, v_rc, n_iter=25):
    n_lc = v_lc.shape[1]
    n_rc = v_rc.shape[1]
    sRl = np.zeros((n_e, n_lc, n_lc), dtype=complex128)
    sRr = np.zeros((n_e, n_rc, n_rc), dtype=complex128)
    gm_l = np.zeros((n_e, n_lc, n_lc), dtype=complex128)
    gm_r = np.zeros((n_e, n_rc, n_rc), dtype=complex128)
    for ie in prange(n_e):
        gsRl = _surface_GR(e_list[ie], l_dim, l_h0, l_t, mu_l, n_iter)
        gsRr = _surface_GR(e_list[ie], r_dim, r_h0, r_t, mu_r, n_iter)
        sRl[ie, :, :] = np.ascontiguousarray(v_lc.T.conjugate()).dot(gsRl).dot(v_lc)
        sRr[ie, :, :] = np.ascontiguousarray(v_rc.T.conjugate()).dot(gsRr).dot(v_rc)
        gm_l[ie, :, :] = (sRl[ie, :, :] - sRl[ie, :, :].T.conjugate()) * 1j
        gm_r[ie, :, :] = (sRr[ie, :, :] - sRr[ie, :, :].T.conjugate()) * 1j
    return sRl, sRr, gm_l, gm_r


@njit
def unitary_trans(mat, uu):
    return uu.conj().T @ mat @ uu


def read_tb_file(tb_file='wannier90_tb.dat'):
    seedname = tb_file.split('/')[-1].split('_')[0]
    # read tb file
    with open(tb_file, 'r') as f:
        line = f.readline()
        print("reading tb file %s ( %s )" % (tb_file, line.strip()))
        # real lattice
        real_lattice = np.array([f.readline().split()[:3] for i in range(3)], dtype=np.float64)
        print('real lattice:')
        print(real_lattice)
        recip_lattice = np.linalg.inv(real_lattice).T * TwoPi
        print('reciprocal lattice:')
        print(recip_lattice)
        num_wann = int(f.readline())
        n_Rpts = int(f.readline())
        # degenerate rpt
        ndegen = []
        while len(ndegen) < n_Rpts:
            ndegen += f.readline().split()
        n_degen = np.array(ndegen, dtype=np.float64)
        # initialize R_vec and Ham_R
        irpt = []
        ham_R = np.zeros((n_Rpts, num_wann, num_wann), dtype=np.complex128)
        # read each R_vec[nRvec0, 3] and Ham_R[n_Rpts, num_wann, num_wann]
        for ir in range(n_Rpts):
            f.readline()
            irpt.append(f.readline().split())
            hh = np.array(
                [[f.readline().split()[2:4] for n in range(num_wann)] for m in range(num_wann)],
                dtype=float).transpose((2, 0, 1))
            ham_R[ir, :, :] = (hh[0, :, :] + 1j * hh[1, :, :])
        R_vec = np.ascontiguousarray(irpt, dtype=float)  # 为了和k.R正常一些
        ham_R = np.ascontiguousarray(ham_R)
        print('ham_R: %s %s' % (ham_R.dtype, list(ham_R.shape)))
        print('R_vec: %s %s' % (R_vec.dtype, list(R_vec.shape)))
        print('n_degen: %s %s' % (n_degen.dtype, list(n_degen.shape)))

        r_mat_R = np.zeros((n_Rpts, 3, num_wann, num_wann), dtype=np.complex128)
        for ir in range(n_Rpts):
            f.readline()
            assert (np.array(f.readline().split(), dtype=int) == R_vec[ir]).all()
            aa = np.array([[f.readline().split()[2:8]
                            for n in range(num_wann)]
                           for m in range(num_wann)], dtype=float)
            r_mat_R[ir, :, :, :] = (aa[:,:,0::2] + 1j*aa[:,:,1::2]).transpose((2,1,0)) / n_degen[ir]
        print('r_mat_R: %s %s' % (r_mat_R.dtype, list(r_mat_R.shape)))
    # real_lattice[3,3] float
    # recip_lattice[3,3] float
    # ham_R[n_Rpts, num_wann, num_wann] complex
    # R_vec[n_Rpts, 3] float
    # n_degen[n_Rpts] int
    # r_mat_R[n_Rpts, 3, num_wann, num_wann] complex
    return {'seedname': seedname,
            'num_wann': num_wann,
            'real_lattice': real_lattice,
            'recip_lattice': recip_lattice,
            'n_Rpts': n_Rpts,
            'ham_R': ham_R,
            'R_vec': R_vec,
            'n_degen': n_degen,
            'r_mat_R': r_mat_R}


@njit('complex128[:](float64[:,:], float64[:], float64[:])', nogil=True)
def fourier_phase_R_to_k(R_vec, n_degen, kpt):
    """
    计算从R表象到k表象的相因子, [n_Rpt]
    """
    rdotk = R_vec @ kpt * TwoPi
    return (np.cos(rdotk) + 1j * np.sin(rdotk)) / n_degen


@njit(nogil=True)
def fourier_R_to_k(mat_R, R_vec_cart_T, phase_fac, iout=[0]):
    """
    把一个厄米矩阵从R空间转变到k空间
    @param mat_R: R空间的厄米矩阵，单位 eV
    @param R_vec_cart_T: R格点的xyz坐标，单位 angst.，数组排列转置
    @param phase_fac: 各个R的相因子
    @param iout: 输出选项tuple，0代表mat_k, 1~3代表xyz三个方向的dmat/dk
    """
    n_rpt, num_wann, _ = mat_R.shape
    output = np.zeros((4, num_wann, num_wann), dtype=complex128)
    # for ir in range(n_rpt):
    #     output[0] += mat_R[ir, :, :] * phase_fac[ir]
    #     if 1 in iout:
    #         output[1] = R_vec_cart_T[0, ir] * phase_fac[ir] * mat_R[ir, :, :] * 1j
    # output[0] = phase_fac @ mat_R
    # if 1 in iout:
    #     output[1] = (R_vec_cart_T[0] * phase_fac) @ mat_R * 1j
    # if 2 in iout:
    #     output[2] = (R_vec_cart_T[1] * phase_fac) @ mat_R * 1j
    # if 3 in iout:
    #     output[3] = (R_vec_cart_T[2] * phase_fac) @ mat_R * 1j
    for i in range(num_wann):
        for j in range(num_wann):
            mat_Rij = np.ascontiguousarray(mat_R[:, i, j])
            output[0, i, j] = np.dot(mat_Rij, phase_fac)
            if 1 in iout:
                output[1, i, j] = np.sum(mat_Rij * R_vec_cart_T[0] * phase_fac) * 1j
            if 2 in iout:
                output[2, i, j] = np.sum(mat_Rij * R_vec_cart_T[1] * phase_fac) * 1j
            if 3 in iout:
                output[3, i, j] = np.sum(mat_Rij * R_vec_cart_T[2] * phase_fac) * 1j
    return output


@njit(nogil=True)
def fourier_R_to_k_vec3(vec_mat_R, phase_fac):
    n_rpt, _, num_wann, _ = vec_mat_R.shape
    oo_true = np.zeros((3, num_wann, num_wann), dtype=complex128)
    for k in prange(3):
        for i in prange(num_wann):
            for j in prange(num_wann):
                vec_mat_Rkij = np.ascontiguousarray(vec_mat_R[:, k, i, j])
                oo_true[k, i, j] = np.dot(vec_mat_Rkij, phase_fac)
    return oo_true


@njit(nogil=True)
def get_eig_da(eig, ham_da, uu, num_wann, eig_diff=1e-4):
    eig_da = np.zeros(num_wann, dtype=float)
    ham_bar_da = unitary_trans(ham_da, uu)
    i = 0
    while i < num_wann:
        diff = eig[i + 1] - eig[i] if i + 1 < num_wann else 1.0
        if diff < eig_diff:
            degen_min = i
            degen_max = i + 1
            while degen_max + 1 < num_wann:
                diff = eig[degen_max + 1] - eig[degen_min]
                if diff < eig_diff:
                    degen_max += 1
                else:
                    break
            eig_da[degen_min:degen_max] \
                = np.linalg.eigvalsh(ham_bar_da[degen_min:degen_max,degen_min:degen_max])
            i = degen_max
        else:
            eig_da[i] = ham_bar_da[i, i].real
        i += 1
    return eig_da


@njit(nogil=True)
def _ham_k_system(ham_R, R_vec, R_vec_cart_T, n_degen, num_wann, kpt, direction):
    """
    计算一个k点的哈密顿量、本征值、征值随k的某个方向的导数和本征态。
    @param ham_R: R空间的紧束缚哈密顿量
    @param R_vec: R格点坐标
    @param R_vec_cart_T: R格点的xyz坐标，数组排列转置
    @param n_degen: R格点简并度
    @param num_wann: WF空间大小
    @param direction: 方向，123分别代表xyz
    @param kpt: k点坐标，倒格矢表象
    @return: k空间哈密顿量，本征值，本征值随k的导数，本征态
    """
    fac = fourier_phase_R_to_k(R_vec, n_degen, kpt)
    out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[0, direction])
    ham_k, ham_k_da = out[0], out[direction]
    eig, uu = np.linalg.eigh(ham_k)
    eig_da = get_eig_da(eig, ham_k_da, uu, num_wann)
    return ham_k, eig, eig_da, uu


# def get_spin_splitting(ham, num_wann):
#     onsite = np.diagonal(ham).real
#     return (- onsite[0:num_wann//2] + onsite[num_wann//2:num_wann]) / 2


@njit('float64(float64, float64, float64)', nogil=True)
def A_n(eig_n, ef, eta):
    de = eig_n - ef
    return eta / (de * de + eta * eta / 4)


@njit(nogil=True)
def A_vec(eig, ef, eta):
    de = eig - ef
    return eta / (de * de + eta * eta / 4)


@njit(nogil=True)
def occ_fermi(eig, ef, eta):
    return 1.0 / (np.exp((eig - ef)/eta) + 1)

# @njit(parallel=True)
# def Sp2_mn(e_s, um, un, num_wann):
#     """
#     计算|<um|S+.delta_0|un>|^2，单位 eV^2
#     @param e_s: real[num_wann/2] 各轨道在位自旋劈裂能
#     @param um: complex[num_wann]左矢
#     @param un: complex[num_wann]右矢
#     @param num_wann: WF空间维度
#     @return: real[num_wann, num_wann] |S+.delta_0|^2[n_, m_] 矩阵
#     """
#     sp2 = np.zeros((num_wann, num_wann), dtype=float)
#     half = num_wann // 2
#     for n_ in prange(num_wann):
#         un_dn = np.ascontiguousarray(un[half:num_wann, n_])
#         for m_ in prange(num_wann):
#             um_up = np.ascontiguousarray(um[0:half, m_].conj())
#             sp2[n_, m_] = abs(np.dot(um_up, un_dn * e_s))**2
#     return sp2

# @njit(parallel=True)
# def sz_n(uu, num_wann: int):
#     """
#     计算 < un | Sz | un > 无量纲
#     @param uu:
#     @param num_wann:
#     @return:
#     """
#     half = num_wann // 2
#     sz = np.zeros(num_wann, dtype=float)
#     for n_ in prange(num_wann):
#         up = np.dot(uu[0:half, n_].conj(), uu[0:half, n_]).real
#         dn = np.dot(uu[half:num_wann, n_].conj(), uu[half:num_wann, n_]).real
#         sz[n_] = up - dn
#     return sz

@njit
def guess(x):
    return 1.0 / (2.0 + exp(-x) + exp(+x))


@njit(nogil=True)
def _get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q, num_wann: int, ef: float, eta: float, q: float):
    half = num_wann // 2
    alpha_k = np.zeros(num_wann, dtype=np.float64)
    alpha_qvd_k = np.zeros(num_wann, dtype=np.float64)
    qvs_k = np.zeros(num_wann, dtype=np.float64)
    alpha_k_n = np.zeros(num_wann, dtype=np.float64)
    alpha_qvd_k_n = np.zeros(num_wann, dtype=np.float64)
    qvs_k_n = np.zeros(num_wann, dtype=np.float64)
    for n_ in prange(num_wann):
        if abs(eig[n_] - ef) > eta * 10.0:
            continue
        # un_up = np.ascontiguousarray(uu[0:half, n_])
        un_dn = np.ascontiguousarray(uu[half:num_wann, n_])
        # sz = np.dot(un_up.conj(), un_up).real - np.dot(un_dn.conj(), un_dn).real
        An = A_n(eig[n_], ef, eta)
        alpha_k_n[:] = 0.0
        alpha_qvd_k_n[:] = 0.0
        qvs_k_n[:] = 0.0
        for m_ in prange(num_wann):
            um_up = np.ascontiguousarray(uu_q[0:half, m_])
            sp2_q = abs(np.dot(um_up.conj(), un_dn * e_s))**2
            Am = A_n(eig_q[m_], ef, eta)
            # ww in units eV^-2
            ww = An * Am
            # alpha_k in units 1
            alpha_k_n[m_] = sp2_q * ww
            # qvd in units eV angst.
            alpha_qvd_k_n[m_] = sp2_q * ww * (eig_q_da[m_] - eig_da[n_])
            # qvs in units eV angst.
            if n_ != m_:
                d_eig_mn = eig_q[m_] - eig[n_]
                if abs(d_eig_mn) < 1E-7:
                    qvs_k_n[m_] = - sp2_q * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                             / (d_eig_mn - 1E-7j)).real
                else:
                    qvs_k_n[m_] = - sp2_q * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                         / d_eig_mn)
            else:
                # qvs_k_n[m_] = eta * q * ww * eig_da[n_] * eig_da[n_] * sz
                # qvs_k_n[m_] = - sp2_q * (eig_q_da[n_] - eig_da[n_]) * An / (eig_q[n_] - eig[n_])
                #             = - sp2_q * (eig_q_da[n_] - eig_da[n_]) * An / ( q * eig_da[n_])
                #             = - sp2_q * (1 + q * eig_dda[n_]/eig_da[n_] -1) * An / q
                #             = - sp2_q * (eig_dda[n_]/eig_da[n_]) * An
                if abs(eig_da[n_] * q) < 1E-7:
                    qvs_k_n[m_] = - sp2_q * ((eig_q_da[n_] * Am - eig_da[n_] * An)
                                             / ( q * eig_da[n_] - 1E-7j)).real
                else:
                    qvs_k_n[m_] = - sp2_q * (eig_q_da[n_] / eig_da[n_] * Am - An) / q
        alpha_k[n_] = np.sum(alpha_k_n)
        alpha_qvd_k[n_] = np.sum(alpha_qvd_k_n)
        qvs_k[n_] = np.sum(qvs_k_n)
    return np.sum(alpha_k), np.sum(alpha_qvd_k), np.sum(qvs_k)


@njit(parallel=True, nogil=True)
def get_alpha_beta_kpar(ham_R, R_vec, R_vec_cart_T, n_degen,
                        num_wann, direction, e_s, kpts, q_frac, q, ef, eta):
    nkpts = kpts.shape[0]
    o_k = np.zeros((nkpts, 3), dtype=float)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        ham_k, eig, eig_da, uu = _ham_k_system(ham_R, R_vec, R_vec_cart_T, n_degen,
                                               num_wann, kpt, direction)
        # k + q
        ham_q, eig_q, eig_q_da, uu_q = _ham_k_system(ham_R, R_vec, R_vec_cart_T, n_degen,
                                                     num_wann, kpt + q_frac, direction)
        o_k[ik, :] = _get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q,
                                       num_wann, ef, eta, q)
    return np.sum(o_k, axis=0) / nkpts


@njit(nogil=True)
def _get_alpha_beta_inter_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q, num_wann: int, ef: float, eta: float):
    half = num_wann // 2
    alpha_k = np.zeros(num_wann, dtype=np.float64)
    alpha_qvd_k = np.zeros(num_wann, dtype=np.float64)
    qvs_k = np.zeros(num_wann, dtype=np.float64)
    alpha_k_n = np.zeros(num_wann, dtype=np.float64)
    alpha_qvd_k_n = np.zeros(num_wann, dtype=np.float64)
    qvs_k_n = np.zeros(num_wann, dtype=np.float64)
    for n_ in prange(num_wann):
        if abs(eig[n_] - ef) > eta * 10.0:
            continue
        # un_up = np.ascontiguousarray(uu[0:half, n_])
        un_dn = np.ascontiguousarray(uu[half:num_wann, n_])
        # sz = np.dot(un_up.conj(), un_up).real - np.dot(un_dn.conj(), un_dn).real
        An = A_n(eig[n_], ef, eta)
        alpha_k_n[:] = 0.0
        alpha_qvd_k_n[:] = 0.0
        qvs_k_n[:] = 0.0
        for m_ in prange(num_wann):
            if n_ == m_:
                continue
            um_up = np.ascontiguousarray(uu_q[0:half, m_])
            sp2_q = abs(np.dot(um_up.conj(), un_dn * e_s)) ** 2
            Am = A_n(eig_q[m_], ef, eta)
            # ww in units eV^-2
            ww = An * Am
            # alpha_k in units 1
            alpha_k_n[m_] = sp2_q * ww
            # qvd in units eV angst.
            alpha_qvd_k_n[m_] = sp2_q * ww * (eig_q_da[m_] - eig_da[n_])
            # qvs in units eV angst.
            d_eig_mn = eig_q[m_] - eig[n_]
            if abs(d_eig_mn) < 1E-7:
                qvs_k_n[m_] = - sp2_q * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                         / (d_eig_mn - 1E-7j)).real
            else:
                qvs_k_n[m_] = - sp2_q * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                         / d_eig_mn)

        alpha_k[n_] = np.sum(alpha_k_n)
        alpha_qvd_k[n_] = np.sum(alpha_qvd_k_n)
        qvs_k[n_] = np.sum(qvs_k_n)
    return np.sum(alpha_k), np.sum(alpha_qvd_k), np.sum(qvs_k)


@njit(parallel=True, nogil=True)
def get_alpha_beta_kpar_kpath(ham_R, R_vec, R_vec_cart_T, n_degen,
                              num_wann, direction, e_s, kpts, q_frac, q, ef, eta):
    nkpts = kpts.shape[0]
    o_k = np.zeros((nkpts, 6), dtype=float)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        ham_k, eig, eig_da, uu = _ham_k_system(ham_R, R_vec, R_vec_cart_T, n_degen,
                                               num_wann, kpt, direction)
        # k + q
        ham_q, eig_q, eig_q_da, uu_q = _ham_k_system(ham_R, R_vec, R_vec_cart_T, n_degen,
                                                     num_wann, kpt + q_frac, direction)
        o_k[ik, 0:3] = _get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q,
                                         num_wann, ef, eta, q)
        o_k[ik, 3:6] = _get_alpha_beta_inter_k(eig, eig_q, eig_da, eig_q_da, e_s, uu, uu_q,
                                               num_wann, ef, eta)
    return o_k


@njit(nogil=True)
def get_berry_curvature_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, n_degen,
                          num_wann, kpt):
    """
    berry curvature for on kpoint in unit angst.^2
    @param ham_R:
    @param r_mat_R:
    @param R_vec:
    @param R_vec_cart_T:
    @param n_degen:
    @param num_wann:
    @param kpt:
    @return:
    """
    fac = fourier_phase_R_to_k(R_vec, n_degen, kpt)
    ham_out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[0, 1, 2, 3])
    eig, uu = np.linalg.eigh(ham_out[0])

    inv_e_d = np.zeros((num_wann,num_wann), dtype=np.float64)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            inv_e_d[m_, n_] = 1.0 / (eig[m_] - eig[n_]) if abs(eig[m_] - eig[n_]) > 1e-6 else 0.0

    Ah_k = np.zeros((3,num_wann,num_wann), dtype=complex128)
    # A_bar^H_a[3, num_wann, num_wann] in units angst.
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    for i in range(3):
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        # A^H_a = A_bar^H_a + i D^H_a = <psi_m| del_a psi_n>
        Ah_k[i] = unitary_trans(A_bar_k[i], uu) + 1j * unitary_trans(ham_out[i+1], uu) * inv_e_d

    omega_k = np.zeros((3, num_wann), dtype=np.float64)
    for n_ in range(num_wann):
        omega_k[0, n_] = np.dot(Ah_k[1, n_, :], Ah_k[2, :, n_]).imag
        omega_k[1, n_] = np.dot(Ah_k[2, n_, :], Ah_k[0, :, n_]).imag
        omega_k[2, n_] = np.dot(Ah_k[0, n_, :], Ah_k[1, :, n_]).imag
    omega_k *= -2.0
    return omega_k, eig


@njit(parallel=True, nogil=True)
def get_ahc_kpar_fermi(ham_R, r_mat_R, R_vec, R_vec_cart_T, n_degen,
                       num_wann, kpts, efs, eta):
    nkpts = kpts.shape[0]
    n_ef = efs.shape[0]
    list_o_ef_k = np.zeros((3, n_ef, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        omega, eig = get_berry_curvature_k(ham_R, r_mat_R, R_vec, R_vec_cart_T, n_degen,
                                           num_wann, kpt)
        for i in range(n_ef):
            ef = efs[i]
            occ = occ_fermi(eig, ef, eta)
            list_o_ef_k[:, i, ik] = omega @ occ
    return (np.sum(list_o_ef_k, axis=2) / nkpts).T


@njit(parallel=True, nogil=True)
def get_carrier_kpar(ham_R, R_vec, R_vec_cart_T, n_degen,
                     num_wann, direction, kpts, q_frac, q, ef, eta):
    nkpts = kpts.shape[0]
    list_o_k = np.zeros(nkpts, dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        ham_k, eig, eig_da, uu = _ham_k_system(ham_R, R_vec, R_vec_cart_T, n_degen,
                                               num_wann, kpt, direction)
        # k + q
        ham_q, eig_q, eig_q_da, uu_q = _ham_k_system(ham_R, R_vec, R_vec_cart_T, n_degen,
                                                     num_wann, kpt + q_frac, direction)
        eig_dd_inv = (q / (eig_q_da - eig_da - 1j * eta/q)).real
        n_eig_ef = A_vec(eig, ef, eta)
        list_o_k[ik] = np.sum(n_eig_ef * eig_dd_inv * eig_da * eig_da)
    return np.sum(list_o_k) / (nkpts * TwoPi * TwoPi)


def get_kpts_mesh(kmesh):
    k1, k2, k3 = np.meshgrid(np.arange(kmesh[0], dtype=float)/kmesh[0],
                             np.arange(kmesh[1], dtype=float)/kmesh[1],
                             np.arange(kmesh[2], dtype=float)/kmesh[2], indexing='ij')
    return np.column_stack((k1.ravel(), k2.ravel(), k3.ravel()))


def get_kpts_mesh_around(kmesh, center, distance_cart, recip_lattice):
    k1, k2, k3 = np.meshgrid(np.arange(kmesh[0], dtype=float) / kmesh[0],
                             np.arange(kmesh[1], dtype=float) / kmesh[1],
                             np.arange(kmesh[2], dtype=float) / kmesh[2], indexing='ij')
    kpts0 = np.column_stack((k1.ravel(), k2.ravel(), k3.ravel()))
    nk = kpts0.shape[0]
    kpts = []
    for ik in range(nk):
        kpt = kpts0[ik] - np.array([0.5, 0.5, 0.5]) + center
        dk_cart = (kpt - center) @ recip_lattice
        r_cart = dk_cart / distance_cart
        if r_cart.dot(r_cart) < 1:
            kpts.append(kpt)
    return np.array(kpts)


def get_kpts_path(kpath, nkpts_path, recip_lattice):
    npath = len(kpath) - 1
    kbegin = 0.0
    kpts = []
    kpts_len = []
    for ip in range(npath):
        kdelta = (kpath[ip+1] - kpath[ip]) @ recip_lattice
        klen = np.sqrt(np.dot(kdelta, kdelta))
        for il in range(nkpts_path + 1):
            kpt = ((nkpts_path - il)* kpath[ip] + il * kpath[ip+1]) / nkpts_path
            kpt_len = klen * il / nkpts_path
            kpts.append(kpt)
            kpts_len.append(kpt_len + kbegin)
        kbegin += klen
    return np.array(kpts, dtype=float), np.array(kpts_len, dtype=float)

