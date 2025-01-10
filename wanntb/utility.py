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


@njit
def _get_ham_k2d(num_wann, ham_R, n_rpts, r_vec, n_degen, k_2d, efermi):
    ham_k = np.zeros((num_wann, num_wann), dtype=complex128)
    for ir in range(n_rpts):
        # 只有Rz为0的H_R是需要考虑的
        # if R_vec[ir, 2] == 0:
        rdotk = TwoPi * np.dot(r_vec[ir, 0:2], k_2d)
        ham_k += ham_R[ir, :, :] * complex(np.cos(rdotk), np.sin(rdotk)) / n_degen[ir]
    return ham_k - np.eye(num_wann, dtype=complex128) * efermi


@njit(parallel=True)
def transmission_k(num_wann, ham_R, n_rpts, r_vec, n_degen, efermi,
                   li_lc, li_rc, sRl, sRr, gm_l, gm_r,
                   n_e, e_list, nkpt, kpts):
    # 每个k点的透射系数数组
    trans_k = np.zeros((n_e, nkpt), dtype=float)
    for ik in prange(nkpt):
        ham_k = _get_ham_k2d(num_wann, ham_R, n_rpts, r_vec, n_degen, kpts[ik, :], efermi)
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


@njit(parallel=True)
def get_dos_e(num_wann, ham_R, n_rpts, r_vec, n_degen, efermi, n_e, e_list, nkpt, kpts):
    dos_k = np.zeros((n_e, nkpt), dtype=float)
    for ik in prange(nkpt):
        ham_k = _get_ham_k2d(num_wann, ham_R, n_rpts, r_vec, n_degen, kpts[ik, :], efermi)
        for ie in range(n_e):
            gR = np.linalg.inv(np.eye(num_wann, dtype=complex128) * (e_list[ie] + 1j * Eta) - ham_k)
            dos_k[ie, ik] = np.trace(gR).imag
    return - np.sum(dos_k, axis=1) / pi / nkpt


@njit(parallel=True)
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


def read_tb_file(tb_file='wannier90_tb.dat'):
    seedname = tb_file.split('/')[-1].split('_')[0]
    # read tb file
    with open(tb_file, 'r') as f:
        line = f.readline()
        print("reading tb file %s ( %s )" % (tb_file, line.strip()))
        # real lattice
        real_lattice = np.array([f.readline().split()[:3] for i in range(3)], dtype=float)
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
        n_degen = np.array(ndegen, dtype=int)
        # initialize R_vec and Ham_R
        irpt = []
        ham_R = np.zeros((n_Rpts, num_wann, num_wann), dtype=complex)
        # read each R_vec[nRvec0, 3] and Ham_R[n_Rpts, num_wann, num_wann]
        for ir in range(n_Rpts):
            f.readline()
            irpt.append(f.readline().split())
            hh = np.array(
                [[f.readline().split()[2:4] for n in range(num_wann)] for m in range(num_wann)],
                dtype=float).transpose((2, 0, 1))
            ham_R[ir, :, :] = (hh[0, :, :] + 1j * hh[1, :, :])
        R_vec = np.array(irpt, dtype=float)  # 为了和k.R正常一些
        print('shape of ham_R is %s' % list(ham_R.shape))
        print('shape of R_vec is %s' % list(R_vec.shape))
        print('shape of n_degen is %s' % list(n_degen.shape))

        r_mat_R = np.zeros((n_Rpts, 3, num_wann, num_wann), dtype=complex)
        for ir in range(n_Rpts):
            f.readline()
            assert (np.array(f.readline().split(), dtype=int) == R_vec[ir]).all()
            aa = np.array([[f.readline().split()[2:8]
                            for n in range(num_wann)]
                           for m in range(num_wann)], dtype=float)
            r_mat_R[ir, :, :, :] = (aa[:,:,0::2] + 1j*aa[:,:,1::2]).transpose((2,1,0)) / n_degen[ir]
        print('shape of r_mat_R is %s' % list(r_mat_R.shape))
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


@njit(parallel=True)
def fourier_phase_R_to_k(R_vec, n_degen, n_Rpts, kpt):
    """
    计算从R表象到k表象的相因子
    """
    # rdotk = np.einsum('ij,j', R_vec, kpt)
    # phase_fac = (np.cos(rdotk) + 1j * np.sin(rdotk)) / n_degen
    # mat_k = np.einsum('ijk, i', mat_R, fac)
    # # eig_n, uu = np.linalg.eigh(mat_k)
    phase_fac = np.zeros(n_Rpts, dtype=complex128)
    # 使用 prange 来替代 for 循环，以启用 Numba 的并行执行
    for ir in prange(n_Rpts):
        rdotk = np.dot(R_vec[ir], kpt) * TwoPi  # 计算点积
        phase_fac[ir] = (cos(rdotk) + 1j * sin(rdotk)) / n_degen[ir]
    return phase_fac


def fourier_R_to_k(mat_R, R_vec, phase_fac, real_lattice, iout='0123'):
    """

    @param mat_R: R空间的厄米矩阵
    @param R_vec: 各个R的坐标
    @param phase_fac: 各个R的相因子
    @param real_lattice: 实空间格子
    @param iout: 输出选项字符串，0代表mat_k, 1~3代表xyz三个方向的dmat/dk
    """
    cart_vec = R_vec @ real_lattice  # frac to cart
    mat_k = np.einsum('ijk, i', mat_R, phase_fac)
    output = {0: mat_k}
    if '1' in iout:
        mat_k_dx = np.einsum('ijk, i, i', mat_R, cart_vec[:, 0], phase_fac) * 1j
        output[1] = mat_k_dx
    if '2' in iout:
        mat_k_dy = np.einsum('ijk, i, i', mat_R, cart_vec[:, 1], phase_fac) * 1j
        output[2] = mat_k_dy
    if '3' in iout:
        mat_k_dz = np.einsum('ijk, i, i', mat_R, cart_vec[:, 2], phase_fac) * 1j
        output[3] = mat_k_dz
    return output


@njit
def get_eig_da(eig, ham_da, uu, num_wann):
    eig_da = np.zeros(num_wann, dtype=float)
    ham_bar_da = unitary_trans(ham_da, uu)
    i = 0
    while i < num_wann:
        diff = eig[i + 1] - eig[i] if i + 1 < num_wann else 1e-4 + 1.0
        if diff < 1e-4:
            degen_min = i
            degen_max = degen_min + 1
            while degen_max + 1 < num_wann:
                diff = eig[degen_max + 1] - eig[degen_max]
                if diff < 1e-4:
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


def get_spin_splitting(ham, num_wann):
    onsite = np.diagonal(ham).real
    return (- onsite[0:num_wann//2] + onsite[num_wann//2:num_wann]) / 2


@njit
def unitary_trans(mat, uu):
    return uu.conj().T @ mat @ uu


@njit
def A_n(eig_n, ef, eta):
    de = eig_n - ef
    return eta / (de * de + eta * eta / 4)


@njit(parallel=True)
def Sp2_mn(e_s, um, un, num_wann):
    """
    计算|<um|S+.delta_0|un>|^2，单位 eV^2
    @param e_s: real[num_wann/2] 各轨道在位自旋劈裂能
    @param um: complex[num_wann]左矢
    @param un: complex[num_wann]右矢
    @param num_wann: WF空间维度
    @return: real[num_wann, num_wann] |S+.delta_0|^2[n_, m_] 矩阵
    """
    sp2 = np.zeros((num_wann, num_wann), dtype=float)
    half = num_wann // 2
    for n_ in prange(num_wann):
        un_dn = np.ascontiguousarray(un[half:num_wann, n_])
        for m_ in prange(num_wann):
            um_up = np.ascontiguousarray(um[0:half, m_].conj())
            sp2[n_, m_] = abs(np.dot(um_up, un_dn * e_s))**2
    return sp2


@njit
def guess(x):
    return 1.0 / (2.0 + exp(-x) + exp(+x))


# @njit(parallel=True)
# def W_mn_q(eig, eig_qp, num_wann, ef, eta):
#     """
#     W 矩阵，单位 eV^-2
#     @param eig:
#     @param eig_qp:
#     @param num_wann:
#     @param ef:
#     @param eta:
#     @return:
#     """
#     ww = np.zeros((num_wann, num_wann), dtype=float)
#     for n_ in prange(num_wann):
#         for m_ in prange(num_wann):
#             ww[n_, m_] = A_n(eig[n_], ef, eta) * A_n(eig_qp[m_], ef, eta)  # * guess((ef - eig[n_])/eta) / eta
#     return ww


@njit(parallel=True)
def get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, sp2_q, num_wann, ef, eta):
    # half = num_wann // 2
    sum_qvd_k = 0.0
    sum_qv_k = 0.0
    sum_alpha_k = 0.0
    for n_ in prange(num_wann):
        An = A_n(eig[n_], ef, eta)
        for m_ in prange(num_wann):
            Am = A_n(eig_q[m_], ef, eta)
            ww = An * Am
            # in units 1
            sum_alpha_k += sp2_q[n_, m_] * ww
            # qvd in units eV angst.
            sum_qvd_k += sp2_q[n_, m_] * ww * (eig_q_da[m_] - eig_da[n_])
            # qvs in units eV angst.
            if n_ != m_:
                sum_qv_k += sp2_q[n_, m_] * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                         / (eig_q[m_] - eig[n_]))

    return sum_alpha_k, sum_qvd_k, - sum_qv_k


@njit(parallel=True)
def get_alpha_beta_inter_k(eig, eig_q, eig_da, eig_q_da, sp2_q, num_wann, ef, eta):
    sum_qvd_k = 0.0
    sum_qv_k = 0.0
    sum_alpha_k = 0.0
    for n_ in prange(num_wann):
        An = A_n(eig[n_], ef, eta)
        for m_ in prange(num_wann):
            if n_ == m_:
                continue
            Am = A_n(eig_q[m_], ef, eta)
            ww = An * Am
            # in units 1
            sum_alpha_k += sp2_q[n_, m_] * ww
            # qvd in units eV angst.
            sum_qvd_k += sp2_q[n_, m_] * ww * (eig_q_da[m_] - eig_da[n_])
            # qvs in units eV angst.
            sum_qv_k += sp2_q[n_, m_] * ((eig_q_da[m_] * Am - eig_da[n_] * An)
                                         / (eig_q[m_] - eig[n_] - 1j*Eta)).real

    return sum_alpha_k, sum_qvd_k, - sum_qv_k


@njit(parallel=True)
def get_alpha_beta_intra_k(eig, eig_q, eig_da, eig_q_da, sp2_q, num_wann, ef, eta):
    sum_qvd_k = 0.0
    sum_qv_k = 0.0
    sum_alpha_k = 0.0
    for n_ in prange(num_wann):
        An = A_n(eig[n_], ef, eta)
        Am = A_n(eig_q[n_], ef, eta)
        ww = An * Am
        # in units 1
        sum_alpha_k += sp2_q[n_, n_] * ww
        # qvd in units eV angst.
        sum_qvd_k += sp2_q[n_, n_] * ww * (eig_q_da[n_] - eig_da[n_])
        # qvs in units eV angst.
        sum_qv_k += sp2_q[n_, n_] * (eig_q_da[n_] * Am - eig_da[n_] * An) / (eig_q[n_] - eig[n_])

    return sum_alpha_k, sum_qvd_k, sum_qv_k

@cuda.jit
def cuda_alpha_beta_k(eig, eig_qp, eig_da, eig_qp_da, uu, uu_qp, e_s, num_wann, ef, eta):
    i = cuda.grid(1)
    if i < num_wann * num_wann:
        m_ = i % num_wann
        n_ = i // num_wann


@njit(parallel=True)
def get_carrier_k(eig, eig_d, eig_dd, num_wann, ef, eta):
    An = A_n(eig, ef, eta)
    nc_ = np.zeros((3,3, num_wann), dtype=float)
    for i in prange(3):
        for j in prange(3):
            for n_ in prange(num_wann):
                nc_[i, j, num_wann] = (An[n_] * eig_d[j, n_] * eig_d[i, n_] / (eig_dd[i, j, n_] - 1j * eta)).real
            # nc[i, j] = np.sum(An * eig_d[j] * eig_d[i] / (eig_dd[i, j] - 1j * eta)).real
    return np.sum(nc_, axis=2)


def get_kpts_mesh(kmesh):
    k1, k2, k3 = np.meshgrid(np.arange(kmesh[0], dtype=float)/kmesh[0],
                             np.arange(kmesh[1], dtype=float)/kmesh[1],
                             np.arange(kmesh[2], dtype=float)/kmesh[2], indexing='ij')
    return np.column_stack((k1.ravel(), k2.ravel(), k3.ravel()))


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

