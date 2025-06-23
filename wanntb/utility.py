import numpy as np
from numba import njit, prange
from .constant import TwoPi, Eta_4, S_


def get_list_index(item, li):
    for index, value in enumerate(li):
        if value == item:
            return index
    return -1


@njit(nogil=True)
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
    e0 = h0 - np.eye(n_dim) * (1j * Eta_4 - mu)
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


@njit(nogil=True)
def _matrix_add_by_index(target, xlist, ylist, add):
    nx = xlist.shape[0]
    ny = ylist.shape[0]
    assert add.shape[0] == nx and add.shape[1] == ny, 'add.shape == (len(xlist), len(ylist))'
    for ix in range(nx):
        for iy in range(ny):
            target[xlist[ix], ylist[iy]] += add[ix, iy]
    pass


@njit(nogil=True)
def _get_submatrix_by_index(target, xlist, ylist):
    nx = int(len(xlist))
    ny = int(len(ylist))
    sub = np.zeros((nx, ny), dtype=np.complex128)
    for ix in range(nx):
        for iy in range(ny):
            sub[ix, iy] = target[xlist[ix], ylist[iy]]
    return sub


# 不用了，废弃
# @njit
# def transmission_jit(num_wann, ham, li_lc, li_rc,
#                      energy, sigmaRl, sigmaRr, gamma_l, gamma_r):
#     _matrix_add_by_index(ham, li_lc, li_lc, sigmaRl)
#     _matrix_add_by_index(ham, li_rc, li_rc, sigmaRr)
#     # 总的推迟格林函数和超越格林函数
#     gR = np.linalg.inv(np.eye(num_wann, dtype=np.complex128) * energy - ham)
#     gA = gR.T.conjugate()
#     gR_lr = _get_submatrix_by_index(gR, li_lc, li_rc)
#     gA_rl = _get_submatrix_by_index(gA, li_rc, li_lc)
#     # tr(gamma_l.gR_lr.gamma_r.gA_rl)
#     return np.trace(np.dot(np.dot(np.dot(gamma_l, gR_lr), gamma_r), gA_rl)).real


@njit(nogil=True)
def _get_ham_k2d(num_wann, ham_R, r_vec, k_2d, efermi):
    # ham_k = np.zeros((num_wann, num_wann), dtype=complex128)
    rdotk = r_vec[:, 0:2] @ k_2d * TwoPi
    phase_fac = np.cos(rdotk) + 1j * np.sin(rdotk)
    ham_k = phase_fac @ ham_R
    # for ir in range(n_rpts):
    #     # 只有Rz为0的H_R是需要考虑的
    #     # if R_vec[ir, 2] == 0:
    #     rdotk = TwoPi * np.dot(r_vec[ir, 0:2], k_2d)
    #     ham_k += ham_R[ir, :, :] * complex(np.cos(rdotk), np.sin(rdotk)) / n_degen[ir]
    return ham_k - np.eye(num_wann, dtype=np.complex128) * efermi


@njit(parallel=True, nogil=True)
def transmission_k2d_kpar(num_wann, ham_R, R_vec, efermi,
                          li_lc, li_rc, sRl, sRr, gm_l, gm_r, e_list,
                          kpts):
    n_e = e_list.shape[0]
    nkpts = kpts.shape[0]
    # 每个k点的透射系数数组
    trans_k = np.zeros((n_e, nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        ham_k = _get_ham_k2d(num_wann, ham_R, R_vec, kpts[ik, :], efermi)
        ham_k2 = np.zeros((num_wann, num_wann), dtype=np.complex128)
        for ie in range(n_e):
            ham_k2[:, :] = ham_k.copy()
            _matrix_add_by_index(ham_k2, li_lc, li_lc, sRl[ie, :, :])
            _matrix_add_by_index(ham_k2, li_rc, li_rc, sRr[ie, :, :])
            # 总的推迟格林函数和超越格林函数
            gR = np.linalg.inv(np.eye(num_wann, dtype=np.complex128) * e_list[ie] - ham_k2)
            gA = gR.T.conjugate()
            gR_lr = _get_submatrix_by_index(gR, li_lc, li_rc)
            gA_rl = _get_submatrix_by_index(gA, li_rc, li_lc)
            tt = np.dot(np.dot(np.dot(gm_l[ie, :, :], gR_lr), gm_r[ie, :, :]), gA_rl)
            trans_k[ie, ik] = np.trace(tt).real

    return np.sum(trans_k, axis=1) / nkpts


@njit(parallel=True, nogil=True)
def get_dos_e_kpar(num_wann, ham_R, R_vec, ef, n_e, e_list, nkpts, kpts):
    dos_k = np.zeros((n_e, nkpts), dtype=float)
    for ik in prange(nkpts):
        ham_k = _get_ham_k2d(num_wann, ham_R, R_vec, kpts[ik, :], ef)
        for ie in range(n_e):
            gR = np.linalg.inv(np.eye(num_wann, dtype=np.complex128) * (e_list[ie] + 1j * Eta_4) - ham_k)
            dos_k[ie, ik] = np.trace(gR).imag
    return - np.sum(dos_k, axis=1) / np.pi / nkpts


@njit(parallel=True, nogil=True)
def get_self_energies_epar(l_h0, l_t, l_dim,
                           r_h0, r_t, r_dim,
                           n_e, e_list,
                           mu_l, mu_r, v_lc, v_rc, n_iter=25):
    n_lc = v_lc.shape[1]
    n_rc = v_rc.shape[1]
    sRl = np.zeros((n_e, n_lc, n_lc), dtype=np.complex128)
    sRr = np.zeros((n_e, n_rc, n_rc), dtype=np.complex128)
    gm_l = np.zeros((n_e, n_lc, n_lc), dtype=np.complex128)
    gm_r = np.zeros((n_e, n_rc, n_rc), dtype=np.complex128)
    for ie in prange(n_e):
        gsRl = _surface_GR(e_list[ie], l_dim, l_h0, l_t, mu_l, n_iter)
        gsRr = _surface_GR(e_list[ie], r_dim, r_h0, r_t, mu_r, n_iter)
        sRl[ie, :, :] = np.ascontiguousarray(v_lc.T.conjugate()).dot(gsRl).dot(v_lc)
        sRr[ie, :, :] = np.ascontiguousarray(v_rc.T.conjugate()).dot(gsRr).dot(v_rc)
        gm_l[ie, :, :] = (sRl[ie, :, :] - sRl[ie, :, :].T.conjugate()) * 1j
        gm_r[ie, :, :] = (sRr[ie, :, :] - sRr[ie, :, :].T.conjugate()) * 1j
    return sRl, sRr, gm_l, gm_r


def hermiization_R(mat, R_vec):
    _mat = np.copy(mat)
    shape = _mat.shape
    if len(shape) == 3:
        nrpt = shape[0]
    else:
        nrpt = shape[0]
        ndim = shape[1]
    lhermi = np.zeros(nrpt, dtype=np.bool_)
    for ir in range(nrpt):
        if lhermi[ir]:
            continue
        for jr in range(ir-1, nrpt):
            dr = R_vec[ir] + R_vec[jr]
            if np.dot(dr, dr) == 0:
                if len(shape) == 3:
                    _real = (_mat[ir] + _mat[jr].T).real / 2
                    _imag = (_mat[ir] - _mat[jr].T).imag / 2
                    _mat[ir] = _real + 1j * _imag
                    _mat[jr] = (_real - 1j * _imag).T
                else:
                    for k in range(ndim):
                        _real = (_mat[ir, k] + _mat[jr, k].T).real / 2
                        _imag = (_mat[ir, k] - _mat[jr, k].T).imag / 2
                        _mat[ir, k] = _real + 1j * _imag
                        _mat[jr, k] = (_real - 1j * _imag).T
                lhermi[ir] = True
                lhermi[jr] = True
                break
    return _mat


@njit(nogil=True)
def unitary_trans(mat, uu, inverse=False):
    if inverse:
        return uu @ mat @ uu.conj().T
    else:
        return uu.conj().T @ mat @ uu

@njit(nogil=True)
def unitary_trans_sub(mat, uu1, uu2):
    return uu1.conj().T @ mat @ uu2


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
        n_degen = np.array(ndegen, dtype=np.uint8)
        # initialize R_vec and Ham_R
        irpt = []
        ham_R = np.zeros((n_Rpts, num_wann, num_wann), dtype=np.complex128)
        # read each R_vec[nRvec0, 3] and Ham_R[n_Rpts, num_wann, num_wann]
        for ir in range(n_Rpts):
            f.readline()
            irpt.append(f.readline().split())
            hh = np.array(
                [[f.readline().split()[2:4] for n in range(num_wann)] for m in range(num_wann)],
                dtype=float).transpose((2, 1, 0))
            ham_R[ir, :, :] = (hh[0, :, :] + 1j * hh[1, :, :])
        # R_vec = np.ascontiguousarray(irpt, dtype=np.float64)  # 为了和k.R正常一些
        R_vec = np.ascontiguousarray(irpt, dtype=np.int16)  # 为了和k.R正常一些
        ham_R = np.ascontiguousarray(ham_R)
        print('ham_R: %s %s' % (ham_R.dtype, list(ham_R.shape)))
        print('R_vec: %s %s' % (R_vec.dtype, list(R_vec.shape)))
        print('n_degen: %s %s' % (n_degen.dtype, list(n_degen.shape)))
        r_mat_R = np.zeros((n_Rpts, 3, num_wann, num_wann), dtype=np.complex128)
        for ir in range(n_Rpts):
            f.readline()
            assert (np.array(f.readline().split(), dtype=np.int16) == R_vec[ir]).all()
            aa = np.array([[f.readline().split()[2:8]
                            for n in range(num_wann)]
                           for m in range(num_wann)], dtype=np.float64)
            r_mat_R[ir, :, :, :] = (aa[:,:,0::2] + 1j*aa[:,:,1::2]).transpose((2,1,0))
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


def read_spin_file(R_vec, n_Rpts, num_wann, ss_file='wannier90_SS_R.dat'):
    ss_R = np.zeros((n_Rpts, 3, num_wann, num_wann), dtype=np.complex128)
    with open(ss_file, 'r') as f:
        line = f.readline()
        print("reading spin file %s ( %s )" % (ss_file, line.strip()))
        assert int(f.readline()) == num_wann
        assert int(f.readline()) == n_Rpts
        ndegen = []
        while len(ndegen) < n_Rpts:
            ndegen += f.readline().split()
        n_degen = np.array(ndegen, dtype=np.uint8)

        for ir in range(n_Rpts):
            f.readline()
            assert (np.array(f.readline().split(), dtype=np.int32) == R_vec[ir]).all()
            aa = np.array([[f.readline().split()[2:8]
                            for n in range(num_wann)]
                           for m in range(num_wann)], dtype=np.float64)
            ss_R[ir, :, :, :] = (aa[:,:,0::2] + 1j*aa[:,:,1::2]).transpose((2,1,0))
        print('ss_R: %s %s' % (ss_R.dtype, list(ss_R.shape)))
    for ir in range(n_Rpts):
        ss_R[ir] /= n_degen[ir]
    _ss_R = hermiization_R(ss_R, R_vec)
    return _ss_R


@njit(nogil=True)
def fourier_phase_R_to_k(R_vec, kpt):
    """
    计算从R表象到k表象的相因子, [n_Rpt]
    @param R_vec: [n_Rpt, 3]
    @param kpt: [3]
    """
    rdotk = R_vec @ kpt * TwoPi
    return np.cos(rdotk) + np.sin(rdotk) *1j


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
    output = np.zeros((4, num_wann, num_wann), dtype=np.complex128)
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
            output[0, i, j] = np.sum(mat_Rij * phase_fac)
            if 1 in iout:
                output[1, i, j] = np.sum(mat_Rij * R_vec_cart_T[0] * phase_fac) * 1j
            if 2 in iout:
                output[2, i, j] = np.sum(mat_Rij * R_vec_cart_T[1] * phase_fac) * 1j
            if 3 in iout:
                output[3, i, j] = np.sum(mat_Rij * R_vec_cart_T[2] * phase_fac) * 1j
    return output


@njit(nogil=True)
def fourier_R_to_k_vec3(vec_R, phase_fac):
    n_rpt, _, num_wann, _ = vec_R.shape
    oo_true = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for k in range(3):
        for i in range(num_wann):
            for j in range(num_wann):
                vec_Rkij = np.ascontiguousarray(vec_R[:, k, i, j])
                oo_true[k, i, :] = np.sum(vec_Rkij * phase_fac)
    return oo_true


@njit(nogil=True)
def fourier_R_to_k_curl(vec_R, phase_fac, R_vec_cart_T):
    n_rpt, _, num_wann, _ = vec_R.shape
    oo_curl = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for i in range(num_wann):
        for j in range(num_wann):
            vec_ij_0 = np.ascontiguousarray(vec_R[:, 0, i, j])
            vec_ij_1 = np.ascontiguousarray(vec_R[:, 1, i, j])
            vec_ij_2 = np.ascontiguousarray(vec_R[:, 2, i, j])
            oo_curl[0, i, j] = np.sum((R_vec_cart_T[1] * vec_ij_2
                                       - R_vec_cart_T[2] * vec_ij_1) * phase_fac) * 1j
            oo_curl[1, i, j] = np.sum((R_vec_cart_T[2] * vec_ij_0
                                       - R_vec_cart_T[0] * vec_ij_2) * phase_fac) * 1j
            oo_curl[2, i, j] = np.sum((R_vec_cart_T[0] * vec_ij_1
                                       - R_vec_cart_T[1] * vec_ij_0) * phase_fac) * 1j
    return oo_curl

@njit(nogil=True)
def get_eig_da(eig, ham_da, uu, num_wann, eig_diff=1e-4):
    eig_da = np.zeros(num_wann, dtype=np.float64)
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
def _ham_k_da_system(ham_R, R_vec, R_vec_cart_T, num_wann, kpt, direction):
    """
    计算一个k点的哈密顿量、本征值、征值随k的某个方向的导数和本征态。
    @param ham_R: R空间的紧束缚哈密顿量
    @param R_vec: R格点坐标
    @param R_vec_cart_T: R格点的xyz坐标，数组排列转置
    @param num_wann: WF空间大小
    @param direction: 方向，123分别代表xyz
    @param kpt: k点坐标，倒格矢表象
    @return: k空间哈密顿量，本征值，本征值随k的导数，本征态
    """
    fac = fourier_phase_R_to_k(R_vec, kpt)
    out = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[direction])
    ham_k, ham_k_da = out[0], out[direction]
    eig, uu = np.linalg.eigh(ham_k)
    eig_da = get_eig_da(eig, ham_k_da, uu, num_wann)
    return ham_k, eig, eig_da, uu

# def get_spin_splitting(ham, num_wann):
#     onsite = np.diagonal(ham).real
#     return (- onsite[0:num_wann//2] + onsite[num_wann//2:num_wann]) / 2


@njit(nogil=True)
def _ham_k_system(ham_R, R_vec, R_vec_cart_T, kpt):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_k = fourier_R_to_k(ham_R, R_vec_cart_T, fac, iout=[5])[0]
    eig, uu = np.linalg.eigh(ham_k)
    return ham_k, eig, uu


@njit(parallel=True, nogil=True)
def get_eig_for_kpts_kpar(ham_R, R_vec, R_vec_cart_T, num_wann, kpts):
    nkpts = kpts.shape[0]
    eigs = np.zeros((nkpts, num_wann), dtype=float)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        fac = fourier_phase_R_to_k(R_vec, kpt)
        ham_k = fourier_R_to_k(ham_R, R_vec_cart_T, fac)[0]
        eig, uu = np.linalg.eigh(ham_k)
        eigs[ik, :] = eig
    return eigs


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


@njit(nogil=True)
def dos_fermi(eig, ef, eta):
    fac = (eig - ef) / eta
    return 1.0 / (np.exp(fac) + 1) / (1 + np.exp(-fac)) / eta


@njit(nogil=True)
def spin_w(gamma, num_wann, udud_order=False):
    n_orbit = num_wann//2
    if udud_order:
        sw = np.kron(np.eye(n_orbit, dtype=np.complex128), S_[gamma])
    else:
        sw = np.kron(S_[gamma], np.eye(n_orbit, dtype=np.complex128))
    return sw


@njit(nogil=True)
def get_inv_e_d(eig, num_wann):
    """
    inv_e_d[m, n] = 1 / (e_n - e_m)
    """
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    for n_ in range(num_wann):
        for m_ in range(num_wann):
            if m_ == n_:
                continue
            e_d = eig[n_] - eig[m_]
            inv_e_d[m_, n_] = 1.0 / e_d if abs(e_d) > 1e-8 else 0.0
    return inv_e_d
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
    return 1.0 / (2.0 + np.exp(-x) + np.exp(+x))


@njit(parallel=True, nogil=True)
def get_carrier_kpar(ham_R, R_vec, R_vec_cart_T,
                     num_wann, direction, kpts, q_frac, q, ef, eta):
    nkpts = kpts.shape[0]
    list_o_k = np.zeros(nkpts, dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        ham_k, eig, eig_da, uu = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T,
                                                  num_wann, kpt, direction)
        # k + q
        ham_q, eig_q, eig_q_da, uu_q = _ham_k_da_system(ham_R, R_vec, R_vec_cart_T,
                                                        num_wann, kpt + q_frac, direction)
        eig_dd_inv = (q / (eig_q_da - eig_da - 1j * eta/q)).real
        n_eig_ef = A_vec(eig, ef, eta)
        list_o_k[ik] = np.sum(n_eig_ef * eig_dd_inv * eig_da * eig_da)
    return np.sum(list_o_k) / (nkpts * TwoPi * TwoPi)




