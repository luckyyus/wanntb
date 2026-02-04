import numpy as np
from wanntb._layeredsystem import LayeredSystem
from ..struct import Structure
from ._gf import _surface_GR, transmission_k2d_kpar, get_self_energies_epar
from time import time
from wanntb.constant import H_


class Lead:
    """
    电极类

    """
    def __init__(self, h0, t):
        self.h0 = np.array(h0, dtype=complex)
        self.n_dim = self.h0.shape[0]
        self.t = np.array(t, dtype=complex)
        assert self.t.shape[-1] == self.n_dim, 'H0 与 t矩阵维度相同'
        if len(self.t.shape) == 2:  # 只有最近邻作用
            self.max_z = 1
        else:
            self.max_z = self.t.shape[0]

    def surface_GR(self, energy, mu=0.0, n_iter=1000):
        return _surface_GR(energy, self.n_dim, self.h0, self.t, mu=mu, n_iter=n_iter)

    def surface_GA(self, energy, mu=0.0, n_iter=1000):
        return self.surface_GR(energy, mu, n_iter=n_iter).T.conjugate()


class NEGF:
    """
    非平衡格林函数系统
    包括左电极、右电极、器件的哈密顿量以及器件部分的结构
    GR = [E-Ham_k-\SigmaR]^-1
    T = Tr(Gamma_l.gR_lr.Gamma_r.gA_rl)
    J = e/H_*integral T(E)|u_l-u_r
    """
    def __init__(self, device: LayeredSystem, pos: Structure, paras: dict):
        self.paras = paras  # 所有设置的引用
        self.device = device
        self.device_type = paras['device_type'][0:2]
        orbits = pos.get_orbital_list(paras['projections'])
        tops, btms = self.device.get_tops_bottoms(orbits, n_top=paras['n_l'], n_bottom=paras['n_r'],
                                                  top_oname=paras['top_orbit_name'],
                                                  bottom_oname=paras['bottom_orbit_name'])
        if paras['is_top_to_bottom']:
            self.li_lc = tops
            self.li_rc = btms
        else:
            self.li_lc = btms
            self.li_rc = tops

        print('list of right and left WFs are %s and %s' % (tops, btms))
        self.lead_l = Lead(paras['lead_l_h0'], paras['lead_l_t'])
        self.lead_r = Lead(paras['lead_r_h0'], paras['lead_r_t'])
        self.v_lc = np.array(paras['v_lc'], dtype=complex)
        self.v_rc = np.array(paras['v_rc'], dtype=complex)

        # 建立网格
        # self.llx, self.lly = np.meshgrid(tops, tops)
        # self.rrx, self.rry = np.meshgrid(btms, btms)
        # self.lrx, self.lry = np.meshgrid(tops, btms)
        # self.rlx, self.rly = np.meshgrid(btms, tops)

    # def _transmission(self, energy, k, sigmaRl, sigmaRr, gamma_l, gamma_r):
    #     """
    #     计算透射系数
    #     :param energy: 入射能量
    #     :param k: k点
    #     :return:
    #     """
    #     # print(self.llx.shape, self.lly.shape, sigmaRl.shape)
    #     # 器件部分
    #     ham_k = self.device.get_H0_k(k)
    #     _matrix_add_by_index(ham_k, self.li_lc, self.li_lc, sigmaRl)
    #     _matrix_add_by_index(ham_k, self.li_rc, self.li_rc, sigmaRr)
    #     # ham_k[self.llx, self.lly] += sigmaRl
    #     # ham_k[self.rrx, self.rry] += sigmaRr
    #     # 总的推迟格林函数和超越格林函数
    #     gR = np.linalg.inv(np.eye(self.device.num_wann) * energy - ham_k)
    #     gA = gR.T.conjugate()
    #     gR_lr = gR[self.lrx, self.lry]
    #     gA_rl = gA[self.rlx, self.rly]
    #     # Tr(Gamma_l.gR_lr.Gamma_r.gA_rl)
    #     return np.trace(gamma_l.dot(gR_lr).dot(gamma_r).dot(gA_rl)).real

    def transmission_list(self, efermi=0.0, mu_l=0.0, mu_r=0.0):
        """
        获得透射系数随能量（0点为器件费米能级）的关系曲线
        :param efermi: 器件的本征费米能级，对半导体可以是价带顶或导带底，也可以是charge neutral level
        :param mu_l: 左电极电势
        :param mu_r: 右电极电势
        :return: 两列：能量和透射系数
        """
        start = time()  # debug
        u_c = mu_l - mu_r
        # n型掺杂时，u_c为负；p型掺杂时，u_c为正
        if self.device_type == 's':
            mu = efermi - u_c
        else:
            mu = efermi - u_c / 2
        emin, emax = tuple(self.paras['e_range'])
        assert emin < emax, 'e_range 必须是第一个数比第二个小'
        e_list = np.linspace(emin, emax, self.paras['e_num'], endpoint=False)
        n_e = self.paras['e_num']
        # 各能量透射系数数组
        trans_s = np.zeros((n_e, 2), dtype=float)
        trans_s[:, 0] = e_list
        # 总k点数
        nkpt = self.paras['k_mesh'][0] * self.paras['k_mesh'][1]

        kxs = np.linspace(0.0, 1.0, self.paras['k_mesh'][0], endpoint=False, dtype=float)
        kys = np.linspace(0.0, 1.0, self.paras['k_mesh'][1], endpoint=False, dtype=float)
        kpts = np.zeros((nkpt, 2), dtype=float)
        kptx, kpty = np.meshgrid(kxs, kys)
        kpts[:, 0] = kptx.reshape(nkpt)
        kpts[:, 1] = kpty.reshape(nkpt)

        sRl, sRr, gm_l, gm_r = get_self_energies_epar(self.lead_l.h0, self.lead_l.t, self.lead_l.n_dim,
                                                      self.lead_r.h0, self.lead_r.t, self.lead_r.n_dim,
                                                      n_e, e_list, mu_l, mu_r, self.v_lc, self.v_rc,
                                                      self.paras['lead_num_iter'])
        print('self energies is calculated. time: %8.2f' % (time() - start))
        np.savetxt('gm_l_0.txt', gm_l[0, :, :], '%16.8e')
        trans_s[:, 1] = transmission_k2d_kpar(self.device.num_wann, self.device.ham_R, self.device.R_vec,
                                              mu, self.li_lc, self.li_rc, sRl, sRr, gm_l, gm_r, e_list, kpts)
        # for i in range(e_list.shape[0]):
        #     # 左电极 表面格林函数 自能 展宽态密度
        #     gsRl = self.lead_l.surface_GR(e_list[i], mu_l, self.paras['lead_num_iter'])
        #     sigmaRl = self.v_lc.T.conjugate().dot(gsRl).dot(self.v_lc)  # 左电极自能
        #     gamma_l = (sigmaRl - sigmaRl.T.conjugate()) * 1j
        #     # 右电极 表面格林函数 自能 展宽态密度
        #     gsRr = self.lead_r.surface_GR(e_list[i], mu_r, self.paras['lead_num_iter'])
        #     sigmaRr = self.v_rc.T.conjugate().dot(gsRr).dot(self.v_rc)  # 右电极自能
        #     gamma_r = (sigmaRr - sigmaRr.T.conjugate()) * 1j
        #     # for ik in range(nkpt):
        #     #     ham_k = get_ham_k2d(self.device.num_wann,
        #     #                         self.device.ham_R,
        #     #                         self.device.n_Rpts,
        #     #                         self.device.R_vec,
        #     #                         self.device.n_degen,
        #     #                         kpts[ik, :],
        #     #                         mu)
        #     #     # ham_k = self.device.get_H0_k(kpts[ik, :])
        #     #     trans_k[ik] = transmission_jit(self.device.num_wann, ham_k, self.li_lc, self.li_rc,
        #     #                                    e_list[i], sigmaRl, sigmaRr, gamma_l, gamma_r)
        #     #     # trans_k[ik] = self._transmission(e_list[i], kpts[ik, :],
        #     #     #                                  sigmaRl, sigmaRr, gamma_l, gamma_r)
        #     trans_s[i, 1] = transmission_k(self.device.num_wann,
        #                                    self.device.ham_R,
        #                                    self.device.n_Rpts,
        #                                    self.device.R_vec,
        #                                    self.device.n_degen,
        #                                    mu,
        #                                    self.li_lc, self.li_rc, e_list[i],
        #                                    sigmaRl, sigmaRr, gamma_l, gamma_r, nkpt, kpts)
        #     print('%8.4f is finished. time: %8.2f' % (e_list[i], time() - start))
        # trans_s[:, 1] /= nkpt
        return trans_s

    def current(self, e_fermi, mu_l, mu_r, kbt=0.0001, filename='trans.txt'):
        emin, emax = tuple(self.paras['e_range'])
        trans_s = self.transmission_list(efermi=e_fermi, mu_l=mu_l, mu_r=mu_r)
        np.savetxt(filename, trans_s, '%16.8e')
        d_i = trans_s[:, 1] * (1.0/(np.exp((trans_s[:, 0] - mu_l)/kbt) + 1.0)
                               - 1.0/(np.exp((trans_s[:, 0] - mu_r)/kbt) + 1.0))
        return np.sum(d_i) / trans_s.shape[0] / H_ * (emax - emin)
