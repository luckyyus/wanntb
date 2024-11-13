import numpy as np
from . import utility as ut


class TBSystem:

    def __init__(self, tb_file='wannier90_tb.dat'):
        data = ut.read_tb_file(tb_file=tb_file)
        self.seedname = data['seedname']
        self.num_wann = data['num_wann']
        self.real_lattice = data['real_lattice']
        self.recip_lattice = data['recip_lattice']
        self.n_rpts = data['n_Rpts']
        self.ham_R = data['ham_R']
        self.R_vec = data['R_vec']
        self.n_degen = data['n_degen']
        self.r_mat_R = data['r_mat_R']
        # calculate wannier centers for each WF
        for ir in range(self.n_rpts):
            if self.R_vec[ir, :].dot(self.R_vec[ir, :]) < 0.001:
                self.iR0 = ir
                self.wann_centers_cart = np.diagonal(self.r_mat_R[ir, :, :, :], axis1=1, axis2=2).T.real
                # print(self.wann_centers_cart)
                # print(self.recip_lattice.shape)
                self.wann_centers_frac = np.einsum('ij,jk',
                                                   self.wann_centers_cart,
                                                   self.recip_lattice) / ut.TwoPi
                # print(self.wann_centers_frac)
                break

    def get_onsite_energy(self):
        return np.diagonal(self.ham_R[self.iR0, :, :]).real

    def get_spin_splitting(self):
        onsite = self.get_onsite_energy()
        return - onsite[0:self.num_wann//2] + onsite[self.num_wann//2:self.num_wann]

    def get_alpha_beta(self, kmesh, ef, eta=1e-3, q=1e-4, direction=1):

        print('relaxation time %e s' % (ut.Hbar_ / eta))
        kpts = ut.get_kpts_mesh(kmesh)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        energy_spin = self.get_spin_splitting()
        # print('spin splitting enrgy is:', energy_spin)
        q_frac = q * np.dot(ut.Cart[direction-1, :], self.recip_lattice) / ut.TwoPi
        print('q in fraction units: %s' % q_frac)
        list_o_k = np.zeros((nkpts, 3), dtype=float)
        # [alpha, beta_qvd, beta_qvs]
        for ik in range(nkpts):
            kpt = kpts[ik]
            fac = ut.fourier_phase_R_to_k(self.R_vec, self.n_degen, self.n_rpts, kpt)
            out = ut.fourier_R_to_k(self.ham_R,
                                    self.R_vec,
                                    fac,
                                    self.real_lattice,
                                    iout='%d%d' % (0, direction))
            ham_k = out[0]
            eig, uu = np.linalg.eigh(ham_k)
            ham_k_da = out[1]
            eig_da = ut.get_eig_da(eig, ham_k_da, uu, self.num_wann)

            # k + q
            fac_p = ut.fourier_phase_R_to_k(self.R_vec, self.n_degen, self.n_rpts, kpt + q_frac)
            outp = ut.fourier_R_to_k(self.ham_R,
                                     self.R_vec,
                                     fac_p,
                                     self.real_lattice,
                                     iout='%d%d' % (0, direction))
            ham_qp = outp[0]
            ham_qp_da = outp[1]
            eig_qp, uu_qp = np.linalg.eigh(ham_qp)
            eig_qp_da = ut.get_eig_da(eig_qp, ham_qp_da, uu_qp, self.num_wann)

            # ww_qp = ut.W_mn_qp(eig, eig_qp, self.num_wann, ef, eta)
            sp2_qp = ut.Sp2_mn(energy_spin, uu_qp, uu, self.num_wann)

            list_o_k[ik, :] = ut.get_alpha_beta_k(uu, eig, eig_qp, eig_da, eig_qp_da, sp2_qp, self.num_wann, ef, eta)
        sum_o = np.sum(list_o_k, axis=0) / nkpts
        print(sum_o)
        alpha = sum_o[0] / (ut.TwoPi * 4)
        beta = sum_o[1] / sum_o[2] / 2
        return alpha, beta




