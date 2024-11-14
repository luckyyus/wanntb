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
        return (- onsite[0:self.num_wann//2] + onsite[self.num_wann//2:self.num_wann]) / 2

    def get_alpha_beta(self, kmesh, ef, mag, eta=1e-3, q=1e-5, direction=1):

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
            fac_q = ut.fourier_phase_R_to_k(self.R_vec, self.n_degen, self.n_rpts, kpt + q_frac)
            out = ut.fourier_R_to_k(self.ham_R,
                                     self.R_vec,
                                     fac_q,
                                     self.real_lattice,
                                     iout='%d%d' % (0, direction))
            ham_q, ham_q_da = out[0], out[1]
            eig_q, uu_q = np.linalg.eigh(ham_q)
            eig_q_da = ut.get_eig_da(eig_q, ham_q_da, uu_q, self.num_wann)

            # ww_qp = ut.W_mn_qp(eig, eig_q, self.num_wann, ef, eta)
            sp2_q = ut.Sp2_mn(energy_spin, uu_q, uu, self.num_wann)

            list_o_k[ik, :] = ut.get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, sp2_q, self.num_wann, ef, eta)
        sum_o = np.sum(list_o_k, axis=0) / nkpts
        print(sum_o)
        alpha = sum_o[0] / (ut.TwoPi * 4 * mag)
        beta = sum_o[1] / sum_o[2] / 2
        return alpha, beta

    def get_carrier(self, kmesh, ef, eta=1e-3, q=1e-5):
        kpts = ut.get_kpts_mesh(kmesh)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        q_frac = q * ut.Cart @ self.recip_lattice / ut.TwoPi
        # print(q_frac)
        list_o_k = np.zeros((nkpts, 3, 3), dtype=float)
        # std = open('nc_k.txt', 'w')
        for ik in range(nkpts):
            kpt = kpts[ik]
            fac = ut.fourier_phase_R_to_k(self.R_vec, self.n_degen, self.n_rpts, kpt)
            out = ut.fourier_R_to_k(self.ham_R,
                                    self.R_vec,
                                    fac,
                                    self.real_lattice)
            ham_k, ham_k_dx, ham_k_dy, ham_k_dz = out[0], out[1], out[2], out[3]
            eig, uu = np.linalg.eigh(ham_k)
            eig_d = np.zeros((3,self.num_wann), dtype=float)
            eig_d[0, :] = ut.get_eig_da(eig, ham_k_dx, uu, self.num_wann)
            eig_d[1, :] = ut.get_eig_da(eig, ham_k_dy, uu, self.num_wann)
            eig_d[2, :] = ut.get_eig_da(eig, ham_k_dz, uu, self.num_wann)

            # k + q
            fac_q = np.zeros((3, self.n_rpts), dtype=complex)
            fac_q[0, :] = ut.fourier_phase_R_to_k(self.R_vec, self.n_degen, self.n_rpts, kpt + q_frac[0, :])
            fac_q[1, :] = ut.fourier_phase_R_to_k(self.R_vec, self.n_degen, self.n_rpts, kpt + q_frac[1, :])
            fac_q[2, :] = ut.fourier_phase_R_to_k(self.R_vec, self.n_degen, self.n_rpts, kpt + q_frac[2, :])
            eig_dd = np.zeros((3,3,self.num_wann), dtype=float)
            for i in range(3):
                out = ut.fourier_R_to_k(self.ham_R,
                                        self.R_vec,
                                        fac_q[i],
                                        self.real_lattice)
                ham_q, ham_q_dx, ham_q_dy, ham_q_dz = out[0], out[1], out[2], out[3]
                eig_q, uu_q = np.linalg.eigh(ham_q)
                eig_q_dx = ut.get_eig_da(eig_q, ham_q_dx, uu_q, self.num_wann)
                eig_q_dy = ut.get_eig_da(eig_q, ham_q_dy, uu_q, self.num_wann)
                eig_q_dz = ut.get_eig_da(eig_q, ham_q_dz, uu_q, self.num_wann)

                eig_dd[i, 0, :] = (eig_q_dx - eig_d[0]) / q
                eig_dd[i, 1, :] = (eig_q_dy - eig_d[1]) / q
                eig_dd[i, 2, :] = (eig_q_dz - eig_d[2]) / q
            # print(eig)
            # print('eig_dx:')
            # print(eig_d[0])
            # print('eig_q_dx:')
            # print(eig_q_dx)
            # print('eig_ddx:')
            # print(eig_dd[0,0])
            list_o_k[ik] = ut.get_carrier_k(eig, eig_d, eig_dd, ef, eta)
            # print(list_o_k[ik], file=std)
        # std.close()
        sum_o = np.sum(list_o_k, axis=0) / (nkpts * ut.TwoPi)
        return sum_o
