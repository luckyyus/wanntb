import numpy as np
from datetime import datetime
from . import utility as ut


class TBSystem:

    def __init__(self, tb_file='wannier90_tb.dat'):
        data = ut.read_tb_file(tb_file=tb_file)
        self.seedname = data['seedname']
        self.num_wann = data['num_wann']
        self.real_lattice = data['real_lattice']
        self.recip_lattice = data['recip_lattice']
        self.n_Rpts = data['n_Rpts']
        self.ham_R = data['ham_R']
        self.R_vec = data['R_vec']
        self.n_degen = data['n_degen']
        self.r_mat_R = data['r_mat_R']
        # calculate wannier centers for each WF
        # print('fraction coordinate of WF centers:')
        for ir in range(self.n_Rpts):
            if self.R_vec[ir, :].dot(self.R_vec[ir, :]) < 0.001:
                self.iR0 = ir
                # WF centers in cart. coordinate
                self.wann_centers_cart = np.diagonal(self.r_mat_R[ir, :, :, :], axis1=1, axis2=2).T.real
                # print(self.wann_centers_cart)
                self.wann_centers_frac = (self.wann_centers_cart @ self.recip_lattice) / ut.TwoPi
                # print(self.wann_centers_frac)
                break

    def get_onsite_energy(self):
        return np.diagonal(self.ham_R[self.iR0, :, :]).real

    def get_spin_splitting(self):
        onsite = self.get_onsite_energy()
        return (- onsite[0:self.num_wann//2] + onsite[self.num_wann//2:self.num_wann]) / 2

    def get_eig_uu_for_one_kpt(self, kpt):
        fac = ut.fourier_phase_R_to_k(self.R_vec, self.n_degen, self.n_Rpts, kpt)
        ham_k = ut.fourier_R_to_k(self.ham_R,
                                  self.R_vec,
                                  fac,
                                  self.real_lattice,
                                  iout='0')[0]
        return np.linalg.eigh(ham_k)

    def get_eig_kpath(self, kpath, nkpts_path=100):
        kpts, kpts_len = ut.get_kpts_path(kpath, nkpts_path, self.recip_lattice)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)

    def get_eig_for_kpts_around(self, kmesh, center, distance_cart):
        kpts = ut.get_kpts_mesh_around(kmesh, center, distance_cart, self.recip_lattice)
        nk = kpts.shape[0]
        print('total number of kpoints for fitting: %d ' % nk)
        eigs = np.zeros((nk, self.num_wann), dtype=float)
        for ik in range(nk):
            kpt = kpts[ik]
            fac = ut.fourier_phase_R_to_k(self.R_vec, self.n_degen, self.n_Rpts, kpt)
            ham_k = ut.fourier_R_to_k(self.ham_R,
                                      self.R_vec,
                                      fac,
                                      self.real_lattice,
                                      iout='0')[0]
            eig, uu = np.linalg.eigh(ham_k)
            eigs[ik, :] = eig
        return eigs, kpts, kpts @ self.recip_lattice

    def get_alpha_beta(self, kmesh, ef, mag, eta=1e-3, q=1e-5, direction=1):
        start = datetime.now()
        print('relaxation time %e s' % (ut.Hbar_ / eta))
        kpts = ut.get_kpts_mesh(kmesh)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        # get the q vector in fraction coordinate
        q_frac = q * np.dot(ut.Cart[direction-1, :], self.recip_lattice) / ut.TwoPi
        print('q in fraction units: %s' % q_frac)
        list_o_k = np.zeros((nkpts, 3), dtype=float)
        # [alpha, beta_alpha*q*vd, beta_q*vs]
        for ik in range(nkpts):
            kpt = kpts[ik]
            ham_k, eig, eig_da, uu = ut.ham_eig_da_uu(self.ham_R, self.R_vec, self.n_degen,
                                                      self.n_Rpts, self.num_wann, self.real_lattice,
                                                      direction, kpt)
            # k + q
            ham_q, eig_q, eig_q_da, uu_q = ut.ham_eig_da_uu(self.ham_R, self.R_vec, self.n_degen,
                                                            self.n_Rpts, self.num_wann, self.real_lattice,
                                                            direction, kpt + q_frac)
            energy_spin = ut.get_spin_splitting(ham_k, self.num_wann)
            list_o_k[ik, :] = ut.get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, energy_spin, uu, uu_q,
                                                  self.num_wann, ef, eta, q)
        sum_o = np.sum(list_o_k, axis=0) / nkpts
        print(sum_o)
        alpha = sum_o[0] / (ut.TwoPi * 4 * mag)
        beta = sum_o[1] / sum_o[2] / 2
        print('time used: %24.6f <-- get_alpha_beta' % (datetime.now() - start).total_seconds())
        return alpha, beta

    def get_alpha_beta_kpath(self, kpath, ef, mag, eta=1e-3, q=1e-5, direction=1, nkpts_path=100):
        print('relaxation time %e s' % (ut.Hbar_ / eta))
        kpts, kpts_len = ut.get_kpts_path(kpath, nkpts_path, self.recip_lattice)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        q_frac = q * np.dot(ut.Cart[direction - 1, :], self.recip_lattice) / ut.TwoPi
        print('q in fraction units: %s' % q_frac)
        # kpts_len, sum_alpha_k, sum_qvd_k, sum_qv_k, sum_alpha_k(inter), sum_qvd_k(inter), sum_qv_k(inter)
        list_o_k = np.zeros((nkpts, 7), dtype=float)
        list_o_k[:, 0] = kpts_len
        for ik in range(nkpts):
            kpt = kpts[ik]
            ham_k, eig, eig_da, uu = ut.ham_eig_da_uu(self.ham_R,
                                                      self.R_vec,
                                                      self.n_degen,
                                                      self.n_Rpts,
                                                      self.num_wann,
                                                      self.real_lattice,
                                                      direction,
                                                      kpt)

            # k + q
            ham_q, eig_q, eig_q_da, uu_q = ut.ham_eig_da_uu(self.ham_R,
                                                            self.R_vec,
                                                            self.n_degen,
                                                            self.n_Rpts,
                                                            self.num_wann,
                                                            self.real_lattice,
                                                            direction,
                                                            kpt + q_frac)
            energy_spin = ut.get_spin_splitting(ham_k, self.num_wann)

            list_o_k[ik, 1:4] = ut.get_alpha_beta_k(eig, eig_q, eig_da, eig_q_da, energy_spin, uu, uu_q,
                                                    self.num_wann, ef, eta, q)
            list_o_k[ik, 4:7] = ut.get_alpha_beta_inter_k(eig, eig_q, eig_da, eig_q_da, energy_spin, uu, uu_q,
                                                          self.num_wann, ef, eta)
        return list_o_k

    def get_carrier(self, kmesh, ef, eta=1e-3, q=1e-5, direction=1):
        start = datetime.now()
        kpts = ut.get_kpts_mesh(kmesh)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        q_frac = q * ut.Cart @ self.recip_lattice / ut.TwoPi
        sum_o = ut.get_carrier_kpar(self.ham_R, self.R_vec, self.n_degen,
                                    self.n_Rpts, self.num_wann, self.real_lattice,
                                    direction, kpts, nkpts, q_frac, q, ef, eta)
        print('time used: %24.6f <-- get_carrier' % (datetime.now() - start).total_seconds())
        return sum_o / (nkpts * ut.TwoPi)
