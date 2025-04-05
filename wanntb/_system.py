import numpy as np
from datetime import datetime
from . import utility as ut
from .constant import Cart, TwoPi, Hbar_
from ._dos import get_occ_kpar, get_occ_proj_kpar
from ._berry import get_ahc_kpar_fermi, get_morb_berry_kpar_kpath, get_morb_berry_kpar
from ._alpha_beta import get_alpha_beta_kpar, get_alpha_beta_kpar_kpath, get_alpha_beta_efs_kpar
# spec = [
#     ('seedname', numba.core.string),
#     ('num_wann', int32),
#     ('real_lattice',float64[:]),
#     ('recip_lattice', float64[:]),
#     ('n_Rpts', int32),
#     ('ham_R', complex128[:,:,:]),
#
# ]


class TBSystem:

    def __init__(self, tb_file='wannier90_tb.dat', npz_file=None):
        start = datetime.now()
        if npz_file is None:
            data = ut.read_tb_file(tb_file=tb_file)
            self.seedname = data['seedname']
            self.real_lattice = data['real_lattice']
            self.recip_lattice = data['recip_lattice']
            _ham_R = data['ham_R']
            self.R_vec = data['R_vec']
            self.n_Rpts = data['n_Rpts']
            self.n_degen = data['n_degen']
            _r_mat_R = data['r_mat_R']
            for ir in range(self.n_Rpts):
                _ham_R[ir] /= self.n_degen[ir]
                _r_mat_R[ir] /= self.n_degen[ir]
            self.ham_R = ut.hermiization_R(_ham_R, self.R_vec)
            self.r_mat_R = ut.hermiization_R(_r_mat_R, self.R_vec)
        else:
            print("reading npz file %s " % npz_file)
            data = np.load(npz_file)
            print(data.files)
            print('seedname:', data['seedname'])
            print('real lattice:')
            print(data['real_lattice'])
            print('reciprocal lattice:')
            print(data['recip_lattice'])
            self.seedname = data['seedname']
            self.real_lattice = data['real_lattice']
            self.recip_lattice = data['recip_lattice']
            self.ham_R = data['ham_R']
            self.R_vec = data['R_vec']
            self.r_mat_R = data['r_mat_R']
            print('ham_R: %s %s' % (data['ham_R'].dtype, list(data['ham_R'].shape)))
            print('R_vec: %s %s' % (data['R_vec'].dtype, list(data['R_vec'].shape)))
            print('r_mat_R: %s %s' % (data['r_mat_R'].dtype, list(data['r_mat_R'].shape)))
            self.num_wann = self.ham_R.shape[1]
            self.n_Rpts = self.R_vec.shape[0]
            self.n_degen = np.ones(self.n_Rpts, dtype=np.uint8)
        self._Rvec = self.R_vec.astype(np.float64)
        for ir in range(self.n_Rpts):
            if self.R_vec[ir, :].dot(self.R_vec[ir, :]) == 0:
                self.iR0 = ir
                # WF centers in cart. coordinate
                self.wann_centers_cart = np.diagonal(self.r_mat_R[ir, :, :, :], axis1=1, axis2=2).T.real
                # print(self.wann_centers_cart)
                self.wann_centers_frac = (self.wann_centers_cart @ self.recip_lattice) / TwoPi
                # print(self.wann_centers_frac)
                break
        self.R_vec_cart_T = np.ascontiguousarray((self._Rvec @ self.real_lattice).T, dtype=float)
        self.volume = np.cross(self.real_lattice[0], self.real_lattice[1]).dot(self.real_lattice[2])
        print('unit cell volume: %.4f angst.^3' % self.volume)
        _ax = np.cross(self.real_lattice[1], self.real_lattice[2])
        _ay = np.cross(self.real_lattice[2], self.real_lattice[0])
        _az = np.cross(self.real_lattice[0], self.real_lattice[1])
        self.area = np.array([np.sqrt(_ax @ _ax), np.sqrt(_ay @ _ay), np.sqrt(_az @ _az)], dtype=np.float64)
        print('area for three direction:')
        print(self.area)
        print('time used: %24.2f <-- initialize TB system' % (datetime.now() - start).total_seconds())

    def output_npz(self, seedname='packaged'):
        start = datetime.now()
        filename = seedname + '-tb.npz'
        np.savez(filename,
                 seedname=seedname,
                 real_lattice=self.real_lattice,
                 recip_lattice=self.recip_lattice,
                 ham_R=self.ham_R,
                 R_vec=self.R_vec,
                 r_mat_R=self.r_mat_R)
        print(filename, ' is saved.')
        print('time used: %24.2f <-- output_npz' % (datetime.now() - start).total_seconds())

# the four functions below is usually for double check

    def r_cart_to_frac(self, r_cart):
        return r_cart @ self.recip_lattice / TwoPi

    def k_cart_to_frac(self, k_cart):
        return k_cart @ self.real_lattice / TwoPi

    def r_frac_to_cart(self, r_frac):
        return r_frac @ self.real_lattice

    def k_frac_to_cart(self, k_frac):
        return k_frac @ self.recip_lattice

    def get_onsite_energy(self):
        return np.diagonal(self.ham_R[self.iR0, :, :]).real

    def get_spin_splitting(self):
        onsite = self.get_onsite_energy()
        return (- onsite[0:self.num_wann//2] + onsite[self.num_wann//2:self.num_wann]) / 2

    def get_eig_uu_for_one_kpt(self, kpt):
        fac = ut.fourier_phase_R_to_k(self._Rvec, kpt)
        ham_k = ut.fourier_R_to_k(self.ham_R, self.R_vec_cart_T, fac, iout=[0])[0]
        eig, uu = np.linalg.eigh(ham_k)
        return eig, uu

    def get_ham_eig_da_uu_for_one_kpt(self, kpt, direction=1):
        fac = ut.fourier_phase_R_to_k(self._Rvec, kpt)
        out = ut.fourier_R_to_k(self.ham_R, self.R_vec_cart_T, fac, iout=[0, direction])
        ham_k, ham_k_da = out[0], out[direction]
        eig, uu = np.linalg.eigh(ham_k)
        eig_da = ut.get_eig_da(eig, ham_k_da, uu, self.num_wann)
        return eig, eig_da, np.diagonal(ut.unitary_trans(ham_k_da, uu)).real

    def output_bands_kpath(self, kpath, nkpts_path=100, filename='bands-debug.txt', spin=True):
        start = datetime.now()
        kpts, kpts_len = ut.get_kpts_path(kpath, nkpts_path, self.recip_lattice)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        eigs = ut.get_eig_for_kpts_kpar(self.ham_R, self._Rvec, self.R_vec_cart_T, self.num_wann, kpts)
        with open(filename, 'w') as outf:
            for ib in range(self.num_wann):
                for ik in range(nkpts):
                    outf.write('%16.8f%16.8f\n' % (kpts_len[ik], eigs[ik, ib]))
                outf.write('\n')
        print('time used: %24.2f <-- plot_bands_kpath' % (datetime.now() - start).total_seconds())

    def get_eig_for_kpts_around(self, kmesh, center, distance_cart):
        start = datetime.now()
        kpts = ut.get_kpts_mesh_around(kmesh, center, distance_cart, self.recip_lattice)
        nk = kpts.shape[0]
        print('total number of kpoints for fitting: %d ' % nk)
        eigs = ut.get_eig_for_kpts_kpar(self.ham_R, self._Rvec, self.R_vec_cart_T, self.num_wann, kpts)
        print('time used: %24.2f <-- get_eig_for_kpts_around' % (datetime.now() - start).total_seconds())
        return eigs, kpts, kpts @ self.recip_lattice

    def get_alpha_beta(self, kmesh, ef, mag, eta=1e-3, q=1e-6, direction=1, adpt_mesh=None):
        start = datetime.now()
        print('relaxation time %e s' % (Hbar_ / eta))
        kpts = ut.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        # get the q vector in fraction coordinate
        q_frac = q * Cart[direction-1, :] @ self.real_lattice / TwoPi
        print('q in fraction units: %s' % q_frac)
        e_s = self.get_spin_splitting()
        # adaptive k-mesh
        if adpt_mesh is not None:
            _adpt_mesh = np.array(adpt_mesh, dtype=np.int32)
            _dk = 1.0 / np.array(kmesh)
            print('_dk in fraction units: %s' % _dk)
            adpt_kpts = ut.get_adpt_kpts(_dk, _adpt_mesh)
            # print('adpt_kpts:')
            # print(adpt_kpts)
        else:
            adpt_kpts = np.zeros(3, dtype=np.float64)
        # [alpha, beta_alpha*q*vd, beta_q*vs]
        o_sum = get_alpha_beta_kpar(self.ham_R, self._Rvec, self.R_vec_cart_T,
                                       self.num_wann, direction, e_s, kpts, q_frac, q, ef, eta,
                                       adpt_kpts=adpt_kpts)
        print(o_sum)
        alpha = o_sum[0] / (TwoPi * 4 * mag)
        beta = o_sum[1] / o_sum[2] / 2
        ratio = beta / alpha
        print('time used: %24.2f <-- get_alpha_beta' % (datetime.now() - start).total_seconds())
        return alpha, o_sum[1], o_sum[2], beta, ratio

    def get_alpha_beta_fermi(self, kmesh, ef0, mu_d, n_ef, mag, eta=1e-3, q=1e-6, direction=1, adpt_mesh=None):
        start = datetime.now()
        print('relaxation time %e s' % (Hbar_ / eta))
        kpts = ut.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        mus = np.linspace(-mu_d, mu_d, n_ef+1, endpoint=True, dtype=float)
        efs = mus + ef0
        # print(efs)
        print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))
        # get the q vector in fraction coordinate
        q_frac = q * Cart[direction - 1, :] @ self.real_lattice / TwoPi
        print('q in fraction units: %s' % q_frac)
        e_s = self.get_spin_splitting()
        out = np.zeros((efs.shape[0], 6), dtype=float)
        # adaptive k-mesh
        if adpt_mesh is not None:
            _adpt_mesh = np.array(adpt_mesh, dtype=np.int32)
            _dk = 1.0 / np.array(kmesh)
            print('_dk in fraction units: %s' % _dk)
            adpt_kpts = ut.get_adpt_kpts(_dk, _adpt_mesh)
            # print('adpt_kpts:')
            # print(adpt_kpts)
        else:
            adpt_kpts = np.zeros(3, dtype=np.float64)
        # [alpha, beta_alpha*q*vd, beta_q*vs]
        o_sum = get_alpha_beta_efs_kpar(self.ham_R, self._Rvec, self.R_vec_cart_T,
                                       self.num_wann, direction, e_s, kpts, q_frac, q, efs, eta,
                                       adpt_kpts=adpt_kpts)
        print(o_sum.shape)
        alpha = o_sum[:, 0] / (TwoPi * 4 * mag)
        beta = o_sum[:, 1] / o_sum[:, 2] / 2
        ratio = beta / alpha
        out[:, 0] = mus
        out[:, 1] = alpha
        out[:, 2] = o_sum[:, 1]
        out[:, 3] = o_sum[:, 2]
        out[:, 4] = beta
        out[:, 5] = ratio
        print('time used: %24.2f <-- get_alpha_beta_fermi' % (datetime.now() - start).total_seconds())
        return out


    def get_alpha_beta_kpath(self, kpath, ef, mag, eta=1e-3, q=1e-6, direction=1, nkpts_path=100):
        start = datetime.now()
        print('relaxation time %e s' % (Hbar_ / eta))
        kpts, kpts_len = ut.get_kpts_path(kpath, nkpts_path, self.recip_lattice)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        q_frac = q * np.dot(Cart[direction - 1, :], self.real_lattice) / TwoPi
        print('q in fraction units: %s' % q_frac)
        e_s = self.get_spin_splitting()
        # kpts_len, sum_alpha_k, sum_qvd_k, sum_qv_k, sum_alpha_k(inter), sum_qvd_k(inter), sum_qv_k(inter)
        list_o_k = np.zeros((nkpts, 7), dtype=float)
        list_o_k[:, 0] = kpts_len
        list_o_k[:, 1:] = get_alpha_beta_kpar_kpath(self.ham_R, self._Rvec, self.R_vec_cart_T,
                                                       self.num_wann, direction, e_s, kpts, q_frac, q, ef, eta)
        print('time used: %24.2f <-- get_alpha_beta_kpath' % (datetime.now() - start).total_seconds())
        return list_o_k

    def get_carrier(self, kmesh, ef, eta=1e-3, q=1e-5, direction=1):
        start = datetime.now()
        kpts = ut.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        q_frac = q * Cart[direction-1, :] @ self.real_lattice / TwoPi
        sum_o = ut.get_carrier_kpar(self.ham_R, self.R_vec, self.R_vec_cart_T,
                                    self.num_wann, direction, kpts, q_frac, q, ef, eta)
        print('time used: %24.2f <-- get_carrier' % (datetime.now() - start).total_seconds())
        return sum_o

    def get_morb_berry_kpath(self, ef, kpath, nkpts_path=100, direction=1, eta=1e-4):
        start = datetime.now()
        kpts, kpts_len = ut.get_kpts_path(kpath, nkpts_path, self.recip_lattice)
        nkpts = kpts.shape[0]
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        list_o_k = np.zeros((nkpts, 2), dtype=float)
        list_o_k[:, 0] = kpts_len
        list_o_k[:, 1] = get_morb_berry_kpar_kpath(self.ham_R, self.r_mat_R, self._Rvec, self.R_vec_cart_T,
                                                      self.num_wann, kpts, ef, eta, direction)
        print('time used: %24.2f <-- get_morb_berry_kpath' % (datetime.now() - start).total_seconds())
        return list_o_k

    def get_morb_berry_kmesh(self, ef, kmesh, direction=1, eta=1e-4):
        start = datetime.now()
        kpts = ut.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        print('E-fermi: %8.4f' % ef)
        morb = get_morb_berry_kpar(self.ham_R, self.r_mat_R, self._Rvec, self.R_vec_cart_T,
                                                      self.num_wann, kpts, ef, eta, direction)
        print('time used: %24.2f <-- get_morb_berry_kmesh' % (datetime.now() - start).total_seconds())
        return morb

    def get_ahc_kmesh_fermi(self, kmesh, ef_min, ef_max, n_ef, eta=1e-4):
        start = datetime.now()
        kpts = ut.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        efs = np.linspace(ef_min, ef_max, n_ef+1, endpoint=True, dtype=float)
        print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))
        ahc_efs = get_ahc_kpar_fermi(self.ham_R, self.r_mat_R, self._Rvec, self.R_vec_cart_T,
                                        self.num_wann, kpts, efs, eta)
        output = np.zeros((efs.shape[0], 4), dtype=float)
        output[:, 0] = efs
        output[:, 1:] = ahc_efs / self.area
        print('time used: %24.2f <-- get_ahc_kmesh_fermi' % (datetime.now() - start).total_seconds())
        return output

    def get_occ_kmesh_fermi(self, kmesh, ef_min, ef_max, n_ef, eta=1e-4, lproj=False):
        start = datetime.now()
        kpts = ut.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        efs = np.linspace(ef_min, ef_max, n_ef+1, endpoint=True, dtype=float)
        print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))
        if lproj:
            # occ_p_efs[n_ef, n_proj]
            occ_p_efs = get_occ_proj_kpar(self.ham_R, self._Rvec, self.R_vec_cart_T,
                                          self.num_wann, kpts, efs, eta)
            occ_efs = np.sum(occ_p_efs, axis=1)
            output = np.zeros((efs.shape[0], 2+self.num_wann), dtype=float)
            output[:, 0] = efs
            output[:, 1] = occ_efs
            output[:, 2:] = occ_p_efs
        else:
            occ_efs = get_occ_kpar(self.ham_R, self._Rvec, self.R_vec_cart_T, kpts, efs, eta)
            output = np.zeros((efs.shape[0], 2), dtype=float)
            output[:, 0] = efs
            output[:, 1] = occ_efs
        print('time used: %24.2f <-- get_occ_kmesh_fermi' % (datetime.now() - start).total_seconds())
        return output
