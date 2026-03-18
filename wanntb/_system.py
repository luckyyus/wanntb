from typing import Tuple, List
import numpy as np
from numpy.typing import NDArray
from datetime import datetime

from . import io
from . import kpoints as kp
from . import utility as ut
from .constant import Cart, TwoPi, Hbar_
from ._orbit import orbital_info
from ._dos import get_occ_dos_kpar, get_occ_dos_proj_kpar
from ._berry import berry_fermi, berry_kpath, intra_shc_fermi, get_OHE_kpar_kmesh, get_OHE_kpar_kmesh_fermi, axion_fermi
from ._edelstein import edelstein_fermi
from ._alpha_beta import get_alpha_beta_kpar, get_alpha_beta_kpar_kpath, get_alpha_beta_efs_kpar
from . import _old as od


class TBSystem:

    def __init__(self):
        start = datetime.now()
        # basic data
        self.seedname = 'tb-system' # npz
        self.real_lattice: NDArray[np.float64]|None = None # npz
        self.recip_lattice: NDArray[np.float64]|None = None
        self.ham_R: NDArray[np.complex128]|None = None # npz
        self.R_vec: NDArray[np.int16]|None = None  # npz
        self.r_mat_R: NDArray[np.complex128] | None = None # npz

        # spin data
        self.ss_R: NDArray[np.complex128]|None = None # npz
        # structure data
        self.atom_pos: NDArray[np.float64]|None = None # npz
        self.atom_names: List[str]|None = None # npz
        self.atom_spec: NDArray[np.int16]|None = None # npz
        self.atom_counts: NDArray[np.int_]|None = None # npz
        self.n_atoms: int = 0 #

        # orbital data
        self.orb_pos: NDArray[np.float64]|None = None # npz
        self.orb_lmsr: NDArray[np.uint8]|None = None # npz
        self.orb_laxis: NDArray[np.float64]|None = None # npz
        self.orb_is_laxis: bool = False
        self.n_orbs: int = 0

        # post initialized data
        self.n_degen: NDArray[np.uint8] | None = None
        self.num_wann: int = 0
        self.n_Rpts: int = 0
        self._ss_R: NDArray[np.complex128]|None = None
        self._ham_RT: NDArray[np.complex128]|None = None
        self._r_RT: NDArray[np.complex128]|None = None
        self._Rvec: NDArray[np.float64]|None = None
        self._R_cartT: NDArray[np.float64]|None = None

        self.iR0: int = 0
        self.wann_centers_cart: NDArray[np.float64] | None = None
        self.wann_centers_frac: NDArray[np.float64] | None = None
        self.volume: float | None = None
        self.area: NDArray[np.float64] | None = None

    def post_init(self):
        self.num_wann = self.ham_R.shape[1]
        self.n_Rpts = self.R_vec.shape[0]
        self.n_degen = np.ones(self.n_Rpts, dtype=np.uint8)

        self._ss_R = np.ascontiguousarray(self.ss_R.transpose((1, 2, 3, 0))) if self.ss_R is not None else \
            np.zeros((3, self.num_wann, self.num_wann, self.n_Rpts), dtype=np.complex128)
        # [num_wann, num_wann, n_Rpts]
        self._ham_RT = np.ascontiguousarray(self.ham_R.transpose((1, 2, 0)))
        # [3, num_wann, num_wann, n_Rpts]
        self._r_RT = np.ascontiguousarray(self.r_mat_R.transpose((1, 2, 3, 0)))
        # [n_Rpts, 3] float64
        self._Rvec = np.ascontiguousarray(self.R_vec.astype(np.float64))
        # [3, n_Rpts] float64
        self._R_cartT = np.ascontiguousarray((self._Rvec @ self.real_lattice).T)
        for ir in range(self.n_Rpts):
            if self.R_vec[ir, :].dot(self.R_vec[ir, :]) == 0:
                self.iR0 = ir
                # WF centers in cart. coordinate
                self.wann_centers_cart = np.diagonal(self.r_mat_R[ir, :, :, :], axis1=1, axis2=2).T.real
                # print(self.wann_centers_cart)
                self.wann_centers_frac = (self.wann_centers_cart @ self.recip_lattice) / TwoPi
                # print(self.wann_centers_frac)
                break

        self.volume = np.cross(self.real_lattice[0], self.real_lattice[1]).dot(self.real_lattice[2])
        print('unit cell volume: %.4f angst.^3' % self.volume)
        _ax = np.cross(self.real_lattice[1], self.real_lattice[2])
        _ay = np.cross(self.real_lattice[2], self.real_lattice[0])
        _az = np.cross(self.real_lattice[0], self.real_lattice[1])
        self.area = np.array([np.sqrt(_ax @ _ax), np.sqrt(_ay @ _ay), np.sqrt(_az @ _az)], dtype=np.float64)
        print('area for three direction (maybe wrong):')
        print(f'{self.area[0]:10.4f}{self.area[1]:10.4f}{self.area[2]:10.4f}')


    def load_spins(self, ss_file='wannier90_SS_R.dat'):
        start = datetime.now()
        print('---------- start load_spins ----------')
        self.ss_R = io.read_spin_file(self.R_vec, self.n_Rpts, self.num_wann, ss_file=ss_file)
        self._ss_R = np.ascontiguousarray(self.ss_R.transpose((1, 2, 3, 0)))
        print('time used: %24.2f <-- load_spins' % (datetime.now() - start).total_seconds())

    def load_poscar(self, pos_file='POSCAR'):
        start = datetime.now()
        print('---------- start load_poscar ----------')
        _, self.atom_pos, self.atom_names, self.atom_counts = io.read_poscar(pos_file, self.real_lattice)
        self.n_atoms = self.atom_pos.shape[0]
        print('number of atoms: %d' % self.n_atoms)
        print('atom_pos: %s %s' % (self.atom_pos.dtype, list(self.atom_pos.shape)))
        print('atom_names: %s' % self.atom_names)
        print('atom_counts: %s' % self.atom_counts)
        self.atom_spec = np.zeros(self.n_atoms, dtype=np.int16)
        begin = 0
        for i_spec in range(len(self.atom_names)):
            self.atom_spec[begin: begin+self.atom_counts[i_spec]] = i_spec
            begin += self.atom_counts[i_spec]
        print('atom_spec: %s' % self.atom_spec)
        print('time used: %24.2f <-- load_poscar' % (datetime.now() - start).total_seconds())

    def load_orbitals(self, projections, is_laxis=False, is_soc=True, order='uudd'):
        start = datetime.now()
        print('---------- start load_orbitals ----------')
        self.orb_pos, self.orb_lmsr, orb_laxis = orbital_info(projections,
                                                              self.real_lattice,
                                                              self.atom_pos, self.atom_names, self.atom_counts,
                                                              is_soc=is_soc,
                                                              order=order)
        self.orb_is_laxis = is_laxis
        self.n_orbs = self.orb_pos.shape[0]
        print('number of orbitals: %d' % self.n_orbs)
        assert self.n_orbs == self.num_wann, 'number of orbitals should be the same with the number of WFs.'

        print('orb_pos: %s %s' % (self.orb_pos.dtype, list(self.orb_pos.shape)))
        print('orb_lmsr: %s %s' % (self.orb_lmsr.dtype, list(self.orb_lmsr.shape)))
        if is_laxis:
            self.orb_laxis = orb_laxis
            print('orb_laxis: %s %s' % (self.orb_laxis.dtype, list(self.orb_laxis.shape)))
        print('time used: %24.2f <-- load_orbitals' % (datetime.now() - start).total_seconds())

    def output_npz(self, seedname='packaged'):
        start = datetime.now()
        filename = seedname + '-tb.npz'
        array_dict = {'seedname': seedname,
                      'real_lattice': self.real_lattice,
                      'ham_R': self.ham_R,
                      'R_vec': self.R_vec,
                      'r_mat_R': self.r_mat_R}
        if self.ss_R is not None:
            array_dict['ss_R'] = self.ss_R
        if self.n_atoms > 0:
            array_dict['atom_pos'] = self.atom_pos
            array_dict['atom_names'] = self.atom_names
            array_dict['atom_counts'] = self.atom_counts
            array_dict['atom_spec'] = self.atom_spec
        if self.n_orbs > 0:
            array_dict['orb_pos'] = self.orb_pos
            array_dict['orb_lmsr'] = self.orb_lmsr
            if self.orb_is_laxis: array_dict['orb_laxis'] = self.orb_laxis
        np.savez(filename, **array_dict)
        print(filename, ' is saved.')
        print('time used: %24.2f <-- output_npz' % (datetime.now() - start).total_seconds())

    def get_ham_one_R(self, rv):
        ir = ut.find_R_vec(rv, self.R_vec)
        return self.ham_R[ir]
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
        return np.diag(self.ham_R[self.iR0, :, :]).real

    def get_spin_splitting(self):
        onsite = self.get_onsite_energy()
        return (- onsite[0:self.num_wann//2] + onsite[self.num_wann//2:self.num_wann]) / 2

    def get_eig_uu_for_one_kpt(self, kpt):
        fac = ut.fourier_phase_R_to_k(self._Rvec, kpt)
        ham_k = ut.fourier_R_to_k(self._ham_RT, self._R_cartT, fac, iout=[0])[0]
        eig, uu = np.linalg.eigh(ham_k)
        return eig, uu

    def get_ham_eig_da_uu_for_one_kpt(self, kpt, direction=1):
        fac = ut.fourier_phase_R_to_k(self._Rvec, kpt)
        out = ut.fourier_R_to_k(self._ham_RT, self._R_cartT, fac, iout=[0, direction])
        ham_k, ham_k_da = out[0], out[direction]
        eig, uu = np.linalg.eigh(ham_k)
        eig_da = ut.get_eig_da(eig, ham_k_da, uu)
        return eig, eig_da, np.diagonal(ut.unitary_trans(ham_k_da, uu)).real

    def output_bands_kpath(self, kpath, nkpts_path=100, filename='bands-debug.txt'):
        start = datetime.now()
        print('---------- start plot_bands_kpath ----------')
        kpts, kpts_len = kp.get_kpts_path(kpath, nkpts_path, self.recip_lattice)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        eigs = ut.get_eig_for_kpts_kpar(self._ham_RT, self._Rvec, self._R_cartT, self.num_wann, kpts)
        with open(filename, 'w') as outf:
            for ib in range(self.num_wann):
                for ik in range(nkpts):
                    outf.write('%16.8f%16.8f\n' % (kpts_len[ik], eigs[ik, ib]))
                outf.write('\n')
        print('time used: %24.2f <-- plot_bands_kpath' % (datetime.now() - start).total_seconds())

    def get_eig_for_kpts_around(self, kmesh, center, distance_cart):
        start = datetime.now()
        print('---------- start get_eig_for_kpts_around ----------')
        kpts = kp.get_kpts_mesh_around(kmesh, center, distance_cart, self.recip_lattice)
        nk = kpts.shape[0]
        print('total number of kpoints for fitting: %d ' % nk)
        eigs = ut.get_eig_for_kpts_kpar(self._ham_RT, self._Rvec, self._R_cartT, self.num_wann, kpts)
        print('time used: %24.2f <-- get_eig_for_kpts_around' % (datetime.now() - start).total_seconds())
        return eigs, kpts, kpts @ self.recip_lattice

    def get_alpha_beta(self, kmesh, ef, mag, eta=1e-3, q=1e-6, direction=1, adpt_mesh=None):
        start = datetime.now()
        print('---------- start get_alpha_beta ----------')
        print('relaxation time %e ps' % (Hbar_ / eta))
        kpts = kp.get_kpts_mesh(kmesh)
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
            adpt_kpts = kp.get_adpt_kpts(_dk, _adpt_mesh)
            # print('adpt_kpts:')
            # print(adpt_kpts)
        else:
            adpt_kpts = np.zeros(3, dtype=np.float64)
        # [alpha, beta_alpha*q*vd, beta_q*vs]
        o_sum = get_alpha_beta_kpar(self._ham_RT, self._Rvec, self._R_cartT,
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
        print('---------- start get_alpha_beta_fermi ----------')
        print('relaxation time %e ps' % (Hbar_ / eta))
        kpts = kp.get_kpts_mesh(kmesh)
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
            adpt_kpts = kp.get_adpt_kpts(_dk, _adpt_mesh)
            # print('adpt_kpts:')
            # print(adpt_kpts)
        else:
            adpt_kpts = np.zeros(3, dtype=np.float64)
        # [alpha, beta_alpha*q*vd, beta_q*vs]
        o_sum = get_alpha_beta_efs_kpar(self._ham_RT, self._Rvec, self._R_cartT,
                                        self.num_wann, direction, e_s, kpts, q_frac, q, efs, eta,
                                        adpt_kpts=adpt_kpts)
        print(o_sum.shape)
        alpha = o_sum[:, 0] / (TwoPi * 4 * mag)
        beta = o_sum[:, 1] / o_sum[:, 2] / 2
        ratio = beta / alpha
        out = np.column_stack((mus, alpha, o_sum[:, 1], o_sum[:, 2], beta, ratio))
        print('time used: %24.2f <-- get_alpha_beta_fermi' % (datetime.now() - start).total_seconds())
        return out

    def get_alpha_beta_kpath(self, kpath, ef, mag, eta=1e-3, q=1e-6, direction=1, nkpts_path=100):
        start = datetime.now()
        print('---------- start get_alpha_beta_kpath ----------')
        print('relaxation time %e ps' % (Hbar_ / eta))
        kpts, kpts_len = kp.get_kpts_path(kpath, nkpts_path, self.recip_lattice)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        q_frac = q * np.dot(Cart[direction - 1, :], self.real_lattice) / TwoPi
        print('q in fraction units: %s' % q_frac)
        e_s = self.get_spin_splitting()
        # kpts_len, sum_alpha_k, sum_qvd_k, sum_qv_k, sum_alpha_k(inter), sum_qvd_k(inter), sum_qv_k(inter)
        list_o_k = np.zeros((nkpts, 7), dtype=float)
        list_o_k[:, 0] = kpts_len
        list_o_k[:, 1:] = get_alpha_beta_kpar_kpath(self._ham_RT, self._Rvec, self._R_cartT,
                                                    self.num_wann, direction, e_s, kpts, q_frac, q, ef, eta)
        print('time used: %24.2f <-- get_alpha_beta_kpath' % (datetime.now() - start).total_seconds())
        return list_o_k

    def get_carrier(self, kmesh, ef, eta=1e-3, q=1e-5, direction=1):
        start = datetime.now()
        print('---------- start get_carrier ----------')
        kpts = kp.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        q_frac = q * Cart[direction-1, :] @ self.real_lattice / TwoPi
        sum_o = ut.get_carrier_kpar(self._ham_RT, self.R_vec, self._R_cartT,
                                    self.num_wann, direction, kpts, q_frac, q, ef, eta)
        print('time used: %24.2f <-- get_carrier' % (datetime.now() - start).total_seconds())
        return sum_o

    def old_berrycurv_kpath(self, ef, kpath, nkpts_path=100, eta=1e-4, mode=0, q=0.0):
        start = datetime.now()
        print('---------- start get_berrycurv_kpath ----------')
        kpts, kpts_len = kp.get_kpts_path(kpath, nkpts_path, self.recip_lattice)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        q_frac = q * self.real_lattice / TwoPi if q > 1e-16 else None
        omega = od.get_berrycurv_kpar_kpath(self._ham_RT, self._r_RT, self._Rvec, self._R_cartT,
                                         self.num_wann, kpts, ef, eta, mode=mode, q_frac=q_frac, q=q)
        list_o_k = np.column_stack((kpts_len, omega))
        print('time used: %24.2f <-- get_berrycurv_kpath' % (datetime.now() - start).total_seconds())
        return list_o_k

    # def get_ahc_fermi(self, kmesh: tuple[int, int, int],
    #                   ef_range: tuple[float, float, int],
    #                   eta=1e-4, mode=0, subwf=None):
    #     start = datetime.now()
    #     print('---------- start get_ahc_kmesh_fermi ----------')
    #     kpts = kp.get_kpts_mesh(kmesh)
    #     print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
    #     ef_min, ef_max, n_ef = ef_range[0], ef_range[1], ef_range[2]
    #     efs = np.linspace(ef_min, ef_max, n_ef+1, endpoint=True, dtype=float)
    #     print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))
    #     ahc_efs = get_ahc_kpar_fermi(self._ham_RT, self._r_RT, self._Rvec, self._R_cartT,
    #                                  self.num_wann, kpts, efs, eta, mode=mode, subwf=subwf)
    #     print('ahc_efs:', type(ahc_efs), ahc_efs.shape)
    #     ahc_efs /= self.area
    #     output = np.column_stack((efs, ahc_efs))
    #     print('time used: %24.2f <-- get_ahc_kmesh_fermi' % (datetime.now() - start).total_seconds())
    #     return output

    def old_shc_fermi(self, kmesh: tuple[int, int, int],
                      ef_range: tuple[float, float, int],
                      eta=1e-4, xyz=2, subwf=None):
        start = datetime.now()
        print('---------- start get_shc_kmesh_fermi ----------')
        if self.ss_R is None:
            print('spin data ss_R is missing.')
            return
        kpts = kp.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        ef_min, ef_max, n_ef = ef_range[0], ef_range[1], ef_range[2]
        efs = np.linspace(ef_min, ef_max, n_ef + 1, endpoint=True, dtype=float)
        print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))
        shc_efs = od.get_shc_kpar_fermi(self._ham_RT, self._r_RT, self._Rvec, self._ss_R, self._R_cartT,
                                     self.num_wann, kpts, efs, eta, xyz, subwf=subwf)
        shc_efs /= self.volume
        output = np.column_stack((efs, shc_efs))
        print('time used: %24.2f <-- get_shc_kmesh_fermi' % (datetime.now() - start).total_seconds())
        return output

    def get_occ_dos_fermi(self, kmesh: tuple[int, int, int],
                          ef_range: tuple[float, float, int],
                          eta=1e-4, lproj=False):
        start = datetime.now()
        print('---------- start get_occ_dos_kmesh_fermi ----------')
        kpts = kp.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        ef_min, ef_max, n_ef = ef_range[0], ef_range[1], ef_range[2]
        efs = np.linspace(ef_min, ef_max, n_ef+1, endpoint=True, dtype=float)
        print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))
        if lproj:
            # occ_p_efs[n_ef, n_proj]
            occ_p_efs, dos_p_efs = get_occ_dos_proj_kpar(self._ham_RT, self._Rvec, self._R_cartT,
                                                         self.num_wann, kpts, efs, eta)
            occ_efs = np.sum(occ_p_efs, axis=1)
            dos_efs = np.sum(dos_p_efs, axis=1)
            out_occ = np.column_stack((efs, occ_efs, occ_p_efs))
            out_dos = np.column_stack((efs, dos_efs, dos_p_efs))
        else:
            occ_efs, dos_efs = get_occ_dos_kpar(self._ham_RT, self._Rvec, self._R_cartT, kpts, efs, eta)
            out_occ = np.column_stack((efs, occ_efs))
            out_dos = np.column_stack((efs, dos_efs))
        print('time used: %24.2f <-- get_occ_dos_kmesh_fermi' % (datetime.now() - start).total_seconds())
        return out_occ, out_dos

    def berry_calc_fermi(self, tasks: str,
                         kmesh: tuple[int, int, int],
                         ef_range: tuple[float, float, int],
                         eta=1e-4, xyz=2, subwf=None):
        start = datetime.now()
        print('---------- start berry_calc_fermi ----------')
        itasks, begin_idx, count = ut.get_itasks(tasks)
        print('itasks:', itasks)
        print('begin_idx:', begin_idx)
        if self.ss_R is None and 10 in itasks:
            print('spin data ss_R is missing.')
            return
        kpts = kp.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        ef_min, ef_max, n_ef = ef_range[0], ef_range[1], ef_range[2]
        efs = np.linspace(ef_min, ef_max, n_ef + 1, endpoint=True, dtype=float)
        print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))
        out = berry_fermi(itasks, self._ham_RT, self._r_RT, self._Rvec, self._R_cartT,
                                     self.num_wann, kpts, efs, eta, xyz=xyz, ss_R=self._ss_R, subwf=subwf)
        for it in itasks:
            if it == 0: # ahc
                out[:, begin_idx[it]: begin_idx[it]+3] /= self.volume
            if it == 10: # shc
                out[:, begin_idx[it]: begin_idx[it] + 3] /= self.volume
        output = np.column_stack((efs, out))
        print('time used: %24.2f <-- berry_calc_fermi' % (datetime.now() - start).total_seconds())
        return output

    def berry_calc_kpath(self, tasks: str, ef: float, kpath, nkpts_path=100, eta=1e-4, xyz=2, subwf=None):
        start = datetime.now()
        print('---------- start berry_calc_kpath ----------')
        itasks, begin_idx, count = ut.get_itasks(tasks)
        print('itasks:', itasks)
        print('begin_idx:', begin_idx)
        if self.ss_R is None and 10 in itasks:
            print('spin data ss_R is missing.')
            return
        kpts, kpts_len = kp.get_kpts_path(kpath, nkpts_path, self.recip_lattice)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        out = berry_kpath(itasks, self._ham_RT, self._r_RT, self._Rvec, self._R_cartT,
                          self.num_wann, kpts, ef, eta, xyz=xyz, ss_R=self.ss_R, subwf=subwf)
        output = np.column_stack((kpts_len, out))
        print('time used: %24.2f <-- berry_calc_kpath' % (datetime.now() - start).total_seconds())
        return output
    
    def edelstein_calc_fermi(self,
                         kmesh: tuple[int, int, int],
                         ef_range: tuple[float, float,int],
                         eta=1e-3,
                         eta_intra=1e-2,
                         subwf=None):
        start = datetime.now()
        print('---------- start edelstein_calc_fermi ----------')
        if self.ss_R is None :
            print('spin data ss_R is missing.')
            return
        kpts = kp.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))
        ef_min, ef_max, n_ef = ef_range[0], ef_range[1], ef_range[2]
        efs = np.linspace(ef_min, ef_max, n_ef + 1, endpoint=True, dtype=float)
        print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))
        inter_efs, intra_efs = edelstein_fermi(self._ham_RT, self._r_RT, self._Rvec, self._R_cartT,
                                     self.num_wann, kpts, efs, eta, eta_intra,  ss_R=self._ss_R, subwf=subwf)
    
        # inter_efs /= self.volume
        # intra_efs /= self.volume
        # out_inter = np.column_stack((efs, inter_efs))
        # out_intra = np.column_stack((efs, intra_efs))
        output = np.column_stack((efs, inter_efs, intra_efs))
        print('time used: %24.2f <-- berry_calc_fermi' % (datetime.now() - start).total_seconds())
        return output

    def berry_calc_intra_shc_fermi(self, kmesh: tuple[int, int, int], ef_range: tuple[float, float, int],
                                   eta=1e-3, xyz=2, subwf=None):
        start = datetime.now()
        print('---------- start berry_calc_intra_shc_fermi----------')

        if self.ss_R is None:
            print('Error: spin data ss_R is missing.')
            return None

        kpts = kp.get_kpts_mesh(kmesh)
        ef_min, ef_max, n_ef = ef_range[0], ef_range[1], ef_range[2]
        efs = np.linspace(ef_min, ef_max, n_ef + 1, endpoint=True, dtype=float)

        out = intra_shc_fermi(self._ham_RT, self._r_RT, self._Rvec, self._R_cartT, self._ss_R,
                                         self.num_wann, kpts, efs, eta, xyz, subwf=subwf)

        shc_out = out / self.volume

        output = np.column_stack((efs, shc_out))
        print('time used: %24.2f <-- berry_calc_intra_shc_fermi' % (datetime.now() - start).total_seconds())
        return output


    def axion_calc_fermi(self, kmesh: tuple[int, int, int], ef_range: tuple[float, float, int],
                         eta=1e-4, mode=0, subwf=None):

        start = datetime.now()
        print('---------- start axion_calc_fermi ----------')

        kpts = kp.get_kpts_mesh(kmesh)
        print('k-points: %s %s' % (kpts.dtype, list(kpts.shape)))

        ef_min, ef_max, n_ef = ef_range[0], ef_range[1], ef_range[2]
        efs = np.linspace(ef_min, ef_max, n_ef + 1, endpoint=True, dtype=float)
        print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))

        theta = axion_fermi(self._ham_RT, self._r_RT, self._Rvec, self._R_cartT, self.num_wann, kpts, efs, eta, mode,
                            subwf=subwf)
        theta /= self.volume
        output = np.column_stack((efs, theta))
        print('time used: %24.2f <-- axion_calc_fermi' % (datetime.now() - start).total_seconds())
        return output

    def get_OHE_kmesh_sys(self, kmesh, ef, dir):
        start = datetime.now()
        print('---------- start spin_moment_kpath ----------')
        kpts = kp.get_kpts_mesh(kmesh)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        OHE = get_OHE_kpar_kmesh(self._ham_RT, self._r_RT, self._Rvec, self._R_cartT, self.num_wann, kpts, ef, dir)
        # list_o_k = np.column_stack((ef,spin_moment))
        print('time used: %24.2f <-- get_morb_berry_kpath' % (datetime.now() - start).total_seconds())
        return OHE / self.volume * 24300  # unit is S/cm

    def get_OHE_kmesh_fermi_sys(self, kmesh, ef_range, dir):
        start = datetime.now()
        print('---------- start OHE_kmesh_fermi_sys ----------')
        kpts = kp.get_kpts_mesh(kmesh)
        nkpts = kpts.shape[0]
        print('total number of k-points: %d' % nkpts)
        ef_min, ef_max, n_ef = ef_range[0], ef_range[1], int(ef_range[2])
        efs = np.linspace(ef_min, ef_max, n_ef + 1, endpoint=True, dtype=float)
        print('E_fermi_list: %s %s' % (efs.dtype, list(efs.shape)))

        OHE = get_OHE_kpar_kmesh_fermi(self._ham_RT, self._r_RT, self._Rvec, self._R_cartT,
                                       self.num_wann, kpts, efs, dir)
        print('time used: %24.2f <-- get_OHE_kmesh_fermi_sys' % (datetime.now() - start).total_seconds())

        list_o_k = np.column_stack((efs, OHE / self.volume * 24300))
        return list_o_k  # unit is S/cm

def get_tbsystem_by_new_ham(tb_in: TBSystem, ham_R_new, r_mat_R_new, R_vec_new, ss_R_new=None):
    start = datetime.now()
    print('---------- start get_tbsystem_by_new_ham ----------')
    tb = TBSystem()
    tb.seedname = tb_in.seedname
    tb.real_lattice = tb_in.real_lattice.copy()
    print('real lattice:')
    print(tb.real_lattice)
    tb.recip_lattice = np.linalg.inv(tb.real_lattice).T * TwoPi
    print('reciprocal lattice:')
    print(tb.recip_lattice)
    assert ham_R_new.shape[1:] == tb_in.ham_R.shape[1:], 'the dimension of Hamiltonian mismatch'
    tb.ham_R = ham_R_new
    tb.R_vec = R_vec_new
    n_Rpts_new = R_vec_new.shape[0]
    print('ham_R: %s %s' % (tb.ham_R.dtype, list(tb.ham_R.shape)))
    print('R_vec: %s %s' % (tb.R_vec.dtype, list(tb.R_vec.shape)))
    if r_mat_R_new is not None:
        tb.r_mat_R = r_mat_R_new
    else:
        if n_Rpts_new == tb_in.n_Rpts: tb.r_mat_R = tb_in.r_mat_R.copy()
    assert tb.r_mat_R is not None, 'new r_mat_R is missing'
    print('r_mat_R: %s %s' % (tb.r_mat_R.dtype, list(tb.r_mat_R.shape)))

    if ss_R_new is not None:
        tb.ss_R = ss_R_new
        print('ss_R: %s %s' % (tb.ss_R.dtype, list(tb.ss_R.shape)))
    if tb_in.atom_pos is not None:
        tb.atom_pos = tb_in.atom_pos.copy()
        tb.atom_names = tb_in.atom_names.copy()
        tb.atom_counts = tb_in.atom_counts.copy()
        tb.atom_spec = tb_in.atom_spec.copy()
        tb.n_atoms = tb.atom_pos.shape[0]
        print('atom_pos: %s %s' % (tb.atom_pos.dtype, list(tb.atom_pos.shape)))
        print('atom_names: %s' % tb.atom_names)
        print('atom_counts: %s' % tb.atom_counts)
        print('atom_spec: %s' % tb.atom_spec)
    if tb_in.orb_pos is not None:
        tb.orb_pos = tb_in.orb_pos.copy()
        tb.orb_lmsr = tb_in.orb_lmsr.copy()
        tb.n_orbs = tb.orb_pos.shape[0]
        print('orb_pos: %s %s' % (tb.orb_pos.dtype, list(tb.orb_pos.shape)))
        print('orb_lms: %s %s' % (tb.orb_lmsr.dtype, list(tb.orb_lmsr.shape)))
        if tb_in.orb_is_laxis:
            tb.orb_is_laxis = True
            tb.orb_laxis = tb_in.orb_laxis.copy()
            print('orb_laxis: %s %s' % (tb.orb_laxis.dtype, list(tb.orb_laxis.shape)))
        else:
            tb.orb_is_laxis = False
    tb.post_init()
    print('time used: %24.2f <-- get_tbsystem_by_new_ham' % (datetime.now() - start).total_seconds())
    return tb

def get_tbsystem_by_tb_file(tb_file='wannier90_tb.dat'):
    start = datetime.now()
    print('---------- start get_tbsystem_by_tb_file ----------')
    tb = TBSystem()

    data = io.read_tb_file(tb_file=tb_file)
    tb.seedname = data['seedname']
    tb.num_wann = data['num_wann']
    tb.real_lattice = data['real_lattice']
    tb.recip_lattice = data['recip_lattice']
    tb.ham_R = data['ham_R']
    tb.R_vec: NDArray[np.int16] = data['R_vec']
    tb.n_Rpts = data['n_Rpts']
    tb.n_degen = data['n_degen']
    tb.r_mat_R = data['r_mat_R']

    tb.post_init()
    print('time used: %24.2f <-- get_tbsystem_by_tb_file' % (datetime.now() - start).total_seconds())
    return tb

def get_tbsystem_by_npz_file(npz_file='wannier90_npz.dat'):
    start = datetime.now()
    print('---------- start get_tbsystem_by_npz_file ----------')
    tb = TBSystem()

    print("reading npz file %s " % npz_file)
    data = np.load(npz_file)
    print(data.files)
    print('seedname:', data['seedname'])
    tb.seedname = data['seedname']
    tb.real_lattice = data['real_lattice']
    print('real lattice:')
    print(tb.real_lattice)
    tb.recip_lattice = np.linalg.inv(tb.real_lattice).T * TwoPi
    print('reciprocal lattice:')
    print(tb.recip_lattice)
    tb.ham_R = data['ham_R']
    tb.R_vec = data['R_vec']
    tb.r_mat_R = data['r_mat_R']
    print('ham_R: %s %s' % (data['ham_R'].dtype, list(data['ham_R'].shape)))
    print('R_vec: %s %s' % (data['R_vec'].dtype, list(data['R_vec'].shape)))
    print('r_mat_R: %s %s' % (data['r_mat_R'].dtype, list(data['r_mat_R'].shape)))
    # tb.num_wann = tb.ham_R.shape[1]
    # tb.n_Rpts = tb.R_vec.shape[0]
    # tb.n_degen = np.ones(tb.n_Rpts, dtype=np.uint8)
    if 'ss_R' in data.files:
        tb.ss_R = data['ss_R']
        print('ss_R: %s %s' % (data['ss_R'].dtype, list(data['ss_R'].shape)))
    if 'atom_pos' in data.files:
        tb.atom_pos = data['atom_pos']
        tb.atom_names = data['atom_names']
        tb.atom_counts = data['atom_counts']
        tb.atom_spec = data['atom_spec']
        tb.n_atoms = tb.atom_pos.shape[0]
        print('atom_pos: %s %s' % (tb.atom_pos.dtype, list(tb.atom_pos.shape)))
        print('atom_names: %s' % tb.atom_names)
        print('atom_counts: %s' % tb.atom_counts)
        print('atom_spec: %s' % tb.atom_spec)
    if 'orb_pos' in data.files:
        tb.orb_pos = data['orb_pos']
        tb.orb_lmsr = data['orb_lmsr']
        tb.n_orbs = tb.orb_pos.shape[0]
        print('orb_pos: %s %s' % (tb.orb_pos.dtype, list(tb.orb_pos.shape)))
        print('orb_lms: %s %s' % (tb.orb_lmsr.dtype, list(tb.orb_lmsr.shape)))
        if 'orb_laxis' in data.files:
            tb.orb_is_laxis = True
            tb.orb_laxis = data['orb_laxis']
            print('orb_laxis: %s %s' % (tb.orb_laxis.dtype, list(tb.orb_laxis.shape)))
        else:
            tb.orb_is_laxis = False
            tb.orb_laxis = np.ones((tb.num_wann, 3, 3), np.float64)
            print('orb_is_laxis = False')
    tb.post_init()
    print('time used: %24.2f <-- get_tbsystem_by_npz_file' % (datetime.now() - start).total_seconds())
    return tb