from typing import Tuple, List
from datetime import datetime
import numpy as np
from numpy.typing import NDArray
from numba import njit, prange

from .._system import TBSystem
from ..constant import DEFAULT_POSITION_TOLERANCE, DEFAULT_HAM_TOLERANCE, DEFAULT_SYMM_TOLERANCE, EPS5
from ..utility import find_R_vec, hermiization_R
from .operations import get_symmetry, SymmetryOperators
from ._rotate import (rotate_spinor, rotation_to_axis_angle, rotate_real_Ylm,
                      rotation_in_cart, get_all_L_rotation_matrix, combine_rotation_with_local_axis)


class Symmetrizer:

    def __init__(self, system: TBSystem, magmom_str: str|None = None,
                 symprec: float = DEFAULT_SYMM_TOLERANCE, is_soc: bool = True, dim: int=3):
        self._system = system
        self._operations: SymmetryOperators = get_symmetry(self._system.real_lattice,
                                                           self._system.atom_pos, self._system.atom_spec,
                                                           magmom_str=magmom_str,
                                                           tol=symprec)
        self._operations.print_symmetry()
        self.is_soc = is_soc
        self.dim = dim
        # process orbital sites data (positions, indices and the length)
        self.orb_site_pos, _orb_site_indices = group_orbitals_by_site(self._system.orb_pos)
        print('orb_site_pos: %s %s' % (self.orb_site_pos.dtype, list(self.orb_site_pos.shape)))
        self.n_orb_sites = self.orb_site_pos.shape[0]
        self.max_orbs = max(len(idx_arr) for idx_arr in _orb_site_indices)
        print('max_orbs: %d' % self.max_orbs)
        self.orb_site_indices = np.full((self.n_orb_sites, self.max_orbs), -1, dtype=int)
        self.orb_site_lens = np.zeros(self.n_orb_sites, dtype=int)
        for i in range(self.n_orb_sites):
            idx_arr = _orb_site_indices[i]
            n_idx = len(idx_arr)
            self.orb_site_indices[i, :n_idx] = idx_arr
            self.orb_site_lens[i] = n_idx
        print('orb_site_indices:')
        print(self.orb_site_indices)
        print('orb_site_lens: %s' % str(self.orb_site_lens))

        self.R_vec_pool = None
        self.n_Rpts_pool = 0
        self.site_maps = None
        self.spin_flip_map = spin_flip_mapping(self._system.orb_pos, self._system.orb_lmsr)
        # print('spin_flip_map:')
        # print(self.spin_flip_map)
        self.is_expand = False
        self.u_matrices_list = []

    def _update_site_mapping(self):
        n_op = len(self._operations)
        self.site_maps = np.zeros((n_op, self.n_orb_sites, 4), dtype=np.int_)
        for i in range(len(self._operations)):
            rotation, translation, time_reversal = self._operations[i]
            self.site_maps[i] = site_mapping(self._system.real_lattice, self.orb_site_pos,
                                             rotation, translation)
            print('site map %d' % i)
            print(self.site_maps[i])
        return

    def _update_R_vec_pool(self, is_expand: bool):
        self.is_expand = is_expand
        if is_expand:
            n_sites = self.orb_site_pos.shape[0]
            if self.site_maps is None:
                self._update_site_mapping()
            Rvec_set = set()
            for ir in range(self._system.n_Rpts):
                Rvec_set.add(tuple(self._system.R_vec[ir]))

            for isym in range(len(self._operations)):
                rotation, translation, time_reversal = self._operations[isym]
                for ir in range(self._system.n_Rpts):
                    rv = np.astype(self._system.R_vec[ir], np.float64)
                    rv_rot = rotation @ rv
                    rv_rounded = np.round(rv_rot)
                    if np.all(np.abs(rv_rot - rv_rounded) < DEFAULT_POSITION_TOLERANCE):
                        Rvec_set.add(tuple(rv_rounded.astype(np.int16)))
                        for a in range(n_sites):
                            a_tgt, v_a = self.site_maps[isym, a, 0], self.site_maps[isym, a, 1:]
                            if a_tgt < 0: continue
                            for b in range(n_sites):
                                b_tgt, v_b = self.site_maps[isym, b, 0], self.site_maps[isym, b, 1:]
                                if b_tgt < 0: continue
                                R_eff = rv_rounded + v_b - v_a
                                Rvec_set.add(tuple(np.round(R_eff).astype(np.int16)))
            rvec_list = list(Rvec_set)
            for rv in rvec_list:
                Rvec_set.add(tuple(np.round(-np.array(rv)).astype(np.int16)))

            self.R_vec_pool = np.array(sorted(Rvec_set), dtype=np.int16)
        else:
            self.R_vec_pool = self._system.R_vec.copy()
        self.n_Rpts_pool = self.R_vec_pool.shape[0]

    def symmetrize(self, tasks: str,
                   disable_list: List[int] | None = None,
                   enable_list: List[int] | None = None,
                   is_expand: bool = False,
                   is_global_tr: bool = False,
                   tol: float = DEFAULT_HAM_TOLERANCE):
        """Symmetrize Hamiltonian and/or other TB operators.
            tasks: string combining: 'h' - ham; 'r' or 'a' - r_mat; 's' - s_mat
        """
        start = datetime.now()
        print('---------- start symmetrize ----------')
        if disable_list is not None: self._operations.set_disabled(disable_list)
        if enable_list is not None: self._operations.set_enabled(enable_list)
        n_enabled = self._operations.n_enabled()
        print('number of enabled operators: %d' % n_enabled)
        _tasks = tasks.lower()
        l_ham = True if 'h' in _tasks else False
        l_rmat = True if ('a' in _tasks or 'r' in _tasks) else False
        l_ss = True if 's' in _tasks else False


        n_orbs = self._system.num_wann
        if self.site_maps is None:
            self._update_site_mapping()
        self._update_R_vec_pool(is_expand)
        print('R_vec_pool:', self.R_vec_pool.dtype, self.R_vec_pool.shape)
        print('time used: %24.2f <-- update R_vec_pool' % (datetime.now() - start).total_seconds())

        self._update_u_matrices()
        print('time used: %24.2f <-- update u_matrices' % (datetime.now() - start).total_seconds())

        args_expand = [self._system.R_vec, n_orbs, self.R_vec_pool, self.n_Rpts_pool]
        args_expand0 = [self._system.ham_R] + args_expand
        args_expand1 = [self._system.r_mat_R] + args_expand
        args_expand2 = [self._system.ss_R] + args_expand

        ham_orig = _get_oo_with_expand_R_vec(*args_expand0) if is_expand else self._system.ham_R
        r_mat_orig = _get_oo3_with_expand_R_vec(*args_expand1) if is_expand else self._system.r_mat_R
        ss_orig = None
        if self._system.ss_R is not None:
            ss_orig = _get_oo3_with_expand_R_vec(*args_expand2) if is_expand else self._system.ss_R

        if l_ham:
            hams = np.zeros((n_enabled, self.n_Rpts_pool, n_orbs, n_orbs), dtype=np.complex128)
        if l_rmat:
            r_mats = np.zeros((n_enabled, self.n_Rpts_pool, 3, n_orbs, n_orbs), dtype=np.complex128)
        if l_ss:
            assert self._system.ss_R is not None, 'ss_R data is missing but symmetrizing spin matrices is required.'
            s_mats = np.zeros((n_enabled, self.n_Rpts_pool, 3, n_orbs, n_orbs), dtype=np.complex128)


        idx_enable = 0
        for isym in range(len(self._operations)):
            if not self._operations.is_enabled[isym]: continue
            print('operator %d is enabled.' % isym)
            rotation, translation, time_reversal = self._operations[isym]
            u_matrices = self.u_matrices_list[isym]
            site_map = self.site_maps[isym]

            is_soc_tr = True if self.is_soc and time_reversal == 1 else False
            print('is_soc_tr: %s' % str(is_soc_tr))
            is_tr_only = (not self.is_soc and time_reversal == 1)
            print('is_tr_only: %s' % str(is_tr_only))
            args = [self._system.num_wann,
                    self.R_vec_pool, self.n_Rpts_pool,
                    rotation, site_map, u_matrices,
                    self.orb_site_indices, self.orb_site_lens,
                    self.spin_flip_map,
                    is_soc_tr, is_tr_only]
            if l_ham:
                args0 = [ham_orig] + args
                hams[idx_enable] = _rotate_site_par(*args0)
                print('time used: %24.2f <-- rotate Hamiltonian for symmetric operator No. %d finished' %
                      ((datetime.now() - start).total_seconds(), isym))
            if l_rmat:
                args1 = [r_mat_orig] + args
                r_mats[idx_enable] = _rotate3_site_par(*args1)
                print('time used: %24.2f <-- rotate r_matrices for symmetric operator No. %d finished' %
                      ((datetime.now() - start).total_seconds(), isym))
            if l_ss:
                args2 = [ss_orig] + args
                s_mats[idx_enable] = _rotate3_site_par(*args2)
                print('time used: %24.2f <-- rotate spin matrices for symmetric operator No. %d finished' %
                      ((datetime.now() - start).total_seconds(), isym))
            idx_enable += 1

        if l_ham:
            # Average
            ham_out = np.sum(hams, axis=0) / n_enabled

            # tolerance check
            max_diff, max_rv = _check_tolerance_R_par(ham_orig, ham_out, self.R_vec_pool, self.n_Rpts_pool)
            if max_diff > tol:
                print(f"WARNING: Symmetrization difference ({max_diff:.6f}) for ham_R "
                      f"exceeds tolerance ({tol:.6f}) at {max_rv}\n")
            # global time reversal
            if is_global_tr:
                ham_out = _global_tr_symmetry_R_par(ham_out, n_orbs, self.R_vec_pool, self.n_Rpts_pool,
                                                    self.spin_flip_map, self.is_soc)

            # hermitian
            ham_out = hermiization_R(ham_out, self.R_vec_pool)
        else:
            ham_out = ham_orig

        if l_rmat:
            # Average
            r_mat_out = np.sum(r_mats, axis=0) / n_enabled

            # tolerance check
            max_diff, max_rv = _check_tolerance_R_par(r_mat_orig, r_mat_out, self.R_vec_pool, self.n_Rpts_pool)
            if max_diff > tol:
                print(f"WARNING: Symmetrization difference ({max_diff:.6f}) for r_mat_R "
                      f"exceeds tolerance ({tol:.6f}) at {max_rv}\n")
            # hermitian
            r_mat_out = hermiization_R(r_mat_out, self.R_vec_pool)
        else:
            r_mat_out = r_mat_orig

        if l_ss:
            # Average
            ss_out = np.sum(s_mats, axis=0) / n_enabled

            # tolerance check
            max_diff, max_rv = _check_tolerance_R_par(ss_orig, ss_out, self.R_vec_pool, self.n_Rpts_pool)
            if max_diff > tol:
                print(f"WARNING: Symmetrization difference ({max_diff:.6f}) for ss_R "
                      f"exceeds tolerance ({tol:.6f}) at {max_rv}\n")
            # hermitian
            ss_out = hermiization_R(ss_out, self.R_vec_pool)
        else:
            ss_out = ss_orig

        print('time used: %24.2f <-- symmetrize' % (datetime.now() - start).total_seconds())
        return ham_out, r_mat_out, ss_out, self.R_vec_pool

    def _update_u_matrices(self):

        for isym in range(len(self._operations)):
            # print('operator %d: ' % isym)
            rotation, translation, time_reversal = self._operations[isym]

            args = [self._system.real_lattice,
                    self.orb_site_indices,
                    self.orb_site_lens, self.site_maps[isym],
                    self.n_orb_sites, self.max_orbs,
                    self._system.orb_lmsr, self._system.orb_laxis,
                    rotation, translation,
                    self.is_soc, self._system.orb_is_laxis]
            u_matrices_arr = _u_matrices_site_par(*args)

            self.u_matrices_list.append(u_matrices_arr)
        # print(self.u_matrices_list[2][0])
        # np.savetxt('u_12.txt', self.u_matrices_list[12][0], fmt='%8.4f')

@njit(parallel=True, cache=True, nogil=True)
def _check_tolerance_R_par(oo_orig, oo_symm, R_vec_pool, n_rpts):
    """Detailed tolerance check."""

    diff_max_arr = np.zeros(n_rpts, dtype=np.float64)

    for ir in prange(n_rpts):
        diff_mat = np.abs(oo_orig[ir] - oo_symm[ir])
        diff_max_arr[ir] = np.max(diff_mat)
    max_diff = np.max(diff_max_arr)
    irmax = np.argmax(diff_max_arr)
    rv = R_vec_pool[irmax]
    return max_diff, rv


@njit(parallel=True, cache=True, nogil=True)
def _global_tr_symmetry_R_par(oo, num_wann, R_vec, n_rpts, spin_flip_map, is_soc: bool):
    """Apply global Time Reversal symmetry (Robust Mapping) for Hamiltonian only."""

    _oo = np.copy(oo)
    if is_soc:
        for ir in prange(n_rpts):
            rv_neg = -R_vec[ir]
            ir_neg = find_R_vec(rv_neg, R_vec)
            if ir_neg >= 0:
                # H_neg = H(-R)*
                # H_tr = K H_neg K^dagger
                for i in range(num_wann):
                    i_flip = spin_flip_map[i, 0]
                    if i_flip < 0: continue
                    sign_i = spin_flip_map[i, 1]
                    for j in range(num_wann):
                        j_flip = spin_flip_map[j, 0]
                        if j_flip < 0: continue
                        sign_j = spin_flip_map[j, 1]
                        _val = sign_i * sign_j * np.conj(oo[ir_neg, i_flip, j_flip])
                        _oo[ir, i, j] = (oo[ir, i, j] + _val) / 2.0
    else:
        for ir in prange(n_rpts):
            rv_neg = -R_vec[ir]
            ir_neg = find_R_vec(rv_neg, R_vec)
            if ir_neg >= 0:
                val_tr = np.conj(oo[ir_neg, :, :])
                _oo[ir, :, :] = (oo[ir, :, :] + val_tr) / 2.0
    return _oo


@njit(parallel=True, cache=True, nogil=True)
def _u_matrices_site_par(real_lattice: NDArray,
                         site_indices: NDArray, site_lens: NDArray, site_map: NDArray,
                         nsites: int, max_orbs: int,
                         orb_lmsr: NDArray, orb_laxis: NDArray,
                         rotation: NDArray, translation: NDArray,
                         is_soc: bool, is_local_axis: bool) -> NDArray:
    if is_local_axis:
        rot_cart = rotation_in_cart(rotation, real_lattice)
    axis, angle, is_inv = rotation_to_axis_angle(rotation, real_lattice)
    # print(axis, angle, is_inv)
    L_rot = get_all_L_rotation_matrix(axis, angle, is_inv)
    s_rot = rotate_spinor(axis, angle) if is_soc else np.eye(2, dtype=np.complex128)

    u_matrices_arr = np.zeros((nsites, max_orbs, max_orbs), dtype=np.complex128)
    for i in prange(nsites):
        idx_src = site_indices[i, :site_lens[i]]
        tgt_idx = site_map[i, 0]
        idx_tgt = site_indices[tgt_idx, :site_lens[i]]
        # print('%d, idx_src[%d]: %s, idx_tgt[%d]: %s' % (i, i, str(idx_src), tgt_idx, str(idx_tgt)))
        if tgt_idx < 0:
            continue
        if is_local_axis:
            u_site = _u_matrix_one_site_laxis(idx_src, idx_tgt, orb_lmsr, orb_laxis, rot_cart, is_soc)
        else:
            u_site = _u_matrix_one_site(idx_src, idx_tgt, orb_lmsr, L_rot, s_rot, is_soc)
        if u_site is not None:
            r, c = u_site.shape
            u_matrices_arr[i, :r, :c] = u_site
    return u_matrices_arr


# @njit(parallel=True, cache=True, nogil=True)
def _rotate_site_par(oo_R, num_wann: int, R_vec_pool, n_Rpts_pool: int,
                     rotation, site_map, u_matrices, orb_site_indices, orb_site_lens, spin_flip_map,
                     is_soc_tr, is_tr_only):
    """
    rotation lattice space Hamiltonian.
    """

    nsites = site_map.shape[0]

    # Initialize output Hamiltonian
    oo_out = np.zeros((n_Rpts_pool, num_wann, num_wann), dtype=np.complex128)

    # Pre-calculate R_rotated = R * S^T
    # This is done inside site loops but can be optimized

    # Parallelize over sites to avoid competition on oo_out for single operation
    # Since rotation is a bijection of sites, a_tgt is unique for each a.
    # However, different (a, b) pairs might map to the same (a_tgt, b_tgt)
    # if the symmetry operation has a translation. But they will map to different R.
    # So we should be safe with prange if we are careful.

    for a in prange(nsites):
        a_tgt = site_map[a, 0]
        if a_tgt < 0: continue
        v_a = site_map[a, 1:]
        n_a = orb_site_lens[a]  # = orb_site_lens[a_tgt]
        idx_a = orb_site_indices[a, :n_a]
        idx_a_tgt = orb_site_indices[a_tgt, :n_a]

        U_a = np.ascontiguousarray(u_matrices[a, :n_a, :n_a])

        for b in range(nsites):
            b_tgt = site_map[b, 0]
            if b_tgt < 0: continue
            v_b = site_map[b, 1:]
            n_b = orb_site_lens[b]  # = orb_site_lens[b_tgt]
            idx_b = orb_site_indices[b, :n_b]
            idx_b_tgt = orb_site_indices[b_tgt, :n_b]

            U_b_H = np.ascontiguousarray(np.conj(u_matrices[b, :n_b, :n_b].T))
            shift = (v_b - v_a).astype(np.float64)
            oo_block = np.zeros((n_a, n_b), dtype=np.complex128)
            oo_orig = np.zeros((n_a, n_b), dtype=np.complex128)
            for ir in range(n_Rpts_pool):
                # R_eff = S @ R + v_b - v_a
                rv_src = R_vec_pool[ir].astype(np.float64)
                rv_eff = np.rint(rotation @ rv_src + shift)
                print(rv_src, rv_eff)
                # Binary search for R index
                ir_tgt = find_R_vec(rv_eff, R_vec_pool)
                # print(ir, ir_tgt)
                if ir_tgt < 0: continue #should not happen

                # Extract block H_ab
                for i_a in range(n_a):
                    for j_b in range(n_b):
                        oo_block[i_a, j_b] = oo_R[ir, idx_a[i_a], idx_b[j_b]]
                        oo_orig[i_a, j_b] = oo_R[ir_tgt, idx_a_tgt[i_a], idx_b_tgt[j_b]]

                # Handle Magnetic Symmetry
                if is_soc_tr:
                    oo_block = _apply_soc_tr(oo_block, idx_a, idx_b, spin_flip_map)
                elif is_tr_only:
                    oo_block = np.conj(oo_block)

                # Use contiguous arrays for better performance with '@'
                oo_rot = U_a @ np.ascontiguousarray(oo_block) @ U_b_H

                # Accumulate
                # Atomic add is not directly available in prange for complex
                # But since each (ir, a, b) contributes to a unique (a_tgt, b_tgt, ir_tgt)
                # for a FIXED symmetry operation, we don't have race conditions here.
                for i_a in range(n_a):
                    for j_b in range(n_b):
                        _p = np.abs(oo_orig[i_a, j_b] - oo_rot[i_a, j_b])
                        _n = np.abs(oo_orig[i_a, j_b] + oo_rot[i_a, j_b])
                        oo_out[ir_tgt, idx_a_tgt[i_a], idx_b_tgt[j_b]] = oo_rot[i_a, j_b]
                                                                          # if _p < _n or _n < DEFAULT_HAM_TOLERANCE
                                                                         # else -oo_rot[i_a, j_b])
    return oo_out

@njit(parallel=True, cache=True, nogil=True)
def _rotate3_site_par(oo_R, num_wann: int, R_vec_pool, n_Rpts_pool: int,
                     rotation, site_map, u_matrices, orb_site_indices, orb_site_lens, spin_flip_map,
                     is_soc_tr, is_tr_only):
    """
    Numba optimized rotation kernel.
    """
    nsites = site_map.shape[0]

    # Initialize output Hamiltonian
    oo_out = np.zeros((n_Rpts_pool, 3, num_wann, num_wann), dtype=np.complex128)

    for a in prange(nsites):
        a_tgt = site_map[a, 0]
        if a_tgt < 0: continue
        v_a = site_map[a, 1:]

        idx_a = orb_site_indices[a, :orb_site_lens[a]]
        idx_a_tgt = orb_site_indices[a_tgt, :orb_site_lens[a_tgt]]
        n_a = len(idx_a) # = len(idx_a_tgt)
        U_a = np.ascontiguousarray(u_matrices[a, :orb_site_lens[a_tgt], :orb_site_lens[a]])

        for b in range(nsites):
            b_tgt = site_map[b, 0]
            if b_tgt < 0: continue
            v_b = site_map[b, 1:]

            idx_b = orb_site_indices[b, :orb_site_lens[b]]
            idx_b_tgt = orb_site_indices[b_tgt, :orb_site_lens[b_tgt]]
            n_b = len(idx_b) # = len(idx_b_tgt)
            U_b_H = np.ascontiguousarray(np.conj(u_matrices[b, :orb_site_lens[b_tgt], :orb_site_lens[b]].T))

            shift = v_b - v_a

            for ir in range(n_Rpts_pool):
                # R_eff = S @ R + v_b - v_a
                rv_src = R_vec_pool[ir].astype(np.float64)
                rv_eff = np.rint(rotation @ rv_src + shift)

                # Binary search for R index
                ir_tgt = find_R_vec(rv_eff, R_vec_pool)
                if ir_tgt < 0: continue

                for i in range(3):
                    oo_block = np.zeros((n_a, n_b), dtype=np.complex128)
                    # oo_orig = np.zeros((n_a, n_b), dtype=np.complex128)

                    for i_a in range(n_a):
                        for j_b in range(n_b):
                            oo_block[i_a, j_b] = oo_R[ir, i, idx_a[i_a], idx_b[j_b]]
                            # oo_orig[i_a, j_b] = oo_R[ir_tgt, i, idx_a_tgt[i_a], idx_b_tgt[j_b]]

                    if is_soc_tr:
                        oo_block = _apply_soc_tr(oo_block, idx_a, idx_b, spin_flip_map)
                    elif is_tr_only:
                        oo_block = np.conj(oo_block)
                    oo_rot = U_a @ np.ascontiguousarray(oo_block) @ U_b_H

                    for i_a in range(n_a):
                        for j_b in range(n_b):
                            # _p = np.abs(oo_orig[i_a, j_b] - oo_rot[i_a, j_b])
                            # _n = np.abs(oo_orig[i_a, j_b] + oo_rot[i_a, j_b])
                            oo_out[ir_tgt, i, idx_a_tgt[i_a], idx_b_tgt[j_b]] = oo_rot[i_a, j_b]
                                                                                 # if _p < _n
                                                                                # else -oo_rot[i_a, j_b])
    return oo_out


@njit(cache=True, nogil=True)
def _get_oo_with_expand_R_vec(oo_R: NDArray[np.complex128],
                              R_vec: NDArray[np.int16], num_wann: int,
                              R_vec_pool: NDArray[np.int16], n_Rpts_pool: int) -> NDArray[np.complex128]:
    nrpt_in = R_vec.shape[0]
    oo_out = np.zeros((n_Rpts_pool, num_wann, num_wann), dtype=np.complex128)

    for ir in range(nrpt_in):
        rv = R_vec[ir]
        ir_tgt = find_R_vec(rv, R_vec_pool)
        if ir_tgt < 0: continue
        oo_out[ir_tgt] = oo_R[ir]

    return oo_out

@njit(cache=True, nogil=True)
def _get_oo3_with_expand_R_vec(oo_R: NDArray[np.complex128],
                              R_vec: NDArray[np.int16], num_wann: int,
                              R_vec_pool: NDArray[np.int16], n_Rpts_pool: int) -> NDArray[np.complex128]:

    nrpt_in = R_vec.shape[0]
    oo_out = np.zeros((n_Rpts_pool, 3, num_wann, num_wann), dtype=np.complex128)

    for ir in range(nrpt_in):
        rv = R_vec[ir]
        ir_tgt = find_R_vec(rv, R_vec_pool)
        if ir_tgt < 0: continue
        oo_out[ir_tgt] = oo_R[ir]

    return oo_out


@njit(cache=True, nogil=True)
def _apply_soc_tr(h_block: NDArray, idx_a: NDArray, idx_b: NDArray, spin_flip_map: NDArray) -> NDArray:
    """Apply SOC Time Reversal (K = i*sigma_y) to a block."""
    # H_TR = K H* K^dagger
    # K = [[0, -1], [1, 0]] per spin block
    h_tr = np.conj(h_block)
    h_out = np.zeros_like(h_tr)

    n_a = idx_a.shape[0]
    n_b = idx_b.shape[0]

    for i_loc in range(n_a):
        i_glob = idx_a[i_loc]
        i_flip_glob, sign_i = spin_flip_map[i_glob, 0], spin_flip_map[i_glob, 1]

        # Find local index of i_flip_glob in idx_a
        i_flip_local = -1
        for k in range(n_a):
            if idx_a[k] == i_flip_glob:
                i_flip_local = k
                break

        if i_flip_local == -1: continue  # Should not happen

        for j_loc in range(n_b):
            j_glob = idx_b[j_loc]
            j_flip_glob, sign_j = spin_flip_map[j_glob, 0], spin_flip_map[j_glob, 1]

            j_flip_local = -1
            for k in range(n_b):
                if idx_b[k] == j_flip_glob:
                    j_flip_local = k
                    break

            if j_flip_local == -1: continue  # Should not happen

            h_out[i_loc, j_loc] = sign_i * sign_j * h_tr[i_flip_local, j_flip_local]

    return h_out


def group_orbitals_by_site(orb_pos: NDArray) -> Tuple[NDArray, List]:
    site_pos = []
    site_indices = []
    norb = orb_pos.shape[0]
    for i in range(norb):
        found = False
        for j, pos in enumerate(site_pos):
            if np.allclose(orb_pos[i], pos, atol=DEFAULT_POSITION_TOLERANCE):
                site_indices[j].append(i)
                found = True
                break
        if not found:
            site_pos.append(orb_pos[i])
            site_indices.append([i])
    site_pos = np.array(site_pos)
    return site_pos, site_indices


@njit(cache=True, nogil=True)
def _u_matrix_one_site(indices_src, indices_tgt, orb_lmsr, L_rot, s_rot, is_soc):
    n_src = indices_src.shape[0]
    n_tgt = indices_tgt.shape[0]
    U = np.zeros((n_tgt, n_src), dtype=np.complex128)

    # src_map = {(orb_lmsr[i, 0],orb_lmsr[i, 1], orb_lmsr[i, 2]): i_loc
    #            for i_loc, i in enumerate(indices_src)}
    # tgt_map = {(orb_lmsr[i, 0],orb_lmsr[i, 1], orb_lmsr[i, 2]): i_loc
    #            for i_loc, i in enumerate(indices_tgt)}
    for i_loc in range(n_src):
        i = indices_src[i_loc]
        l, mr, ms = orb_lmsr[i, 0], orb_lmsr[i, 1], orb_lmsr[i, 2]
        for j_loc in range(n_tgt):
            j = indices_tgt[j_loc]
            if orb_lmsr[j, 0] != l: continue
            mr_prime, ms_prime = orb_lmsr[j, 1], orb_lmsr[j, 2]
            if is_soc:
                U[j_loc, i_loc] = L_rot[l, mr_prime, mr] * s_rot[ms_prime, ms]
            else:
                U[j_loc, i_loc] = L_rot[l, mr_prime, mr]

        # for mr_prime in range(0, 2*l + 1):
        #     if is_soc:
        #         for ms_prime in [0, 1]:
        #
        #             if (l, mr_prime, ms_prime) in tgt_map:
        #                 j_loc = tgt_map[(l, mr_prime, ms_prime)]
        #                 U[j_loc, i_loc] = D_l[l][mr_prime, mr] * s_rot[ms_prime, ms]
        #     else:
        #         if (l, mr_prime, ms) in tgt_map:
        #             j_loc = tgt_map[(l, mr_prime, ms)]
        #             U[j_loc, i_loc] = D_l[l][mr_prime, mr]
    return U


@njit(cache=True, nogil=True)
def _u_matrix_one_site_laxis(indices_src, indices_tgt, orb_lmsr, orb_laxis, rot_cart, is_soc):
    n_src = indices_src.shape[0]
    n_tgt = indices_tgt.shape[0]
    U = np.zeros((n_tgt, n_src), dtype=np.complex128)
    for i_local in range(n_src):
        i = indices_src[i_local]
        l, mr, ms = orb_lmsr[i, 0], orb_lmsr[i, 1], orb_lmsr[i, 2]
        for j_local in range(n_tgt):
            j = indices_tgt[j_local]
            if orb_lmsr[j, 0] != l: continue
            mr_prime, ms_prime = orb_lmsr[j, 1], orb_lmsr[j, 2]
            rot_combined = combine_rotation_with_local_axis(rot_cart, orb_laxis[i], orb_laxis[j])
            axis, angle, is_inv = rotation_to_axis_angle(rot_combined, np.eye(3))

            D = rotate_real_Ylm(l, axis, angle, is_inv)
            orb_factor = D[mr_prime, mr]

            if is_soc:
                s_rot_local = rotate_spinor(axis, angle)
                spin_factor = s_rot_local[ms_prime, ms]
                U[j_local, i_local] = orb_factor * spin_factor
            else:
                if ms_prime == ms:
                    U[j_local, i_local] = orb_factor
    return U


@njit(nogil=True)
def orbital_mapping(lattice: NDArray, orb_pos: NDArray, orb_lmsr: NDArray,
                    rotation: NDArray, translation: NDArray) -> NDArray:
    """Find how each orbital maps under symmetry operation.

    Args:
        orb_info: List of WannOrb objects.
        symm: Symmetry operator.
        lattice: Lattice vectors [3, 3].

    Returns:
        List of (target_orbital_index, rvec) for each input orbital.
        If an orbital doesn't map to another, returns (-1, [0,0,0]).
    """
    norb = orb_pos.shape[0]
    mapping = np.zeros((norb, 4), dtype=np.int_)

    for i in range(norb):
        tau_transformed = rotation @ orb_pos[i] + translation

        # Find matching orbital
        found = False
        for j in range(norb):
            if orb_lmsr[j, 0] != orb_lmsr[i, 0] or orb_lmsr[j, 3] != orb_lmsr[i, 3]:
                continue
            # For SOC, also check spin if needed (though spin can mix)

            diff = tau_transformed - orb_pos[j]
            rvec = np.round(diff)
            remainder = diff - rvec
            if np.linalg.norm(remainder @ lattice) < EPS5:
                mapping[i, 0] = j
                mapping[i, 1:] = np.array(rvec, dtype=np.int_)
                found = True
                break
        if not found:
            mapping[i, 0] = -1
            mapping[i, 1:] = np.zeros(3)
    return mapping


@njit(nogil=True)
def site_mapping(lattice: NDArray, site_positions: NDArray,
                 rotation: NDArray, translation: NDArray, dim=3) -> NDArray:
    nsites = site_positions.shape[0]
    mapping = np.zeros((nsites, 4), dtype=np.int_)
    # rot = -rotation if is_inv else rotation
    for i in range(nsites):
        tau_new = rotation @ site_positions[i] + translation
        found = False
        for j in range(nsites):
            diff = tau_new - site_positions[j]
            rvec = np.round(diff)
            remainder = (diff - rvec).astype(np.float64)
            if np.linalg.norm(remainder @ lattice) < EPS5:
                mapping[i, 0] = j
                mapping[i, 1:dim+1] = rvec[:dim]
                found = True
                break
        if not found: # should not happen
            mapping[i, 0] = -1
            mapping[i, 1:] = np.zeros(3)
    return mapping


@njit(nogil=True)
def spin_flip_mapping(orb_pos: NDArray, orb_lmsr: NDArray) -> NDArray:
    """Build a mapping for SOC Time Reversal (K = i*sigma_y).
    Returns:
        List [num_wann, 2] of (target_orbital_index, time_reversal) for each input orbital
        If an orbital doesn't map to another, returns (self_index, 1).
    """
    norb = orb_pos.shape[0]
    mapping = np.zeros((norb, 2), dtype=np.int_)
    for i in range(norb):
        # Look for the spin partner (same site, l, mr, but opposite ms)
        found = False
        for j in range(norb):
            if (np.allclose(orb_pos[i], orb_pos[j]) and orb_lmsr[i, 0] == orb_lmsr[j, 0] and
                orb_lmsr[i, 1] == orb_lmsr[j, 1] and orb_lmsr[i, 2] != orb_lmsr[j, 2]):
                # ms=0 (Up) -> sign -1 (from -1 in i*sigma_y)
                # ms=1 (Down) -> sign 1 (from 1 in i*sigma_y)
                sign = -1 if orb_lmsr[i, 2] == 0 else 1
                mapping[i, :] = j, sign
                found = True
                break
        if not found:
            # Should not happen in valid SOC hr.dat
            mapping[i, :] = i, 1
    return mapping
