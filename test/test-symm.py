import numpy as np
import os
import wanntb

# MBT-sl3
path = 'MBT-sl2-soc-af001'

# GaAs fcc
# path = 'GaAs-soc-001'
# fermi = 3.0713

kpath = np.array([[0.50, 0.00, 0.00],
                  [0.00, 0.00, 0.00],
                  [1.0/3.0, 1.0/3.0, 0.00],
                  [0.50, 0.00, 0.00]])

# kpath = np.array([[0.50, 0.50, 0.50],
#                   [0.00, 0.00, 0.00],
#                   [0.50, 0.50, 0.00],
#                   [0.50, 0.50, 0.50],
#                   [0.50, 0.25, 0.75]])

npzfile = os.path.join('tbdata', path + '-tb.npz')

tb = wanntb.get_tbsystem_by_npz_file(npz_file=npzfile)

tb.output_bands_kpath(kpath, nkpts_path=100, filename=path + '-bands-orig.txt')
# print(tb.atom_pos)
# print(tb.get_onsite_energy())
symm = wanntb.symmetrize.Symmetrizer(tb, magmom_str='0 0 5 0 0 -5 36*0', is_soc=True)
# print(symm.site_maps[1])
# symm = wanntb.symmetrize.Symmetrizer(tb, magmom_str='5 -5 5 18*0', is_soc=True)
# symm = wanntb.symmetrize.Symmetrizer(tb, magmom_str='0 0', is_soc=True)

ham_out, r_mat_out, ss_out, r_vec = symm.symmetrize('h',
                                                    # enable_list=[0, 1, 4, 5, 8, 9, 12, 13],
                                                    # enable_list=[2,3],
                                                    # enable_list=[16, 17],
                                                    enable_list=[1],
                                                    is_expand=True)
tb_new = wanntb.get_tbsystem_by_new_ham(tb, ham_out, r_mat_out, r_vec, ss_R_new=ss_out)

tb_new.output_bands_kpath(kpath, nkpts_path=100, filename=path + '-bands-symm.txt')

rv = np.array((0, 0, 0), dtype=np.int16)
orig = tb.get_ham_one_R(rv)#[:8, :8]
symm = tb_new.get_ham_one_R(rv) #[:8, :8]
np.savetxt('hr_orig_000.txt', orig, fmt='%8.4f')
np.savetxt('hr_symm_000.txt', symm, fmt='%8.4f')

np.savetxt('hr_diff_000.txt', symm - orig, fmt='%8.4f')