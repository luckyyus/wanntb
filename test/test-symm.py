import numpy as np
import os
import wanntb

# MBT-sl2
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
# np.savetxt(path + '-wfc-frac.txt', tb.wann_centers_frac, fmt='%8.4f')
# print(tb.atom_pos)
# print(tb.get_onsite_energy())
symm = wanntb.symmetrize.Symmetrizer(tb, magmom_str='0 0 5 0 0 -5 36*0', is_soc=True)

# symm = wanntb.symmetrize.Symmetrizer(tb, magmom_str='5 -5 5 18*0', is_soc=True)
# symm = wanntb.symmetrize.Symmetrizer(tb, magmom_str='0 0', is_soc=True)
#
ham_out, r_mat_out, ss_out, r_vec = symm.symmetrize('h',
                                                    # enable_list=[0, 1, 4, 5, 8, 9, 12, 13],
                                                    # enable_list=[2,3],
                                                    # enable_list=[16, 17],
                                                    # enable_list=[1],
                                                    is_expand=True)
# print(symm.site_maps[1])
tb_new = wanntb.get_tbsystem_by_new_ham(tb, ham_out, r_mat_out, r_vec, ss_R_new=ss_out)
#
tb_new.output_bands_kpath(kpath, nkpts_path=100, filename=path + '-bands-symm.txt')
