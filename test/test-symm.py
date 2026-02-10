import numpy as np
import os
import wanntb

# MBT-sl3
path = 'MBT-sl3-soc-af001'

kpath = np.array([[0.50, 0.00, 0.00],
                  [0.00, 0.00, 0.00],
                  [1.0/3.0, 1.0/3.0, 0.00],
                  [0.50, 0.00, 0.00]])

npzfile = os.path.join('tbdata', path + '-tb.npz')

tb = wanntb.get_tbsystem_by_npz_file(npz_file=npzfile)

tb.output_bands_kpath(kpath, nkpts_path=100, filename='mbt-sl3-bands-orig.txt')
# print(tb.get_onsite_energy())

symm = wanntb.symmetrize.Symmetrizer(tb, magmom_str='5 -5 5 18*0', is_soc=True)

ham_out, r_mat_out, ss_out, r_vec = symm.symmetrize(np.array([1, 0, 0], dtype=np.bool),
                                                    enable_list=[0, 1], is_expand=True)
tb_new = wanntb.get_tbsystem_by_new_ham(tb, ham_out, r_mat_out, r_vec, ss_R_new=ss_out)
tb_new.output_bands_kpath(kpath, nkpts_path=100, filename='mbt-sl3-bands-symm.txt')
# print(tb_new.get_onsite_energy())