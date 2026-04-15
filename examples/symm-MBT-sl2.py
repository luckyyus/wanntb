import numpy as np
import os
import wanntb

# MBT-sl2
path = 'MBT-sl2-soc-af001'

kpath = np.array([[0.50, 0.00, 0.00],
                  [0.00, 0.00, 0.00],
                  [1.0/3.0, 1.0/3.0, 0.00],
                  [0.50, 0.00, 0.00]])

npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')

tb = wanntb.get_tbsystem_by_npz_file(npz_file=npzfile)

tb.output_bands_kpath(kpath, nkpts_path=100, filename=path + '-bands-orig.txt')

symm = wanntb.symmetrize.Symmetrizer(tb, magmom_str='0 0 5 0 0 -5 36*0', is_soc=True)


ham_out, r_mat_out, ss_out, r_vec = symm.symmetrize('h',
                                                    # enable_list=[1],
                                                    is_expand=True)

tb_new = wanntb.get_tbsystem_by_new_ham(tb, ham_out, r_mat_out, r_vec, ss_R_new=ss_out)

tb_new.output_bands_kpath(kpath, nkpts_path=100, filename=path + '-bands-symm.txt')
