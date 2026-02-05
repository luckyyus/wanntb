import numpy as np
import os
import wanntb

# MBT-sl3
path = 'MBT-sl3-soc-af001'

npzfile = os.path.join('tbdata', path + '-tb.npz')

tb = wanntb.get_tbsystem_by_npz_file(npz_file=npzfile)

symm = wanntb.symmetrize.Symmetrizer(tb, magmom_str='5 -5 5 18*0', is_soc=True)

ham_out, r_mat_out, ss_out, r_vec = symm.symmetrize(np.array([1, 1, 1], dtype=np.bool))
tb_new = wanntb.get_tbsystem_by_new_ham(tb, ham_out, r_mat_out, r_vec, ss_R_new=ss_out)