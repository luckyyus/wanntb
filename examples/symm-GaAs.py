import numpy as np
import os
import wanntb as wtb
import wanntb.symmetrize as sym

# GaAs fcc
path = 'GaAs-soc-001'
# fermi = 3.0713


kpath = np.array([[0.50, 0.50, 0.50],
                  [0.00, 0.00, 0.00],
                  [0.50, 0.50, 0.00],
                  [0.50, 0.50, 0.50],
                  [0.50, 0.25, 0.75]])

npzfile = os.path.join('..','data', path + '-tb.npz')

tb = wtb.get_tbsystem_by_npz_file(npz_file=npzfile)

tb.output_bands_kpath(kpath, nkpts_path=100, filename=path + '-bands-orig.txt')

symm = sym.Symmetrizer(tb, magmom_str='0 0', is_soc=True)

ham_out, r_mat_out, ss_out, r_vec = symm.symmetrize('h',
                                                    # enable_list=[1],
                                                    is_expand=True)
# print(symm.site_maps[1])
tb_new = wtb.get_tbsystem_by_new_ham(tb, ham_out, r_mat_out, r_vec, ss_R_new=ss_out)
#
tb_new.output_bands_kpath(kpath, nkpts_path=100, filename=path + '-bands-symm.txt')
