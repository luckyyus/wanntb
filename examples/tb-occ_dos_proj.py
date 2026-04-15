import numpy as np
import os
import wanntb


# MBT-sl2
path = 'MBT-sl2-soc-af001'

npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')

tb = wanntb.get_tbsystem_by_npz_file(npz_file=npzfile)
# print(tb.get_onsite_energy())

kmesh = 192
ef_range = (-1.000, 3.000, 1000)
occ, dos = tb.get_occ_dos_fermi((kmesh, kmesh, 1), ef_range, eta=1e-3, lproj=True)

np.savetxt(path + '-occ-k%d.txt' % kmesh, occ, fmt='%12.6f')
np.savetxt(path + '-dos-k%d.txt' % kmesh, dos, fmt='%12.6f')
