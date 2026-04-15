import numpy as np
import os
import wanntb


# MBT-sl2
path = 'MBT-sl2-soc-af001'
# ef = 1.442

npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')

tb = wanntb.get_tbsystem_by_npz_file(npz_file=npzfile)

# MBT
ks = 96
kmesh = (ks, ks, 1)

#sl2
ef_range = (1.300, 1.600, 300)

output = tb.axion_calc_fermi(kmesh, ef_range, eta=1e-3)
np.savetxt(path + '-axion-k%d.txt' % ks, output, fmt='%16.6f')

# the bottom layer
subwf0 = np.r_[0:5, 10:13, 16:19, 22:25, 31:34, 34:37, 43:46]
subwf = np.append(subwf0, subwf0+46)


output = tb.berry_calc_fermi('ahc+shc', kmesh, ef_range, eta=1e-3, xyz=2, subwf=subwf)
np.savetxt(path + '-berry-k%d.txt' % ks, output, fmt='%16.6f')

