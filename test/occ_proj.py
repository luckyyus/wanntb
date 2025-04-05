import numpy as np
import os
import wanntb

path = 'MnTe-u4-soc-af1x'

# tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')
npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')
# tb = wanntb.TBSystem(tb_file=tbfile)
tb = wanntb.TBSystem(npz_file=npzfile)
kmesh = 18
out = tb.get_occ_kmesh_fermi((kmesh, kmesh, kmesh // 9 * 5), 5.0, 6.5, 1500, eta=1e-2, lproj=True)

np.savetxt(path + '-ab-k%d.txt' % (kmesh), out, fmt='%12.6f')