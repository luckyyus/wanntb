import numpy as np
import os
import wanntb

path = 'FeGe-u05-soc'

# tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')
npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')
# tb = wanntb.TBSystem(tb_file=tbfile)
tb = wanntb.TBSystem(npz_file=npzfile)
kmesh = 32
occ, dos = tb.get_occ_dos_kmesh_fermi((kmesh, kmesh, kmesh), 6.0, 8.0, 1000, eta=1e-2, lproj=True)

np.savetxt(path + '-occ-k%d.txt' % (kmesh), occ, fmt='%12.6f')
np.savetxt(path + '-dos-k%d.txt' % (kmesh), dos, fmt='%12.6f')