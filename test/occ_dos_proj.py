import numpy as np
import os
import wanntb

# MBT-sl2
path = 'MBT-sl2-soc-af001'

# MBT-sl3
# path = 'MBT-sl3-soc-af001'

# AgRuO
# path = 'AgRuO-soc'
# ef = 5.3588

# path = 'FeGe-u05-soc'

# tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')
npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')
# tb = wanntb.TBSystem(tb_file=tbfile)
tb = wanntb.TBSystem(npz_file=npzfile)
print(tb.get_onsite_energy())

# kmesh = 16
# occ, dos = tb.get_occ_dos_kmesh_fermi((kmesh, kmesh, kmesh), 3.0, 8.0, 1000, eta=1e-2)

kmesh = 384
ef_range = (1.000, 2.000, 1000)
occ, dos = tb.get_occ_dos_fermi((kmesh, kmesh, 1), ef_range, eta=1e-3, lproj=False)

np.savetxt(path + '-occ-k%d.txt' % kmesh, occ, fmt='%12.6f')
np.savetxt(path + '-dos-k%d.txt' % kmesh, dos, fmt='%12.6f')
