import numpy as np
import os
import wanntb

kpath = np.array([[0.00, 0.00, -0.50], [0.00, 0.00, 0.50]])
# kpath = np.array([[-0.50, 0.00, 0.00], [0.50, 0.00, 0.00]])

# MBT-sl3
# path = 'MBT-sl3-soc-af001'

# AgRuO
# path = 'AgRuO-soc'
# ef = 5.3588

# bilayer MnSe
path = 'MnSe-soc-af100'
ef = -2.2955

# tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')
npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')
# tb = wanntb.TBSystem(tb_file=tbfile)
tb = wanntb.TBSystem(npz_file=npzfile)

# MBT-sl3
# kmesh = 384
# output = tb.get_ahc_kmesh_fermi((kmesh,kmesh,1), 2.0200, 2.4200, 200, eta=1e-3)
# np.savetxt(path + '-ahc-k%d.txt' % kmesh, output, fmt='%16.6f')
# output = tb.get_ahc_kmesh_fermi((kmesh,kmesh,1), 2.0200, 2.4200, 200, eta=1e-3, lnew=True)
# np.savetxt(path + '-ahc-k%d-2.txt' % kmesh, output, fmt='%16.6f')

# bilayer MnSe
kmesh = 192
# subwf = np.array([1,2,3,4,5,11,12,13,17,18,19,20,21,27,28,29], dtype=int)
subwf = np.array([6,7,8,9,10,14,15,16,22,23,24,25,26,30,31,32], dtype=int)
output = tb.get_shc_kmesh_fermi((kmesh, kmesh, 1), -2.600, -2.100, 500,
                                alpha_beta=2, gamma=2, eta=1e-3, subwf=subwf)
np.savetxt(path + '-shc-k%d-2.txt' % kmesh, output, fmt='%16.6f')

# AgRuO
# output = tb.get_morb_berry_kpath(ef, kpath, nkpts_path=200, direction=3, eta=1e-6)
# np.savetxt(path + '-morb-zgz.txt', output, fmt='%16.8e')
# output = tb.get_berrycurv_kpath(ef, kpath, nkpts_path=200, eta=1e-6)
# np.savetxt(path + '-omega-zgz.txt', output, fmt='%16.8e')
# output = tb.get_morb_berry_kpath(ef, kpath, nkpts_path=200, direction=3, eta=1e-6, lnew=True)
# np.savetxt(path + '-morb-zgz-2.txt', output, fmt='%16.8e')
# output = tb.get_berrycurv_kpath(ef, kpath, nkpts_path=200, eta=1e-6, lnew=True)
# np.savetxt(path + '-omega-zgz-2.txt', output, fmt='%16.8e')

# np.savetxt(path + '-occ-k%d.txt' % kmesh, output, fmt='%16.6f')