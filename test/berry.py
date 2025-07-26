import numpy as np
import os
import wanntb

kpath = np.array([[0.00, 0.00, -0.50], [0.00, 0.00, 0.50]])
# kpath = np.array([[-0.50, 0.00, 0.00], [0.50, 0.00, 0.00]])

# MBT-sl2
# path = 'MBT-sl2-soc-af001'
# ef = 1.442
# MBT-sl3
path = 'MBT-sl3-soc-af001'
# ef = 2.2445
#MBT-sl6
# path = 'MBT-sl6-soc-af001'
# ef = 3.2791

# AgRuO
# path = 'AgRuO-soc'
# ef = 5.3588

# bilayer MnSe
# path = 'MnSe-soc-af100'
# ef = -2.2955

# tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')
npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')
# tb = wanntb.TBSystem(tb_file=tbfile)
tb = wanntb.TBSystem(npz_file=npzfile)

# MBT
ks = 384
kmesh = (ks, ks, 1)
#sl2
# ef_range = (1.300, 2.600, 300)
#sl3
ef_range = (2.100, 2.400, 300)
output = tb.berry_calc_fermi('ahc+shc', kmesh, ef_range, eta=1e-3, xyz=2)
np.savetxt(path + '-berry-k%d.txt' % ks, output, fmt='%16.6f')

# output = tb.get_ahc_kmesh_fermi((kmesh,kmesh,1), ef_range, eta=1e-3)
# np.savetxt(path + '-ahc-k%d.txt' % kmesh, output, fmt='%16.6f')

# subwf0 = np.array([1,2,3,4,5,31,32,33,55,56,57,67,68,69,97,98,99,103,104,105,133,134,135], dtype=int)
# subwf0 = np.r_[0:5, 30:33, 54:57, 66:69, 96:99, 102:105, 132:135]
# subwf = np.append(subwf0, subwf0+138)
# subwf -= 1
# print(subwf)
# output = tb.get_shc_fermi(kmesh, ef_range, eta=1e-3, direction=2)
# np.savetxt(path + '-shc-0-k%d.txt' % ks, output, fmt='%16.6f')


# bilayer MnSe
# kmesh = 192
# subwf = np.array([1,2,3,4,5,11,12,13,17,18,19,20,21,27,28,29], dtype=int)
# subwf = np.array([6,7,8,9,10,14,15,16,22,23,24,25,26,30,31,32], dtype=int)
# output = tb.get_shc_kmesh_fermi((kmesh, kmesh, 1), -2.600, -2.100, 500,
#                                 alpha_beta=2, gamma=2, eta=1e-3, subwf=subwf)
# np.savetxt(path + '-shc-k%d-2.txt' % kmesh, output, fmt='%16.6f')

# AgRuO
# output = tb.get_morb_berry_kpath(ef, kpath, nkpts_path=200, alpha_beta=2, eta=1e-6)
# np.savetxt(path + '-morb_berry-zgz.txt', output, fmt='%16.8e')
# output = tb.get_berrycurv_kpath(ef, kpath, nkpts_path=200, eta=1e-6, mode=1, q=1e-6)
# np.savetxt(path + '-omega-zgz-AD.txt', output, fmt='%16.8e')  # Ah, Abar_Dh, W
# output = tb.get_totmorb_kpath(ef,kpath, nkpts_path=200, alpha_beta=2, eta=1e-4, q=1e-8)
# np.savetxt(path + '-totmorb-zgz.txt', output, fmt='%16.8e')

# np.savetxt(path + '-occ-k%d.txt' % kmesh, output, fmt='%16.6f')