import numpy as np
import os
import wanntb

kpath = np.array([[0.00, 0.00, -0.50], [0.00, 0.00, 0.50]])
# kpath = np.array([[-0.50, 0.00, 0.00], [0.50, 0.00, 0.00]])

# MBT-sl3
# path = 'MBT-sl3-soc-af001'

# AgRuO
path = 'AgRuO-soc'
ef = 5.3588

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


# AgRuO
output = tb.get_morb_berry_kpath(ef, kpath, nkpts_path=200, direction=3, eta=1e-6)
np.savetxt(path + '-morb-zgz.txt', output, fmt='%16.8e')
output = tb.get_berrycurv_kpath(ef, kpath, nkpts_path=200, eta=1e-6)
np.savetxt(path + '-omega-zgz.txt', output, fmt='%16.8e')
# output = tb.get_morb_berry_kpath(ef, kpath, nkpts_path=200, direction=3, eta=1e-6, lnew=True)
# np.savetxt(path + '-morb-zgz-2.txt', output, fmt='%16.8e')
# output = tb.get_berrycurv_kpath(ef, kpath, nkpts_path=200, eta=1e-6, lnew=True)
# np.savetxt(path + '-omega-zgz-2.txt', output, fmt='%16.8e')

# np.savetxt(path + '-occ-k%d.txt' % kmesh, output, fmt='%16.6f')