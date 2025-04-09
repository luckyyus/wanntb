import numpy as np
import os
import wanntb

kpath = np.array([[0.00, 0.00, -0.50], [0.00, 0.00, 0.50]])
# kpath = np.array([[-0.50, 0.00, 0.00], [0.50, 0.00, 0.00]])
# path = 'MBT-sl3-soc-af001'
path = 'AgRuO-soc'
ef = 5.3588
# tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')
npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')
# tb = wanntb.TBSystem(tb_file=tbfile)
tb = wanntb.TBSystem(npz_file=npzfile)
# kmesh = 32
# output = tb.get_occ_kmesh_fermi((kmesh,kmesh,kmesh//8*7), 5.3300, 5.3800, 500, eta=1e-4)


# kmesh = 384
# output = tb.get_occ_kmesh_fermi((kmesh,kmesh,1), 2.1000, 2.3000, 400, eta=5e-4)
# output = tb.get_ahc_kmesh_fermi((kmesh,kmesh,1), 2.0200, 2.4200, 200, eta=1e-4)
output = tb.get_berry_curv_kpath(ef, kpath, nkpts_path=200)
# AgRuO
# output = tb.get_morb_berry_kpath(ef, kpath, nkpts_path=200, direction=3, eta=1e-5)

# np.savetxt(path + '-morb-kz.txt', output, fmt='%16.8e')
np.savetxt(path + '-omega-kz.txt', output, fmt='%16.8e')
# np.savetxt(path + '-ahc-k%d.txt' % kmesh, output, fmt='%16.6f')
# np.savetxt(path + '-occ-k%d.txt' % kmesh, output, fmt='%16.6f')