import numpy as np
import os
import wanntb
import yaml

#===== FeGe =====
etas = [0.20]
with open('FeGe.yml', 'r', encoding='utf-8') as f:
    str = f.read()
data = yaml.safe_load(str)

kmeshs = [64]

for iu in [0]:
    path = data[iu]['path']
    # path = paths[iu]
    ef = data[iu]['ef']
    mag = data[iu]['mag']
    # tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')
    npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')
    # tb = wanntb.TBSystem(tb_file=tbfile)
    tb = wanntb.TBSystem(npz_file=npzfile)
    for eta in etas:
        for kmesh in kmeshs:
            out = tb.get_alpha_beta_fermi((kmesh, kmesh, kmesh), ef, 0.01, 10, mag,
                                                                    eta=eta, adpt_mesh=(4,4,4))

            np.savetxt(path + '-ab-eta%.3f-k%d.txt' % (eta, kmesh), out, fmt='%12.6f')
            print('\n')