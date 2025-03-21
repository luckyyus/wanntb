import numpy as np
import os
import wanntb

kpath = np.array([[0.00, 0.00, 0.00],
                      [0.50, 0.00, 0.00],
                      [0.50, 0.50, 0.00],
                      [0.00, 0.00, 0.00],
                      [0.50, 0.50, 0.50],
                      [0.00, 0.50, 0.00]])

    # path = 'Fe-soc-fm111'
    # eta = 0.01
    # ef = 5.4363
    # mag = 2.2395
uu = ['00', '05', '10', '15', '20']
efs = [6.8805, 6.9398, 7.0203, 7.1337, 7.2398]
mags = [4.2905213, 4.8207855, 5.8574778, 7.3501370, 8.000]
eta = 0.20
for iu in [3,4]:
    path = 'FeGe-u%s-soc' % uu[iu]
    ef = efs[iu]
    mag = mags[iu]
    tbfile = os.path.join('..','tbdata', path, 'wannier90_tb.dat')
    tb = wanntb.TBSystem(tb_file=tbfile)
    output = tb.get_alpha_beta_kpath(kpath, ef, mag, eta)
    np.savetxt(path + '-ab-kpath.txt', output, fmt='%20.12e')