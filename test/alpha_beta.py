import numpy as np
import os
import wanntb

#===== FeGe =====
uu = ['00','05','06','07','08','09', '10', '15', '20']
efs = [6.8805, 6.9398, 6.9511, 6.9688, 6.9906,7.0057, 7.0203, 7.1337, 7.2398]
mags = [4.2905213, 4.8207855,4.8833362, 5.1240044,5.5951034, 5.7582872, 5.8574778, 7.3501370, 8.0014691]
etas = [0.20]


#====== Fe bcc and Ni fcc =====
# paths = ['Fe-soc-fm111', 'Ni-soc-fm111']
# efs = [5.4363, 9.2951]
# mags = [2.2395390, 0.6422275]
# eta = 0.01

# kmeshs = [32, 64, 96, 128]
kmeshs = [96]

for iu in [1]:
    path = 'FeGe-u%s-soc' % uu[iu]
    # path = paths[iu]
    ef = efs[iu]
    mag = mags[iu]
    # tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')
    npzfile = os.path.join('tbdata', path + '-tb.npz')
    # tb = wanntb.TBSystem(tb_file=tbfile)
    tb = wanntb.TBSystem(npz_file=npzfile)
    output = []
    for eta in etas:
        for mu in [0.00]:
            for kmesh in kmeshs:
            # carrier = tb.get_carrier((kmesh, kmesh, kmesh), ef)
            # output.append([kmesh, carrier])
                alpha, alpha_qvd, qvs, beta, ratio = tb.get_alpha_beta((kmesh, kmesh, kmesh), ef+mu, mag,
                                                                    eta=eta, adpt_mesh=(4,4,4))
                output.append([eta, mu, kmesh, alpha, alpha_qvd, qvs, beta, ratio])
            print(output[-1])
    np.savetxt(path + '-ab.txt', np.array(output, dtype=float), fmt='%12.6f')
    print('\n')