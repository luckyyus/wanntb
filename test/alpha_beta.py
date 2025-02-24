import numpy as np
import os
import wanntb

if __name__ == "__main__":

    uu = ['00', '05', '10', '15', '20']
    efs = [6.8805, 6.9398, 7.0203, 7.1337, 7.2398]
    mags = [4.2905213, 4.8207855, 5.8574778, 7.3501370, 8.0014691]
    eta = 0.20
    # kmeshs = [32, 48, 64, 80, 96]
    kmeshs = [32, 64]
    for iu in [3]:
        path = 'FeGe-u%s-soc' % uu[iu]
        ef = efs[iu]
        mag = mags[iu]
        tbfile = os.path.join('..', 'data', path, 'wannier90_tb.dat')
        tb = wanntb.TBSystem(tb_file=tbfile)
        output = []
        for kmesh in kmeshs:
            # carrier = tb.get_carrier((kmesh, kmesh, kmesh), ef)
            # output.append([kmesh, carrier])
            alpha, alpha_qvd, qvs, beta, ratio = tb.get_alpha_beta((kmesh, kmesh, kmesh), ef, mag, eta=eta)
            output.append([kmesh, alpha, alpha_qvd, qvs, beta, ratio])
        np.savetxt('%s-ab-eta%.3f.txt' % (path, eta), np.array(output, dtype=float), fmt='%12.6f')
        print('\n')


    ### Fe111 ###
    # path = 'wann.10/wann-Fe-soc-fm111'
    # ef = 5.4363
    # mag = 2.2395
    #############

    ### Co111 ###
    # path = 'wann.10/wann-Co-soc-fm001'
    # ef = 5.4363
    # mag = 2.2395
    #############

    # kpath = np.array([[0.00, 0.00, 0.00],
    #                   [0.50, 0.00, 0.00],
    #                   [0.50, 0.50, 0.00],
    #                   [0.00, 0.00, 0.00],
    #                   [0.50, 0.50, 0.50],
    #                   [0.00, 0.50, 0.00]])
    # output = system.get_alpha_beta_kpath(kpath, ef, mag, eta=0.20, nkpts_path=100)
    # np.savetxt('kpath.txt', output, fmt='%20.12e')
    # carrier = system.get_carrier((24, 24, 24), ef)
    # print(carrier)
    # carrier = system.get_carrier((32, 32, 32), ef)
    # print(carrier)
    # carrier = system.get_carrier((48, 48, 48), ef)
    # print(carrier)
    # print('\n')
    # carrier = system.get_carrier((24, 24, 24), ef)
    # print(carrier)
    # for eta in [0.20]:
    #     alpha, beta = tb.get_alpha_beta((40, 40, 40), ef, mag, eta=eta)
    #     ratio = beta / alpha
    #     print('#alpha    #beta     #ratio    ')
    #     print('%10.6f%10.6f%10.6f' % (alpha, beta, ratio))
    #     print('\n')

