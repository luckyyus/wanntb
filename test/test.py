import numpy as np
import os
import wann_tb

if __name__ == "__main__":
    ### FeGe pbe ###
    path = 'wann-FeGe-pbe'
    ef = 6.8805
    mag = 4.2905213
    ################
    # path = 'wann-FeGe-u2'
    tbfile = os.path.join('',path, 'wannier90_tb.dat')
    print(tbfile)
    system = wann_tb.TBSystem(tb_file=tbfile)


    ### FeGe u2 ###
    # ef = 7.2398
    # mag = 8.0
    ###############
    print('\n')
    carrier = system.get_carrier((16, 16, 16), ef)
    print(carrier)
    print('\n')
    carrier = system.get_carrier((24, 24, 24), ef)
    print(carrier)
    # for eta in [0.30, 0.20, 0.10]:
    #     print('\n')
    #     alpha, beta = system.get_alpha_beta((16, 16, 16), ef, mag, eta=eta)
    #     ratio = beta / alpha
    #     print('alpha = %f, beta = %f, ratio = %f' % (alpha, beta, ratio))


