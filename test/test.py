import numpy as np
import os
import wann_tb

if __name__ == "__main__":
    # path = 'D:\\cluster_data\\vasp.output\\39.SrCoO\\wann.01\\wann-pu4a-a1-soc-co1-fm110-k20-w124-b196'
    # path = 'wann-FeGe-pbe'
    path = 'wann-FeGe-u2'
    tbfile = os.path.join('',path, 'wannier90_tb.dat')
    print(tbfile)
    system = wann_tb.TBSystem(tb_file=tbfile)
    ### FeGe pbe ###
    # ef = 6.8805
    # mag = 4.2905213
    ################

    ### FeGe u2 ###
    ef = 7.2398
    mag = 8.0
    ###############
    # for eta in [0.1, 0.07, 0.05, 0.03, 0.01, 0.005]:
    q = 1e-6
    for eta in [0.25, 0.20, 0.15, 0.10, 0.05]:
        print('\n')
        alpha, beta = system.get_alpha_beta((32, 32, 32), ef, eta=eta, q=q)
        alpha /= mag
        ratio = beta / alpha
        print('alpha = %f, beta = %f, ratio = %f' % (alpha, beta, ratio))


