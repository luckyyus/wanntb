import numpy as np
import os
import wanntb

if __name__ == "__main__":
    # set the path of the tight-binding file
    tbfile = 'wannier90_tb.dat'

    system = wanntb.TBSystem(tb_file=tbfile)

    print('\n')

    # generate k-points according to kpath
    kpath = np.array([[0.00, 0.00, 0.00],
                      [0.50, 0.00, 0.00],
                      [0.50, 0.50, 0.00],
                      [0.00, 0.00, 0.00],
                      [0.50, 0.50, 0.50],
                      [0.00, 0.50, 0.00]])
    kpts, klen = wanntb.utility.get_kpts_path(kpath, 100, system.recip_lattice)
    # klen is used to plot the band structure
    # or you can generate k-points by a mesh
    # kpts = wanntb.utility.get_kpts_mesh((96, 96, 64))

    # or you can generate k-points by your own functions
    # the k-points are in the fraction coordinate
    # output k-points in a file
    np.savetxt('k-points.txt', kpts, fmt='%12.8f')
    n_kpts = kpts.shape[0]
    eigs = np.zeros(n_kpts, dtype=float)
    for ik in range(n_kpts):
        eigs[ik], uu = system.get_eig_uu_for_one_kpt(kpts[ik])
    # output eigenvalues in a file
    np.savetxt('eigvalues.txt', eigs, fmt='%12.8f')





