import numpy as np
import os
import wanntb

# MBT-sl2
path = 'MBT-sl2-soc-af001'

npzfile = os.path.join('..', 'tbdata', path + '-tb.npz')

tb = wanntb.get_tbsystem_by_npz_file(npz_file=npzfile)

kpath = np.array([[0.00, 0.00, 0.00],
                  [0.50, 0.00, 0.00],
                  [1.0/3.0, 1.0/3.0, 0.00],
                  [0.00, 0.00, 0.00]])

# this part shows how to plot band structure using API functions manually
kpts, klen = wanntb.kpoints.get_kpts_path(kpath, 100, tb.recip_lattice)
# klen is used to plot the band structure
# or you can generate k-points by a mesh
# kpts = wanntb.kpoints.get_kpts_mesh((96, 96, 64))
np.savetxt('k-points.txt', kpts, fmt='%12.8f')

# kpoint-path is used to plot the band structure
n_kpts = kpts.shape[0]
eigs = np.zeros((n_kpts, tb.num_wann), dtype=float)
for ik in range(n_kpts):
    eigs[ik, :], uu = tb.get_eig_uu_for_one_kpt(kpts[ik])
# output eigenvalues in a file
np.savetxt('bands-one-by-one.txt', np.column_stack((klen, eigs)), fmt='%12.8f')

# or use output_bands_kpath
tb.output_bands_kpath(kpath, nkpts_path=100, filename='bands.txt')







