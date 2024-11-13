import numpy as np
import os
import wann_tb

if __name__ == "__main__":
    path = 'D:\\cluster_data\\vasp.output\\39.SrCoO\\wann.01\\wann-pu4a-a1-soc-co1-fm110-k20-w124-b196'
    tbfile = os.path.join('', path, 'wannier90_tb.dat')
    print(tbfile)
    system = wann_tb.TBSystem(tb_file=tbfile)
    onsite = system.get_onsite_energy()
    data = np.zeros((5,8), dtype=float)
    data[:, 0] = onsite[0:5]
    data[:, 1] = onsite[62:67]
    data[:, 2] = (data[:, 0] + data[:, 1]) / 2
    data[:, 3] = (-data[:, 0] + data[:, 1]) / 2
    data[:, 4] = onsite[5:10]
    data[:, 5] = onsite[67:72]
    data[:, 6] = (data[:, 4] + data[:, 5]) / 2
    data[:, 7] = (-data[:, 4] + data[:, 5]) / 2
    np.savetxt('sco327-orbital-energy.txt', data, fmt='%8.3f')

