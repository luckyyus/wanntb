import numpy as np
import os
import wann_tb

if __name__ == "__main__":
    path = 'D:\\cluster_data\\vasp.output\\39.SrCoO\\wann.01\\wann-pu4a-a1-soc-co1-fm110-k20-w124-b196'
    tbfile = os.path.join('', path, 'wannier90_tb.dat')
    print(tbfile)
    system = wann_tb.TBSystem(tb_file=tbfile)
    onsite = system.get_onsite_energy()
    print(onsite[0:5])
    print(onsite[62:67])
    print(onsite[5:10])
    print(onsite[67:72])

