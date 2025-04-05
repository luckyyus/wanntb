import numpy as np
import os
import wanntb

# for uu in ['05', '10', '15', '20']:
# path = 'MBT-sl3-soc-af001'
# path = 'AgRuO-soc'
path = 'MnTe-u4-soc-af1x'
#     path = 'FeGe-u%s-soc' % uu
tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')

tb = wanntb.TBSystem(tb_file=tbfile)
tb.output_npz(path)

tb2 = wanntb.TBSystem(npz_file=path + '-tb.npz')