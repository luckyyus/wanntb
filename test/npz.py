import numpy as np
import os
import wanntb

# for uu in ['05', '10', '15', '20']:
path = 'MBT-sl3-soc-af001'
# path = 'MBT-sl2-soc-af001'
# path = 'AgRuO-soc'
# path = 'MnSe-soc-af100'
# path = 'MnTe-u4-soc-af1x'
#     path = 'FeGe-u%s-soc' % uu
tbfile = os.path.join('..', 'tbdata', path, 'wannier90_tb.dat')
# ssfile = os.path.join('..', 'tbdata', path, 'wannier90_SS_R.dat')
npzfile = os.path.join('..', 'tbdata', path+'-tb.npz')
posfile = os.path.join('..', 'tbdata', path+'.vasp')
# tb = wanntb.TBSystem(tb_file=tbfile)
# tb.load_spins(ss_file=ssfile)
# tb.output_npz(path)

# tb2 = wanntb.TBSystem(npz_file=path + '-tb.npz')
# tb2.load_poscar(pos_file=posfile)
tb = wanntb.get_tbsystem_by_npz_file(npz_file=npzfile)
# tb = wanntb.get_tbsystem_by_tb_file(tb_file=tbfile)
# tb.load_poscar(pos_file=posfile)
# tb.output_npz(seedname=path)