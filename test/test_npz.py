import numpy as np
import os
import yaml
import wanntb


path = 'MBT-sl3-soc-af001'
# path = 'GaAs-soc-001'

tbfile = os.path.join('tbdata', path, 'wannier90_tb.dat')
ssfile = os.path.join('tbdata', path, 'wannier90_SS_R.dat')
npzfile = os.path.join('tbdata', path+'-tb.npz')
posfile = os.path.join('tbdata', path, 'POSCAR.vasp')
orbfile = os.path.join('tbdata', path, 'projections.yml')
with open(orbfile, 'r', encoding='utf-8') as f:
    projections = yaml.safe_load(f)
print(projections)


tb = wanntb.get_tbsystem_by_tb_file(tb_file=tbfile)
tb.load_spins(ss_file=ssfile)
tb.load_poscar(pos_file=posfile)
tb.load_orbitals(projections, is_laxis=False, is_soc=True, order='uudd')
tb.output_npz(seedname=path)

tb2 = wanntb.get_tbsystem_by_npz_file(npz_file=path + '-tb.npz')