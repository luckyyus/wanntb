import numpy as np
import pandas as pd

from .constant import TwoPi
from .utility import get_dos_e_kpar
from time import time
from ._system import TBSystem
"""
input.yml文件的设置：
    efermi: 0.0 # 费米能级(eV)，用来把输入的哈密顿量费米能级调到0；计算时会把电极H0对角元上调这个能级
    e_range: [-0.5, 0.5] # 相对费米能级的需要计算的能量上下限
    e_num: 1000 # 需要计算的能量的数量
    projections: dict{'atom': 'orbit'} #轨道projection信息
    计算表面格林函数部分：
    lead_(l,r)_h0: # 电极的H0矩阵 （目前只考虑1维链）
    lead_(l,r)_t: # 电极到电极的跃迁矩阵（目前只考虑最近邻跃迁）
    lead_num_iter: 1000 #计算表面格林函数的迭代次数

    v_lc(rc): # 电极到器件的跃迁矩阵（具体谁是行谁是列让我想想）
    top(bottom)_orbit_name: # 与电极相连的轨道名称（目前只一个轨道）
    n_l(r): 1 # 与电极相连的原子数
    
    device_type: semiconductor(s)/metal(m) # 器件的类型，决定化学势的计算方式
    is_top_to_bottom: true/false  # 是否是左接上，右接下。（这个还没确定是作为input参数好还是直接写在main里）
"""


class LayeredSystem(TBSystem):

    def __init__(self, tb_file='wannier90_tb.dat',npz_file=None, lspinors=True):
        super().__init__(tb_file=tb_file, npz_file=npz_file)
        self.l_spinors = lspinors
        self.n_wann_half = self.num_wann // 2

    # 不用了
    def get_H0_k(self, k_2d, e_fermi):
        ham_k2d = np.zeros((self.num_wann, self.num_wann), dtype=complex)
        for ir in range(self.n_Rpts):
            # 只有Rz为0的H_R是需要考虑的
            if self.R_vec[ir, 2] == 0:
                fac = np.exp(1j * TwoPi * np.dot(k_2d, self.R_vec[ir, 0:2]))
                ham_k2d += self.ham_R[ir, :, :] * fac / self.n_degen[ir]
        return ham_k2d - np.eye(self.num_wann, dtype=complex) * e_fermi

    def get_dos(self, emin, emax, e_num, kmesh, efermi=0.0):
        start = time()  # debug
        e_list = np.linspace(emin, emax, e_num + 1)
        print(e_list.shape)
        n_e = e_num + 1
        nkpt = kmesh[0] * kmesh[1]
        kxs = np.linspace(0.0, 1.0, kmesh[0], endpoint=False, dtype=float)
        kys = np.linspace(0.0, 1.0, kmesh[1], endpoint=False, dtype=float)
        kpts = np.zeros((nkpt, 2), dtype=float)
        kptx, kpty = np.meshgrid(kxs, kys)
        kpts[:, 0] = kptx.reshape(nkpt)
        kpts[:, 1] = kpty.reshape(nkpt)
        dos = np.zeros((e_list.shape[0], 2), dtype=float)
        dos[:, 0] = e_list
        dos[:, 1] = get_dos_e_kpar(self.num_wann, self.ham_R, self.R_vec, efermi, n_e, e_list,
                                   nkpt)
        # for i in range(e_list.shape[0]):
        #     dos[i, 1] = get_dos_e(self.num_wann, self.ham_R, self.n_Rpts, self.R_vec, self.n_degen, efermi,
        #                           e_list[i], nkpt, kpts)
        #     print('%8.4f is finished. time: %8.2f' % (e_list[i], time() - start))
        print('Calculate DOS finished. time %8.2f' % (time() - start))
        return dos

    def get_tops_bottoms(self, orbital_list, n_top=1, n_bottom=1, top_oname='s', bottom_oname='s'):
        assert self.n_wann_half == len(orbital_list), 'num_wann必须是orbital_list长度的2倍'
        da = pd.DataFrame(orbital_list, columns=['orbit', 'z']).sort_values(by='z')
        tops = da[da.orbit == top_oname].index.tolist()[:n_top]
        bottoms = da[da.orbit == bottom_oname].index.tolist()[-n_bottom:]
        # 先不变
        n_top_half = len(tops)
        n_bottom_half = len(bottoms)
        tops1 = np.zeros(n_top_half * 2, dtype=int)
        bottoms1 = np.zeros(n_bottom_half * 2, dtype=int)
        tops1[:n_top_half] = tops
        tops1[n_top_half:] = tops1[:n_top_half] + self.n_wann_half
        bottoms1[:n_bottom_half] = bottoms
        bottoms1[n_bottom_half:] = bottoms1[:n_bottom_half] + self.n_wann_half
        return tops1, bottoms1

    def get_u_c(self, k, orbital_list):
        # 计算hamk
        ham_k = np.zeros((self.num_wann, self.num_wann), dtype=complex)
        for ir in range(self.n_Rpts):
            # 只有Rz为0的H_R是需要考虑的
            if self.R_vec[ir, 2] == 0:
                fac = np.exp(1j * TwoPi * np.dot(k, self.R_vec[ir, 0:2]))
                ham_k += self.ham_R[ir, :, :] * fac / self.n_degen[ir]
        # 分层
        da1 = pd.DataFrame(orbital_list, columns=['orbit', 'z']).sort_values(by='z')
        half = int(len(orbital_list)/2)
        top_u = da1.index.tolist()[:half]
        top_d = (np.array(top_u)+self.n_wann_half).tolist()
        top = top_u+top_d
        bot_u = da1.index.tolist()[-half:]
        bot_d = (np.array(bot_u) + self.n_wann_half).tolist()
        bot = bot_u+bot_d
        t1,t2 = np.meshgrid(top, top)
        b1,b2 = np.meshgrid(bot, bot)
        # 上下层相减
        h_u = ham_k[t1,t2]-ham_k[b1,b2]
        u_c = np.sum(h_u)/(self.n_wann_half*self.n_wann_half)
        return u_c.real


