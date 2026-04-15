from dataclasses import dataclass, field

import numpy as np

from .constant import Orbitals
from .utility import get_list_index


class Structure:
    """
    原胞结构, 从POSCAR读取
    """
    def __init__(self, pos_file='POSCAR', version=5):
        # 原始字符串数据的list
        raw = []
        with open(pos_file, 'r') as poscar:
            for line in poscar:
                raw.append(line.strip().split())
                pass
        # 原胞基矢
        self.real_lattice = np.array(raw[2:5], dtype=float) * float(raw[1][0])
        # 表示原子数量的行号，VASP5以前是第6行，之前是第5行而没有原子信息
        i_n_spec = 5 if version < 5 else 6
        # 原子类型list
        self.spec_names = raw[5] if version >= 5 else []
        # 获取各类原子的数目
        n_spec = np.array(raw[i_n_spec], dtype=int)
        # 原子坐标是笛卡尔坐标(c)还是相对基矢坐标(d)，先记着，目前默认都是d，并且不考虑select dynamics这种情况
        self.pos_type = raw[i_n_spec+1][0][0].lower()
        # print(self.pos_type)
        # 总原子数
        self.n_atoms = np.sum(n_spec)
        i_spec = 0
        self.spec_list = np.zeros(self.n_atoms, dtype=int)
        for i in range(n_spec.shape[0]):
            self.spec_list[i_spec: i_spec + n_spec[i]] = i
            i_spec += n_spec[i]
        # print(self.spec_list)
        # 原子位置
        self.atom_pos = np.array(raw[i_n_spec + 2: i_n_spec + 2 + self.n_atoms], dtype=float)
        # print(self.atom_pos)

    def get_orbital_list(self, projections: list):
        """
        从Projection和原子结构得出轨道列表，列表每项为[轨道名称,z轴坐标]
        :param projections:
        :return:
        """
        orb_list = []
        for proj in projections:
            spec_name, spec_orb0 = list(proj.items())[0]
            spec_id = get_list_index(spec_name, self.spec_names)
            spec_orb = []
            # 把一些轨道合计展开
            for orb in spec_orb0:
                spec_orb += Orbitals[orb]
            # print(spec_name, spec_orb)
            for i in range(self.n_atoms):
                if self.spec_list[i] == spec_id:
                    for orb in spec_orb:
                        orb_list.append([orb, self.atom_pos[i, 2]])
        return orb_list

    def area(self):
        return np.dot(self.real_lattice[0, :], self.real_lattice[1, :])


@dataclass
class WannOrb:
    """Wannier orbital information."""
    site: np.ndarray
    l: int
    ml: int
    ms: int = 0
    r: int = 1
    axis: np.ndarray = field(default_factory=lambda: np.eye(3))

    def __post_init__(self):
        self.site = np.asarray(self.site, dtype=np.float64)
        self.axis = np.asarray(self.axis, dtype=np.float64)

    def copy(self) -> 'WannOrb':
        return WannOrb(
            site=self.site.copy(),
            l=self.l,
            ml=self.ml,
            ms=self.ms,
            r=self.r,
            axis=self.axis.copy()
        )
