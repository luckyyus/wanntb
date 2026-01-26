import numpy as np
from scipy.linalg import expm, inv

# Pauli矩阵
S_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
S_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
S_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def L_matrix(l):
    """
    生成角动量算符L_x, L_y, L_z的矩阵表示（用于球谐函数基）。
    参数:
        l: 角动量量子数。
    返回:
        元组 (Lx, Ly, Lz)，每个是(2l+1)x(2l+1)的复数矩阵。
    """
    dim = 2 * l + 1
    Lz = np.zeros((dim, dim), dtype=np.complex128)
    Lp = np.zeros((dim, dim), dtype=np.complex128)  # 升算符L+
    Lm = np.zeros((dim, dim), dtype=np.complex128)  # 降算符L-
    
    # 填充Lz和升降算符
    for m in range(-l, l+1):
        idx = m + l  # 索引从0开始
        Lz[idx, idx] = m
        if m < l:
            Lp[idx, idx+1] = np.sqrt((l - m) * (l + m + 1))  # L+|lm>
        if m > -l:
            Lm[idx, idx-1] = np.sqrt((l + m) * (l - m + 1))  # L-|lm>
    
    # Lx = (L+ + L-)/2, Ly = (L+ - L-)/(2j)
    Lx = (Lp + Lm) / 2
    Ly = (Lp - Lm) / (2j)
    
    return Lx, Ly, Lz

def R2Y_Y2R(l):
    """
    生成球谐函数(Ylm)和实球谐函数(Real Spherical Harmonics)之间的变换矩阵。
    
    参数:
        l: 角动量量子数 (0=s, 1=p, 2=d, 3=f)
    
    返回:
        R2Y: (shape: (2l+1) x (2l+1))
        Y2R:  (shape: (2l+1) x (2l+1))
    """
    N = 2 * l + 1
    v2 = np.sqrt(2.0)
    
    # 初始化变换矩阵
    R2Y = np.zeros((N, N), dtype=np.complex128)
    
    if l == 0:  # s轨道
        R2Y[0, 0] = 1.0
    elif l == 1:  # p轨道
        # pz = |1,0>
        R2Y[0, 1] = 1.0  # pz对应m=0
        
        # px = (|1,-1> - |1,1>)/sqrt(2)
        R2Y[1, 0] = 1.0/v2   # m=-1分量
        R2Y[1, 2] = -1.0/v2  # m=+1分量
        
        # py = i(|1,-1> + |1,1>)/sqrt(2)
        R2Y[2, 0] = 1j/v2    # m=-1分量
        R2Y[2, 2] = 1j/v2    # m=+1分量
        
    elif l == 2:  # d轨道
        # dz2 = |2,0>
        R2Y[0, 2] = 1.0
        
        # dzx = (|2,-1> - |2,1>)/sqrt(2)
        R2Y[1, 1] = 1.0/v2
        R2Y[1, 3] = -1.0/v2
        
        # dyz = i(|2,-1> + |2,1>)/sqrt(2)
        R2Y[2, 1] = 1j/v2
        R2Y[2, 3] = 1j/v2
        
        # dx2-y2 = (|2,-2> + |2,2>)/sqrt(2)
        R2Y[3, 0] = 1.0/v2
        R2Y[3, 4] = 1.0/v2
        
        # dxy = i(|2,-2> - |2,2>)/sqrt(2)
        R2Y[4, 0] = 1j/v2
        R2Y[4, 4] = -1j/v2
        
    elif l == 3:  # f轨道
        # fz3 = |3,0>
        R2Y[0, 3] = 1.0
        
        # fxz2 = (|3,-1> - |3,1>)/sqrt(2)
        R2Y[1, 2] = 1.0/v2
        R2Y[1, 4] = -1.0/v2
        
        # fyz2 = i(|3,-1> + |3,1>)/sqrt(2)
        R2Y[2, 2] = 1j/v2
        R2Y[2, 4] = 1j/v2
        
        # fz(x2-y2) = (|3,-2> + |3,2>)/sqrt(2)
        R2Y[3, 1] = 1.0/v2
        R2Y[3, 5] = 1.0/v2
        
        # fxyz = i(|3,-2> - |3,2>)/sqrt(2)
        R2Y[4, 1] = 1j/v2
        R2Y[4, 5] = -1j/v2
        
        # fx3-3xy2 = (|3,-3> - |3,3>)/sqrt(2)
        R2Y[5, 0] = 1.0/v2
        R2Y[5, 6] = -1.0/v2
        
        # fy(3x2-y2) = i(|3,-3> + |3,3>)/sqrt(2)
        R2Y[6, 0] = 1j/v2
        R2Y[6, 6] = 1j/v2
    
    else:
        # 对于其他l值，使用单位矩阵（简化处理）
        R2Y = np.eye(N, dtype=np.complex128)
    
    # 计算逆矩阵：Y2C = C2Y^{-1}
    Y2R = inv(R2Y)
    
    return R2Y, Y2R


def rotate_Ylm(l, axis, alpha, inv=False):
    """
    生成球谐函数基下的旋转矩阵。
    参数:
        l: 角动量量子数。
        axis: 旋转轴。
        alpha: 旋转角度。
        inv: 是否反演。
    返回:
        (2l+1)x(2l+1)旋转矩阵。
    """
    Lx, Ly, Lz = L_matrix(l)
    # 旋转生成器: n·L
    L_dot_n = axis[0] * Lx + axis[1] * Ly + axis[2] * Lz
    # 旋转矩阵: exp(-i * alpha * n·L)
    rotation_matrix = expm(-1j * alpha * L_dot_n)
    
    if inv:
        # 反演处理：根据l的奇偶性调整符号
        if l % 2 == 1:
            rotation_matrix = -rotation_matrix
    
    return rotation_matrix


def rotate_real_Ylm(l, axis, alpha, inv=False):
        """
        生成实球谐函数基下的旋转矩阵。
        参数:
            l: 角动量量子数。
            axis: 旋转轴。
            alpha: 旋转角度。
            inv: 是否反演。
        返回:
            (2l+1)x(2l+1)旋转矩阵。
        """
        r2y, y2r = R2Y_Y2R(l)
        rot_Ylm = rotate_Ylm(l, axis, alpha, inv)
        rot_r = r2y @ rot_Ylm @ y2r
        return rot_r


def rotate_spinor(axis, alpha, inv=False):
    """
    生成SU(2)旋量旋转矩阵。
    参数:
        axis: 3元素数组，旋转轴（单位向量）。
        alpha: 旋转角度。
        inv: 是否包含反演（默认为False）。
    返回:
        2x2复数矩阵，表示旋转。
    """
    # 旋转轴点乘Pauli矩阵
    sigma_n = axis[0] * S_x + axis[1] * S_y + axis[2] * S_z
    # 计算旋转矩阵: exp(-i * alpha * (n·sigma) / 2)
    rotation_matrix = expm(-1j * alpha * sigma_n / 2)
    
    if inv:
        rotation_matrix = -1j * rotation_matrix  # inv时乘以-i
    
    return rotation_matrix


# def get_sym_op_reciprocal(lattice, orb_info, kpt, rotation, translation, TR):
#     """
#     简化版倒易空间对称操作符计算。
#     参数:
#         lattice: 晶格向量（3x3矩阵）。
#         orb_info: 轨道信息（列表字典，包含位置等）。
#         kpt: k点向量。
#         rotation: 实空间旋转矩阵。
#         translation: 平移。
#         TR: 时间反演标志。
#     返回:
#         对称操作符矩阵（norb x norb）。
#     """
#     norb = len(orb_info)
#     sym_op = np.zeros((norb, norb), dtype=np.complex128)
    
#     # 简化：假设每个轨道是局域的，旋转只影响轨道基矢
#     # 生成轨道旋转矩阵（使用rotate_Ylm类似逻辑）
#     # 这里省略详细轨道类型匹配，仅示例
    
#     for i in range(norb):
#         for j in range(norb):
#             # 计算相位因子：e^{-i k·(R·r_j + t - r_i)} 等
#             # 简化处理：假设旋转后轨道位置变化
#             phase = np.exp(-2j * np.pi * np.dot(kpt, translation))  # 示例相位
#             # 轨道旋转部分（需根据轨道角动量计算）
#             orb_rot = 1.0  # placeholder，实际调用rotate_Ylm
#             sym_op[i, j] = phase * orb_rot
    
#     if TR:
#         # 时间反演：取共轭并可能乘以因子
#         sym_op = sym_op.conj()
    
#     return sym_op