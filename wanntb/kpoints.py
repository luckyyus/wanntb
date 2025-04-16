import numpy as np


def get_kpts_mesh(kmesh):
    k1, k2, k3 = np.meshgrid(np.arange(kmesh[0], dtype=float)/kmesh[0],
                             np.arange(kmesh[1], dtype=float)/kmesh[1],
                             np.arange(kmesh[2], dtype=float)/kmesh[2], indexing='ij')
    return np.column_stack((k1.ravel(), k2.ravel(), k3.ravel()))


def get_kpts_mesh_around(kmesh, center, distance_cart, recip_lattice):
    k1, k2, k3 = np.meshgrid(np.arange(kmesh[0], dtype=float) / kmesh[0],
                             np.arange(kmesh[1], dtype=float) / kmesh[1],
                             np.arange(kmesh[2], dtype=float) / kmesh[2], indexing='ij')
    kpts0 = np.column_stack((k1.ravel(), k2.ravel(), k3.ravel()))
    nk = kpts0.shape[0]
    kpts = []
    for ik in range(nk):
        kpt = kpts0[ik] - np.array([0.5, 0.5, 0.5]) + center
        dk_cart = (kpt - center) @ recip_lattice
        r_cart = dk_cart / distance_cart
        if r_cart.dot(r_cart) < 1:
            kpts.append(kpt)
    return np.array(kpts)


def get_kpts_path(kpath, nkpts_path, recip_lattice):
    npath = len(kpath) - 1
    kbegin = 0.0
    kpts = []
    kpts_len = []
    for ip in range(npath):
        kdelta = (kpath[ip+1] - kpath[ip]) @ recip_lattice
        klen = np.sqrt(np.dot(kdelta, kdelta))
        for il in range(nkpts_path + 1):
            kpt = ((nkpts_path - il)* kpath[ip] + il * kpath[ip+1]) / nkpts_path
            kpt_len = klen * il / nkpts_path
            kpts.append(kpt)
            kpts_len.append(kpt_len + kbegin)
        kbegin += klen
    return np.array(kpts, dtype=float), np.array(kpts_len, dtype=float)


def get_adpt_kpts(dk, adpt_mesh):
    # print(dk.shape, dk.dtype)
    # print(adpt_mesh.shape, adpt_mesh.dtype)
    offset = 0.5 - (adpt_mesh + 1) % 2 / 2 / adpt_mesh
    # print(offset)
    kk1 = np.arange(adpt_mesh[0], dtype=float) / adpt_mesh[0] - offset[0] if adpt_mesh[0] > 1 else np.array([0.0])
    kk2 = np.arange(adpt_mesh[1], dtype=float) / adpt_mesh[1] - offset[1] if adpt_mesh[1] > 1 else np.array([0.0])
    kk3 = np.arange(adpt_mesh[2], dtype=float) / adpt_mesh[2] - offset[2] if adpt_mesh[2] > 1 else np.array([0.0])
    k1, k2, k3 = np.meshgrid(kk1, kk2, kk3, indexing='ij')

    kpts = np.column_stack((k1.ravel(), k2.ravel(), k3.ravel())) * dk
    return kpts