import numpy as np
from numba import njit, prange
from .constant import TwoPi, Hbar_, Mu_B_
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi


I_A = np.array([1, 2, 0], dtype=np.int32)
I_B = np.array([2, 0, 1], dtype=np.int32)

@njit(nogil=True)
def get_morb_mat(ef,ham_R,r_mat_R,R_vec, _R_cartT, num_wann, kpt):
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, _R_cartT, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    e_d = inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    f= occ_fermi(eig,ef,eta=1e-8)
    g=np.diag((1-f)) .astype(np.complex128)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d[m_,n_] = eig[m_] - eig[n_]
            e_d1 = eig[m_] - eig[n_]
            inv_e_d[m_, n_] = - 1.0 / e_d1 if abs(e_d1) > 1e-8 else 0.0
    Abar_h_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Dbar_h_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Dh_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    vh_bar_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    vh_k  = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    tmp = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Ah_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    fo_k  = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    for i in range(3):
        Abar_h_k[i] = unitary_trans(A_bar_k[i], uu)
        vh_bar_k[i] =   unitary_trans(ham_out[i + 1], uu)
        # D^H_a = UU^dag.del_a UU (a=x,y,z) = {H_bar^H_nma / (e_m - e_n)}
        Dbar_h_k[i] = vh_bar_k[i] * inv_e_d
        Ah_k[i] = Abar_h_k[i] +1j*Dbar_h_k[i]
        Dh_k[i] =  Dbar_h_k[i] - 1j * Abar_h_k[i]
        vh_k[i] = vh_bar_k[i] + 1j*e_d*Abar_h_k[i]
        tmp[i] = g@Dh_k[i]
    
    deltaU = np.zeros((3,num_wann,num_wann),dtype=np.complex128)
    for  i in range(num_wann):
        for j in range(num_wann):
            for ii in range(3):
                deltaU[ii,:,i] += tmp[ii,j,i]*uu[:,j]
    g=1-f 
    for i in range(3):
        for m_ in range(num_wann):
            for n_ in range(num_wann):
                fo_k[i, m_, n_] = np.sum(g * (Ah_k[I_A[i], m_, :] * Ah_k[I_B[i], :, n_]
                                              - Ah_k[I_A[i], :, n_] * Ah_k[I_B[i], m_, :]))
    fo_k *= 1j 
    
    morb1_1= np.zeros((3,num_wann,num_wann),dtype=np.complex128)
    morb1_2  = np.zeros((3,num_wann,num_wann),dtype=np.complex128)
    eig_mat  = np.zeros((num_wann,num_wann),dtype=np.float64)
    for i in range(num_wann):
        for j in range(num_wann):
            eig_mat[i,j]= 0.25*(eig[i]+eig[j])
    for i in range(3):

        morb1_2[i] = eig_mat* fo_k[i]

    operator  =  0.5*ham_out[0]
    for i in range(num_wann):
         for j in range(num_wann):
            morb1_1[0,i,j] = deltaU[1,:,i].conj().T @operator @deltaU[2,:,j] - deltaU[2,:,i].conj().T@ operator @ deltaU[1,:,j]
            morb1_1[1,i,j] = deltaU[2,:,i].conj().T @operator @deltaU[0,:,j] - deltaU[0,:,i].conj().T@ operator @ deltaU[2,:,j]
            morb1_1[2,i,j] = deltaU[0,:,i].conj().T @operator @deltaU[1,:,j] - deltaU[1,:,i].conj().T@ operator @ deltaU[0,:,j]
    morb1_1 *= -1j
    morb1 = morb1_1+morb1_2
    
    return morb1,vh_k,eig,f

@njit(parallel=True, nogil=True)
def get_OHE_kpar_kmesh(ham_R,r_mat_R, R_vec, _R_cartT,num_wann, kpts,ef,dir):
    nkpts = kpts.shape[0]
    # num_ef = ef_list.shape[0]
    OBC_f= np.zeros((3,nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        OBC,f = get_OBC_kpath(ham_R,r_mat_R, R_vec, _R_cartT,num_wann, kpt,ef,dir)
        for i in range(3):
            OBC_f[i,ik] = np.sum(f*OBC[i])

    return 2*np.sum(OBC_f, axis=1)/nkpts

@njit(nogil=True)
def get_OBC_kpath(ham_R,r_mat_R, R_vec, _R_cartT,num_wann, kpt,ef,dir):
    factor = 0.262
    morb,vh_k,eig,f=get_morb_mat(ef,ham_R,r_mat_R,R_vec, _R_cartT, num_wann, kpt)
    j_mat = np.zeros((3,num_wann,num_wann),dtype=np.complex128)
    inv_e_d = np.zeros((num_wann,num_wann),dtype=np.float64)
    OBC = np.zeros((3,num_wann),dtype=np.complex128)
    g=(1-f).astype(np.complex128)
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d1 = eig[m_] - eig[n_]
            inv_e_d[m_, n_] = - 1.0 / e_d1 if abs(e_d1) > 1e-8 else 0.0
    for i in range(3):
        mat= morb[dir]@vh_k[i]
        j_mat[i] = (mat+mat.conj().T)*0.5*factor*inv_e_d*inv_e_d
    for i in  range(3):
        for n_ in range(num_wann):
             OBC[i,n_] = np.sum(g*j_mat[I_A[i],n_,:]*vh_k[I_B[i],:,n_])
            #  OBC[i,n_] = np.sum(j_mat[I_A[i],n_,:]*vh_k[I_B[i],:,n_])
    return OBC.imag,f
    

    
    



