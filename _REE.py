import numpy as np
from numba import njit, prange
from .constant import TwoPi, Hbar_, Mu_B_
from .utility import fourier_phase_R_to_k, fourier_R_to_k, fourier_R_to_k_vec3, unitary_trans, occ_fermi



I_A = np.array([1, 2, 0], dtype=np.int32)
I_B = np.array([2, 0, 1], dtype=np.int32)


@njit(nogil=True)
def get_delta_E(eig,ef,tau):
    return tau/((ef-eig)**2 + tau*tau) /(TwoPi)


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
        vh_k[i] = vh_bar_k[i] - 1j*e_d*Abar_h_k[i]
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
    


    return morb1,vh_k,eig
    




@njit(parallel=True, nogil=True)
def get_REtensors_kpar_kmesh(ham_R,r_mat_R, R_vec, _R_cartT, ss_R,num_wann, kpts,ef_list,tau,mode):
    nkpts = kpts.shape[0]
    num_ef = ef_list.shape[0]
    REE= np.zeros((num_ef,9,3,nkpts), dtype=np.float64)
    for ik in prange(nkpts):
        kpt = kpts[ik]
        if mode== 10:
            REE[:,:,0,ik],REE[:,:,1,ik],REE[:,:,2,ik] = get_orb_REtensors_kpath(ham_R,r_mat_R, R_vec, _R_cartT, num_wann, kpt,ef_list,tau)
        elif mode == 20:
            REE[:,:,0,ik],REE[:,:,1,ik],REE[:,:,2,ik] = get_spin_REtensors_kpath(ham_R,r_mat_R,R_vec, _R_cartT, num_wann, kpt,ef_list,ss_R,tau)
    return (np.sum(REE, axis=3) / nkpts )

@njit(nogil=True)
def get_orb_REtensors_kpath(ham_R,r_mat_R, R_vec, _R_cartT, num_wann, kpt,ef_list,tau):
    factor = 0.262
    num_ef =ef_list.shape[0]
    index1 = (0,0,0,1,1,1,2,2,2)
    index2 = (0,1,2,0,1,2,0,1,2)

    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, _R_cartT, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])
    e_d=  np.zeros((num_wann, num_wann), dtype=np.float64)
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    eig_mat  = np.zeros((num_wann,num_wann),dtype=np.float64)
    for i in range(num_wann):
        for j in range(num_wann):
            eig_mat[i,j]= 0.25*(eig[i]+eig[j])
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d[m_,n_] = eig[m_] - eig[n_]
            tmp = eig[m_] - eig[n_]
            inv_e_d[m_, n_] = - 1.0 / tmp if abs(tmp) > 1e-8 else 0.0
    Abar_h_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Dbar_h_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Dh_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    vh_bar_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    vh_k  = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    tmp = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Ah_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    # o_k = np.zeros((3, num_wann, num_wann,num_wann), dtype=np.complex128)
    for i in range(3):
        Abar_h_k[i] = unitary_trans(A_bar_k[i], uu)
        vh_bar_k[i] =   unitary_trans(ham_out[i + 1], uu)
        Dbar_h_k[i] = vh_bar_k[i] * inv_e_d
        Ah_k[i] = Abar_h_k[i] +1j*Dbar_h_k[i]
        Dh_k[i] =  Dbar_h_k[i] - 1j * Abar_h_k[i]
        vh_k[i] = vh_bar_k[i] - 1j*e_d*Abar_h_k[i]

    chi1 = np.zeros((num_ef,9),dtype=np.complex128)
    chi2 = np.zeros((num_ef,9),dtype=np.complex128)
    OAM_mat = np.zeros((3,num_wann,num_wann),dtype=np.complex128)
    e_d1 = np.zeros((num_wann, num_wann), dtype=np.complex128)
    
    
    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            # e_d1[m_,n_]  = eig[m_] - eig[n_]
            e_d1[m_,n_] = -1/((eig[m_] - eig[n_])**2 + tau**2)
            # inv_e_d[m_, n_] = - 1.0 / e_d if abs(e_d) > 1e-8 else 0.0
 
    for I in range(num_ef):
        f_ij =  np.zeros((num_wann, num_wann), dtype=np.complex128)
        fo_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
        tmp = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
        deltaU = np.zeros((3,num_wann,num_wann),dtype=np.complex128)
        morb1_1= np.zeros((3,num_wann,num_wann),dtype=np.complex128)
        morb1_2  = np.zeros((3,num_wann,num_wann),dtype=np.complex128)


        ef = ef_list[I]
        f =occ_fermi(eig,ef,eta=1e-8)
        g=1-f
        for m_ in range(num_wann):
            for n_ in range(num_wann):
                    f_ij[m_,n_] = f[m_] - f[n_]
        for i in range(3):
            for m_ in range(num_wann):
                for n_ in range(num_wann):
                    if f_ij[m_,n_]!=0 or m_==n_:
                        fo_k[i, m_, n_] = np.sum(g *(Ah_k[I_A[i], m_, :] * Ah_k[I_B[i], :, n_]
                                              - Ah_k[I_A[i], :, n_] * Ah_k[I_B[i], m_, :]) )
        fo_k *= 1j 
        for i in range(3):
            tmp[i] = ((np.diag(g)).astype(np.complex128))@Dh_k[i]
        
        for  i in range(num_wann):
            for j in range(num_wann):
                for ii in range(3):
                    deltaU[ii,:,i] += tmp[ii,j,i]*uu[:,j]



        
        for i in range(3):
            morb1_2[i] = eig_mat* fo_k[i]

        operator  =  0.5*ham_out[0]
        for i in range(num_wann):
            for j in range(num_wann):
                if f_ij[i,j]!=0 or i==j:
                    morb1_1[0,i,j] = deltaU[1,:,i].conj().T @operator @deltaU[2,:,j] - deltaU[2,:,i].conj().T@ operator @ deltaU[1,:,j]
                    morb1_1[1,i,j] = deltaU[2,:,i].conj().T @operator @deltaU[0,:,j] - deltaU[0,:,i].conj().T@ operator @ deltaU[2,:,j]
                    morb1_1[2,i,j] = deltaU[0,:,i].conj().T @operator @deltaU[1,:,j] - deltaU[1,:,i].conj().T@ operator @ deltaU[0,:,j]
        morb1_1 *= -1j
        morb1 = morb1_1+morb1_2
        for i in range(3):
            OAM_mat[i] = factor*f_ij*e_d1*morb1[i]
        for i in range(9):
            chi1[I,i] =1j * np.sum(np.diag(OAM_mat[index1[i]] @ vh_k[index2[i]])) 
            chi2[I,i] =  np.sum(get_delta_E(eig,ef,tau)*np.diag(factor*morb1[index1[i]])*np.diag(vh_k[index2[i]]))/tau
    return chi1.real , chi2.real ,(chi1+chi2).real


@njit(nogil=True)
def get_spin_REtensors_kpath(ham_R,r_mat_R, R_vec, _R_cartT, num_wann, kpt,ef_list,ss_R,tau):
    num_ef = ef_list.shape[0]
    index1 = (0,0,0,1,1,1,2,2,2)
    index2 = (0,1,2,0,1,2,0,1,2)
    fac = fourier_phase_R_to_k(R_vec, kpt)
    ham_out = fourier_R_to_k(ham_R, _R_cartT, fac, iout=[1, 2, 3])
    A_bar_k = fourier_R_to_k_vec3(r_mat_R, fac)
    eig, uu = np.linalg.eigh(ham_out[0])    
    vh_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    Ah_bar_k = np.zeros((3, num_wann, num_wann), dtype=np.complex128)
    chi1 = np.zeros((num_ef,9),dtype=np.complex128)
    chi2 = np.zeros((num_ef,9),dtype=np.complex128)
    tmp2 = np.zeros((9,num_wann),dtype=np.complex128)
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.complex128)
    e_d = np.zeros((num_wann, num_wann), dtype=np.complex128)
    sw  = fourier_R_to_k_vec3(ss_R, fac)
    s_mat = np.zeros((3,num_wann,num_wann),dtype=np.complex128)
    s_mat1 = np.zeros((3,num_wann,num_wann),dtype=np.complex128)

    for m_ in range(num_wann):
        for n_ in range(num_wann):
            if m_ == n_:
                continue
            e_d[m_,n_]  = eig[m_] - eig[n_]
            inv_e_d[m_,n_] = -1/((eig[m_] - eig[n_])**2 + tau**2)

    for i in range(3):
        Ah_bar_k[i] = unitary_trans(A_bar_k[i],uu)  
        vh_k[i] = unitary_trans(ham_out[i + 1], uu) - 1j*e_d*Ah_bar_k[i]
        s_mat[i] = unitary_trans(sw[i],uu)
        s_mat1[i] = inv_e_d*s_mat[i]
        

    f_ij =  np.zeros((num_wann, num_wann), dtype=np.complex128)
    # f =occ_fermi(eig,ef,eta=1e-8)
    for i in range(9):
        tmp2[i]=np.diag(s_mat[index1[i]])*np.diag(vh_k[index2[i]])/tau


    for I in range(num_ef):
        ef = ef_list[I]
        f  = occ_fermi(eig,ef,eta=1e-8)
        delta=get_delta_E(eig,ef,tau)
        for m_ in range(num_wann):
            for n_ in range(num_wann):
                if m_ == n_:
                    continue
                f_ij[m_,n_] = f[m_] - f[n_]
    
        for i in range(9):
            chi1[I,i] =1j * np.sum(np.diag((f_ij * s_mat1[index1[i]])@ vh_k[index2[i]]))
            chi2[I,i] =np.sum(delta*tmp2[i])
        chi= chi1 +chi2
    return chi1.real ,chi2.real ,chi.real









@njit(nogil=True)
def get_inv_e_d(eig, num_wann):
    """
    inv_e_d[m, n] = 1 / (e_n - e_m)
    """
    inv_e_d = np.zeros((num_wann, num_wann), dtype=np.float64)
    for n_ in range(num_wann):
        for m_ in range(num_wann):
            if m_ == n_:
                continue
            e_d = eig[n_] - eig[m_]
            inv_e_d[m_, n_] = 1.0 / e_d if abs(e_d) > 1e-8 else 0.0
    return inv_e_d