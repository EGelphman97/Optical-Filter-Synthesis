#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#January 19, 2020

#Implementation of Madsen and Zhao's Optical FIR Lattice Filter Design Algorithm
#Version 1.0.0

import numpy as np
import designFilter as df

"""
Function to find the polynomial B_N(z) which is needed to find the 2x2 transfer function of the optical filter
Parameters: A: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that also is part of the 2x2 transfer function
            phi_total: Total phase shift of filter
Return:     Polynomial(poly1d) in z^-1 of degree N B_N(z)
"""
def findBPoly(A,phi_total):
    phase_arr = np.zeros(A.coef.size, dtype=complex)
    phase_arr[0] = np.exp(-np.imag(phi_total))#)nly coef. of Z^(-N) term is nonzero for phase term
    phase_poly = np.poly1d(phase_arr)#poly1d object, only coef. of Z^(-N) term is nonzero
    A_R = np.poly1d(np.flip(A.coef))
    bbr_coef = -np.polymul(A.coef,np.flip(A.coef))
    bbr = np.poly1d(bbr_coef)
    bbr = np.polyadd(bbr,phase_poly)#Polynomial B_N(z)B_NR(z)
    roots = bbr.roots#Find roots of B_N(z)B_NR(z)
    print(roots)
    #Construct Polynomial B_N(z) and B_NR(z)
    b_roots = [];#List of roots of polynomial B_N(z) 
    br_roots = [];#List of roots of polynomial B_NR(z)
    for ii in range(0,roots.size - 1):
        zp = roots[ii]
        #Spectral Factorization: Pick which roots to assign to B(z), which to assign to B_R(z)
        if ii % 2 == 0:
            b_roots.append(zp)
        else:
            br_roots.append(zp)
    B_tild = np.poly1d(b_roots, True)#Construct polynomial from its roots
    alpha = np.sqrt((-A.coef[0]*A.coef[A.coef.size - 1])/(B_tild.coef[0]*B_tild.coef[B_tild.coef.size - 1]))#Scale factor
    B = alpha*B_tild#Build B_N(z) by scaling B_tild(z) by alpha
    return np.poly1d(B)

"""
Function to compute the coupling coefficient (kappa_N)^2 of stage N of the optical filter
Parameters: A_N: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that is part of the 2x2 transfer function
            B_N: Polynomial(poly1d) in z^-1 of degree N that is part of the 2x2 transfer function
Return:     kappa_N_sq: Coupling coefficient of stage N of the filter
"""
def calcCouplingCoef(A_N, B_N):
    arr = B_N.coef
    beta = np.absolute(B_N.coef[B_N.coef.size-1]/A_N.coef[A_N.coef.size-1])**2
    kappa_N_sq = beta/(1.0 + beta)
    return kappa_N_sq


def main():
    A_N = np.poly1d(df.kaiserWindow(0.65*np.pi, 0.5*np.pi, 0.07, 0.05))
    N = A_N.coef.size - 1
    #A_N = np.poly1d([-0.25,0.5,-0.25])
    print(A_N)
    gamma = 1.0
    phi_total = 0.0
    B_N = findBPoly(A_N,phi_total)
    results = []#Array of tuples (kappa_N_sq,phi_N)
    n = N
    while n >= 0:
        kappa_n_sq = calcCouplingCoef(A_N,B_N)
        c_n = np.sqrt(1.0-kappa_n_sq)
        s_n = np.sqrt(kappa_n_sq)
        B_N1_arr = np.polyadd((-s_n)*A_N,c_n*B_N)#Step-down recursion relation for B polynomial of stage N-1, this is an ndArray
        if n != 0:
            B_N1_arr = B_N1_arr[0:B_N1_arr.size-1]#Chop off term with highest negative power of z
        A_N1_tild = np.polyadd(c_n*A_N,s_n*B_N)
        phi_n = -(np.angle(A_N1_tild[A_N1_tild.size-2])+np.angle(B_N1_arr[B_N1_arr.size-1]))
        A_N1_tild = A_N1_tild[1:]#Chop off term with highest positive power of z
        A_N1 = np.poly1d(A_N1_tild)#Build polynomial A_N1(z)
        results.insert(0,(kappa_n_sq,phi_n))
        n = n - 1
        A_N = A_N1
        B_N = np.poly1d(B_N1_arr)
        
    for ii in range(len(results)):
        print("kappa sub " + str(ii) + " squared: " + str(results[ii][0]) + "  phase: " + str(results[ii][1]))
    
                 
if __name__ == '__main__':
    main()
    
        
    
    
    
    
