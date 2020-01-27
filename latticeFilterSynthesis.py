#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#January 27, 2020

#Implementation of Madsen and Zhao's Optical FIR Lattice Filter Design Algorithm
#Version 1.0.2

import designFilter as dF
import numpy as np

"""
Function to find the polynomial B_N(z) which is needed to find the 2x2 transfer function of the optical filter
Parameters: A: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that also is part of the 2x2 transfer function
            phi_total: Total phase shift of filter
Return:     Polynomial(poly1d) in z^-1 of degree N B_N(z)
"""
def findBPoly(A):
    phase_arr = np.zeros(A.coef.size)
    phase_arr[0] = 1.0#only coef. of Z^(-N) term is nonzero for phase term
    #print(np.poly1d(phase_arr))
    A_R = np.poly1d(np.flip(A.coef))
    bbr_coef = -1.0*np.polymul(A.coef,np.flip(A.coef))
    bbr = np.poly1d(np.polyadd(bbr_coef,phase_arr))#Polynomial B_N(z)B_NR(z)
    roots = bbr.roots#Find roots of B_N(z)B_NR(z)
    b_roots = roots[0:roots.size-1:2]#B(z)BR(z) has n roots, n is event, B(z) has n/2 roots
    print("Roots:")
    print(roots)
    print()
    B_tild = np.poly1d(b_roots, True)#Construct polynomial from its roots
    #print(B_tild)
    alpha = np.sqrt((-A.coef[0]*A.coef[A.coef.size - 1])/(B_tild.coef[0]*B_tild.coef[B_tild.coef.size - 1]))#Scale factor
    
    print(B_tild.coef[0]*B_tild.coef[B_tild.coef.size - 1])
    B = alpha*B_tild#Build B_N(z) by scaling B_tild(z) by alpha
    print("B: ")
    print(np.poly1d(B))
    print()
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

"""
Function to synthesize an FIR optical lattice filter using the algorithm outlined in Section 4.5 of Madsden and Zhao
Paramters: A_N: Polynomial(represented by NumPy's poly1d class) in z^-1 of degree N that is part of the 2x2 transfer function
           N:   Filter order
Return: kappa_l: List of coupling coefficients kappa_n_sq list index is same as n
        phi_l:   List of phase terms phi_n list index is n-1
"""
def synthesizeFIRLattice(A_N, N):
    gamma = 1.0
    phi_total = 0.0
    B_N = findBPoly(A_N)
    kappa_l = []#List of kappa_n_sq's
    phi_l = []#List of phi_n's
    n = N
    while n >= 0:
        #print(A_N)
        #print(B_N)
        kappa_n_sq = calcCouplingCoef(A_N,B_N)
        kappa_l.insert(0,kappa_n_sq)
        if n > 0:
            c_n = np.sqrt(1.0-kappa_n_sq)
            s_n = np.sqrt(kappa_n_sq)
            B_N1_arr = np.polyadd(-s_n*A_N,c_n*B_N)#Step-down recursion relation for B polynomial of stage N-1, this is an ndArray
            print(B_N1_arr)
            B_N1 = np.poly1d(B_N1_arr[0:B_N1_arr.size-1])#Reduce order by 1
            A_N1_tild = np.polyadd(c_n*A_N,s_n*B_N)
            print(A_N1_tild)
            phi_n = -(np.angle(A_N1_tild[A_N1_tild.size-1])+np.angle(B_N1.coef[B_N1.coef.size-1]))
            phi_l.insert(0,phi_n)
            A_N1_tild = np.exp(1j*phi_n)*A_N1_tild
            A_N1 = np.poly1d(A_N1_tild[1:A_N1_tild.size])#Build polynomial A_N1(z), and reduce order by 1
        n = n - 1
        A_N = A_N1
        B_N = B_N1
    print(kappa_l)
    print(phi_l)
    return kappa_l, phi_l
    
def main():
    #PI = np.pi
    #bands = [(0.3*PI, 0.5*PI,1.0)]
    #A_N, N = dF.designFIRFilterKaiser(40, 0.05*PI, bands, plot=True)
    #print("Order: " + str(N))
    N = 2
    A_N = np.poly1d([-0.25,0.25*2.0*np.cos(np.pi/6),-0.25])
    kap_l, ph_l = synthesizeFIRLattice(A_N, N)
    #N = 2
    #A_N = np.poly1d([-0.25,0.25*2.0*np.cos(np.pi/6),-0.25])
    #kap_l, ph_l = synthesizeFIRLattice(A_N, N)
    
                 
if __name__ == '__main__':
    main()
    
        
    
    
    
    
