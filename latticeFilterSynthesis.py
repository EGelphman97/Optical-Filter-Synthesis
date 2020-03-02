#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#March 1, 2020

#Implementation of Madsen and Zhao's Optical MA/FIR and AR/IIR Lattice Filter Design Algorithm
#Version 1.0.11

import designFilter as dF
import numpy as np
from scipy import integrate
from scipy.signal import freqz
import matplotlib.pyplot as plt
import synthesisDriver as sD

"""
Function to perform spectral factorization for the roots of B(z)BR(z)(MA case) or B(z)B_star(z)(AR case)
Parameters: roots: array of roots of the polynomial B(z)BR(z) or B(z)B_star(z) which is of degree 2*order
            order: degree of B(z)
Return: array of roots of B(z)
"""
def spectralFactorization(roots, order):
    b_roots = []
    for ii in range(roots.size):
        #Assign min. phase roots to B(z)
        if np.abs(roots[ii]) < 1.0:
            b_roots.append(roots[ii])
            if len(b_roots) == order:
                break
    #print(len(b_roots))
    if len(b_roots) < order:#B(z)BR(z) has n roots, n is even, B(z) has n/2 roots, make sure B(z) has this many roots
        num_remaining = order - len(b_roots)
        #identify real roots, even if they're not inside unit circle and add them to b_roots
        for ii in range(roots.size):
            if np.imag(roots[ii]) == 0.0 and b_roots.count(roots[ii]) == 0:
                b_roots.append(roots[ii])
                num_remaining = num_remaining - 1
                if num_remaining == 0:
                    break
        #Just pick from roots until b_roots is of the correct size
        if num_remaining > 0:
            for ii in range(roots.size):
                if b_roots.count(roots[ii]) == 0:
                    b_roots.append(roots[ii])
                    num_remaining = num_remaining - 1
                    if num_remaining == 0:
                        break
    return np.array(b_roots)

"""
Function to find the polynomial B_N(z) which is needed to find the 2x2 MA transfer function of the optical filter
Parameters: A: coefficient array of polynomial in z^-1 of degree N that also is part of the 2x2 transfer function
               Note that z^-N term should occupy position 0 in coefficient array, with other terms occupying
               the indices in descending powers of z^-1. E.g.: z^-(N-1) term should occupy position 1 in coefficient
               array, z^-(N-2) term should occupy position2, ... , constant term occupies position A.coef.size-1 in array
Return:     B: coefficient array of polynomial in z^-1 of degree N B_N(z)
"""
def findBPolyMA(A):
    phase_arr = np.zeros(A.size)
    phase_arr[0] = 1.0#only coef. of Z^(-N) term is nonzero for phase term
    bbr_coef = -1.0*np.polymul(A,np.flip(A))
    bbr = np.poly1d(np.polyadd(bbr_coef,phase_arr))#Polynomial B_N(z)B_NR(z)
    roots = bbr.roots#Find roots of B_N(z)B_NR(z)
    """
    for ii in range(roots.size):
        print(f"{np.abs(roots[ii])}e^{np.angle(roots[ii])}")
    """
    #Plot the zeros
    theta = np.linspace(-np.pi, np.pi, 201)
    plt.plot(np.sin(theta), np.cos(theta), color = 'gray', linewidth=0.2)
    plt.plot(np.real(roots),np.imag(roots), 'Xb', label='Zeros')
    plt.title("Pole-Zero Plot")
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.grid()
    plt.show()

    #Spectral factorization
    b_roots = spectralFactorization(roots, bbr.order/2)
    B_tild = np.poly1d(b_roots, True)#Construct polynomial from its roots
    alpha = np.sqrt((-A[A.size-1]*A[0])/(B_tild.coef[B_tild.coef.size-1]*B_tild.coef[0]))#Scale factor
    B = alpha*B_tild#Build B_N(z) by scaling B_tild(z) by alpha
    
    return np.flip(B)#Hihest degree coef. is actually in lowest degree coef. before flip, so need flip

"""
Function to find the polynomial B_N(z) which is needed to find the 2x2 AR transfer function of the optical filter
Parameters: A: polynomial(poly1d) in z^-1 of degree N that also is part of the 2x2 transfer function
               Note that z^-N term should occupy position 0 in coefficient array, with other terms occupying
               the indices in descending powers of z^-1. E.g.: z^-(N-1) term should occupy position 1 in coefficient
               array, z^-(N-2) term should occupy position2, ... , constant term occupies position A.coef.size-1 in array
            sigma: Gain parameter, see pg. 248 of Madsden and Zhao 
Return:     Polynomial(poly1d) in z^-1 of degree N B_N(z)
"""
def findBPolyAR(A, sigma):
    A_star_coef = np.conj(np.flip(A.coef))
    #Form polynomial B(z)B_star(conj(z^-1)), then find its roots
    bbstar_coef = np.polymul(A.coef,A_star_coef)
    bbstar_coef[int(bbstar_coef.size/2)] = bbstar_coef[int(bbstar_coef.size/2)] + sigma
    bbstar = np.poly1d(bbstar_coef)
    roots = bbstar.roots
    
    b_roots = spectralFactorization(roots, bbstar.order/2)
    B_prime = np.poly1d(b_roots, True)#Construct polynomial from its roots
    B_primep = np.poly1d(np.flip(B_prime.coef))#Want highest degree term to occupy position 0 in coef. array
    alpha = np.sqrt(A.coef[0]/B_primep.coef[0])
    B = alpha*B_primep
    return np.poly1d(B)

"""
Function to compute the cross-over length and other lengths needed for layout
Parameters: kappa: power coupling coefficient(Prof. Mookherjea's kappa^2, Madsden and Zhao's eta)
            lc: length, in um
            lend: length, in um
"""
def calcLengths(kappa, lc, lend):
    psy = (2.0/np.pi)*np.arcsin(np.sqrt(kappa))
    L_c = (lc+lend)/psy
    return L_c
    

"""
Function to synthesize an FIR optical lattice filter using the algorithm outlined in Section 4.5 of Madsden and Zhao
Paramters: A_N: Coefficient array of polynomial in z^-1 of degree N that is part of the 2x2 transfer function
           N:   Filter order
Return: kappalcs: List of power coupling coefficients kappa_n's, Lc's and lc + lend for each stage, as well as c_n and s_n list index is same as n
                  Format is: kappalcs[n] = (kappa_n, L_c_n, lc + lend of stage n, c_n, s_n)
        phi_l:   List of phase terms phi_n list index is n-1
        B_N: Polynomial (in z^-1) B_N(z)
"""
def synthesizeFIRLattice(A_N, N):
    gamma = 1.0
    B_N = findBPolyMA(A_N)
    B_N_OG = B_N
    print(np.poly1d(B_N))
    phi_l = []#List of phi_n's
    kappalcs = []#List of kappas, Lc's. This is what we want to return
    n = N
    while n >= 0:
        #print(A_N)
        
        #Calculate kappa
        beta = np.absolute(B_N[0]/A_N[0])**2
        kappa = beta/(1.0 + beta)
        L_c = calcLengths(kappa, 2.0, 2.0)#Find lengths of structures we need for layout, convert wavelengths to micrometers
        c_n = np.sqrt(1.0-kappa)
        s_n = np.sqrt(kappa)
        kappalcs.insert(0,(kappa, L_c))
        if n > 0:
            B_N1 = np.polyadd(-s_n*A_N,c_n*B_N)#Step-down recursion relation for B polynomial of stage N-1, this is an ndArray
            B_N1 = B_N1[1:B_N1.size]
            #B_N1 = np.poly1d(B_N1_arr[1:B_N1_arr.size])#Reduce order by 1
            #Shouldn't have complex coefs.
            for ii in range(B_N1.size):
                if np.imag(B_N1[ii]) < 2.0E-16:
                    B_N1[ii] = np.real(B_N1[ii])
            A_N1_tild = np.polyadd(c_n*A_N,s_n*B_N)
            phi_n = -(np.angle(A_N1_tild[0]) + np.angle(B_N1[0]))
            phi_l.insert(0,phi_n)
            A_N1_tild = np.exp(1j*phi_n)*A_N1_tild
            A_N1 = A_N1_tild[0:A_N1_tild.size-1]#Build polynomial A_N1(z), and reduce order by 1 by eliminating the constant term(multiplying by z)
            #Shouldn't have complex coefs.
            for ii in range(A_N1.size):
                if np.imag(A_N1[ii]) < 2.0E-16:
                    A_N1[ii] = np.real(A_N1[ii])
        n = n - 1
        A_N = A_N1
        B_N = B_N1
    return kappalcs, phi_l, B_N_OG

"""
Function to get the 2x2 FIR transfer function of the filter given the power coupling ratio kappa for each stage.
This is the "inverse" operation of the FIR synthesis algorithm
Parameters: kappas: array of kappa_n's 0 <= n <= N    N = filter order
            phis:   array of phi_n's 1 <= n <= N
Return: Polynomials A_N(z), B_N(z), A_N_R(z), B_N_R(z) that form 2x2 transfer function of filter
"""
def inverseFIRSynthesis(kappas, phis):
    A_N1 = np.array([np.sqrt(1.0-kappas[0])])#A_0(z) = c_0
    B_N1 = np.array([np.sqrt(kappas[0])])#B_0(z) = s_0
    for ii in range(1,kappas.size):
        c_n = np.sqrt(1.0-kappas[ii])
        s_n = np.sqrt(kappas[ii])
        #Form A_N(z)
        A1arr = np.exp(-1j*phis[ii-1])*np.pad(A_N1, (0,1), 'constant', constant_values=(0,0))#Increase degree of each term by left-shifting array and filling with 0
        A_N = np.polyadd(c_n*A1arr,np.multiply(-1.0*s_n,B_N1))
        #Form B_N(z)
        B_N = np.polyadd(s_n*A1arr,np.multiply(c_n,B_N1))
        #Shouldn't have complex coefs.
        for ii in range(A_N.size):
            if np.imag(A_N[ii]) < 2.0E-16:
                A_N[ii] = np.real(A_N[ii])
            if np.imag(B_N[ii]) < 2.0E-16:
                B_N[ii] = np.real(B_N[ii])

        A_N1 = A_N
        B_N1 = B_N

    A_N_R = np.conj(np.flip(A_N1))
    B_N_R = np.conj(np.flip(B_N1))
    return A_N1, B_N1, A_N_R, B_N_R

"""
Function to implement the AR lattice synthesis algorithm as outlined in Section 5.2 of Madsden and Zhao
Parameters: A_N: Polynomial(poly1d) in z^-1 of degree N that is the filter transfer function
              N: Order of filter/degree of A_N(z)
      big_gamma: Overall gain parameter
   little_gamma: Uniform loss coefficient
Return: List of kappa_n's and phi_n's for each stage. kappa_0 exists but indexing for phi_n starts at 1
"""
def synthesizeARLattice(A_N, N, big_gamma, little_gamma):
    sigma_N = 0.9*9.025E-03
    B_N = findBPolyAR(A_N, sigma_N)
    phis = []#List of phi_n's
    kappas = []#List of kappas
    n = N
    while n >= 0:
        #print(A_N)
        #print(B_N)
        c_n = B_N.coef[B_N.coef.size-1]
        kappa = 1.0 - (c_n**2)
        kappas.insert(0,kappa)
        if n > 0:
            A_coef = A_N.coef
            B_coef = B_N.coef
            A_N1_arr = (1.0/kappa)*np.polyadd(A_coef, -c_n*B_coef)
            A_N1 = np.poly1d(A_N1_arr[1:A_N1_arr.size])#Reduce order by 1
            B_N1_tild = (1.0/(little_gamma*kappa))*np.polyadd(c_n*A_coef,-1.0*B_coef)#This is an ndarray
            phi_n = np.angle(A_N1.coef[0]) - np.angle(B_N1_tild[0])
            phis.insert(0,phi_n)
            B_N1 = np.poly1d(np.exp(1j*phi_n)*B_N1_tild[0:B_N1_tild.size-1])#Reduce order by 1 by eliminating the constant term
            for ii in range(B_N1.coef.size):
                if np.imag(B_N1.coef[ii]) < 2.0E-16:
                    B_N1.coef[ii] = np.real(B_N1.coef[ii])
            
        n = n - 1
        A_N = A_N1
        B_N = B_N1
    return kappas, phis

"""
Function to do the inverse operation of AR lattice filter synthesis, that is, given kappas and phis, obtain A_N(z)
Parameters: kappas: array of power coupling ratios kappa_n for each stage, 0 <= n <= N
              phis: list of phase terms for each stage 1 <= n <= N
      little_gamma: overall loss coefficient gamma
Return: Polynomials A_N(z), B_N(z)
"""
def inverseARSynthesis(kappas, phis, little_gamma):
    A_N1 = np.poly1d([1.0])#A_0(z) = 1
    B_N1 = np.poly1d([np.sqrt(1.0 - kappas[0])])#B_0(z) = c_0
    for ii in range(1,kappas.size):
        c_n = np.sqrt(1.0-kappas[ii])
        #Form A_N(z)
        B1arr = np.exp(-1j*phis[ii-1])*little_gamma*np.pad(B_N1.coef, (0,1), 'constant', constant_values=(0,0))#Increase degree of each term by left-shifting array and filling with 0
        A_N = np.poly1d(np.polyadd(A_N1.coef,-1.0*c_n*B1arr))
        #Form B_N(z)
        B_N = np.poly1d(np.polyadd(c_n*A_N1.coef,-1.0*B1arr))
        #Shouldn't have complex coefs.
        for ii in range(A_N.coef.size):
            if np.imag(A_N.coef[ii]) < 2.0E-16:
                A_N.coef[ii] = np.real(A_N.coef[ii])
            if np.imag(B_N.coef[ii]) < 2.0E-16:
                B_N.coef[ii] = np.real(B_N.coef[ii])

        A_N1 = A_N
        B_N1 = B_N
        
    return A_N1, B_N1

   
  
"""   
def main():
    PI = np.pi
    lamda_0 = 1565#nanometers
    lamda_1 = 1520#nanometers
    #bands = [(0.3*PI,0.45*PI,1.0)]
    #bands = [(0.3*PI, 0.4*PI, 1.0), (0.7*PI, 0.8*PI,1.0)]
    #A_N, order = dF.designFIRFilterKaiser(50, bands, 0.05*PI, plot=True)
    #A_N = np.flip(A_N)#So indices match in numpy's poly1d  class
    center_l1 = 1532
    center_l2 = 1550
    omega_c1 = sD.convertToNormalizedFrequency(lamda_0, lamda_1, center_l2)
    omega_c2 = sD.convertToNormalizedFrequency(lamda_0, lamda_1, center_l1)
    center_freqs = np.array([omega_c1, omega_c2])
    print(center_freqs)
    passbandwidth = 0.1*np.pi
    
    
    
    print(bandwidth)
    
    bands = dF.obtainBandsFromCenterFreqs(center_freqs, bandwidth, "Pass")
    A_N, N = dF.designKaiserIter(20, center_freqs, bands)
    #Need highest degree term to be negative for synthesis algorithm to work, see Madsden and Zhao Section 4.5
    if A_N[0] > 0.0:
        A_N = -1.0*A_N
    A_z = np.poly1d(A_N)
    print(A_z)
    
    
    L_U = sD.calcUnitDelayLength(lamda_1, lamda_0, 4.0)
    insertionLoss = sD.calcInsertionLoss(A_z, bands)
    kappalcs, phis = synthesizeFIRLattice(A_N, N, lamda_0, lamda_1)
    sD.writeLayoutParametersToFile(kappalcs, phis, L_U, 25.0, "layoutParameters.txt", insertionLoss, N, "Kaiser")
    kappas = np.zeros(len(kappalcs))
    #print(np.array(phis))
    for ii in range(len(kappalcs)):
        kappas[ii] = kappalcs[ii][0]
    A_N, B_N, A_N_R, B_N_R = inverseFIRSynthesis(kappas, phis)
    print(np.poly1d(A_N))
    w, h = freqz(A_N)
    c = 3.0E8
    f1 = c/lamda_1
    f0 = c/lamda_0
    wvlength = np.zeros(w.size)
    for ii in range(w.size):
        denom = (1.0/PI)*(w[ii]*(f1-f0)) + f0
        wvlength[ii] = c/denom  
    plt.title('MA filter frequency response')
    plt.plot(np.flip(wvlength), 20*np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Wavelength [um]')
    plt.show()
                 
if __name__ == '__main__':
    main()
"""    
        
    
    
    
    
