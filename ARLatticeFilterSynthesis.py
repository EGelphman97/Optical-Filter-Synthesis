"""
Eric Gelphman
UC San Diego Department of Electrical and Computer Engineering
Python file that has methods for the synthesis of autoregressive(AR) IIR digital optical filters
using the lattice architecture using the algorithm described in Section 5.2 of Madsen and Zhao
"""

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
