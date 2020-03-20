"""
Eric Gelphman
UC San Diego Department of Electrical and Computer Engineering
Last Updated March 20, 2020 Version 1.1.2

Implementation of Madsen and Zhao's Optical MA/FIR Lattice Filter Design Algorithm

Required Packages:
-numpy
-scipy.signal:freqz
-matplotlib.pyplot
"""

#import designFilter as dF
import numpy as np
from scipy import integrate
from scipy.signal import freqz
import matplotlib.pyplot as plt


def spectralFactorization(roots, order):
    """
    Function to perform spectral factorization for the roots of B(z)BR(z)
    
    Parameters: roots: array of roots of the polynomial B(z)BR(z) or B(z)B_star(z) which
                       is of degree 2*order
                order: degree of B(z)
                
    Return: array of roots of B(z)
    """
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
        #Just pick from roots until b_roots is of the correct size
        if num_remaining > 0:
            for ii in range(roots.size):
                if b_roots.count(roots[ii]) == 0:
                    b_roots.append(roots[ii])
                    num_remaining = num_remaining - 1
                    if num_remaining == 0:
                        break
    return np.array(b_roots)


def findBPolyMA(A, plot=True):
    """
    Function to find the polynomial B_N(z) which is needed to find the 2x2 MA transfer function
    of the optical filter

    Parameters: A: coefficient array of polynomial in z^-1 of degree N that also is part of the
                   2x2 transfer function Note that z^-N term should occupy position 0 in coeffi-
                   cient array, with other terms occupying the indices in descending powers of
                   z^-1. E.g.: z^-(N-1) term should occupy position 1 in coefficient array,
                   z^-(N-2) term should occupy position2, ... , constant term occupies position
                   A.coef.size-1 in array
             plot: Boolean, default is True, that indicates whether or not the zeros of B_N(z) should be
                   plotted
         
    Return:     B: coefficient array of polynomial in z^-1 of degree N B_N(z)
    """
    phase_arr = np.zeros(A.size)
    phase_arr[0] = 1.0#only coef. of Z^(-N) term is nonzero for phase term
    bbr_coef = -1.0*np.polymul(A,np.flip(A))
    bbr = np.poly1d(np.polyadd(bbr_coef,phase_arr))#Polynomial B_N(z)B_NR(z)
    roots = bbr.roots#Find roots of B_N(z)B_NR(z)
    #Plot the zeros, if desired
    if plot:
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
    #print(B_tild)
    #print(B_tild.coef[B_tild.coef.size-1]*B_tild.coef[0])
    alpha = np.sqrt((-A[A.size-1]*A[0])/(B_tild.coef[B_tild.coef.size-1]*B_tild.coef[0]))#Scale factor
    B = alpha*B_tild#Build B_N(z) by scaling B_tild(z) by alpha
    return np.flip(B)#Hihest degree coef. is actually in lowest degree coef. before flip, so need flip


def calcLengths(kappa, lc, lend):
    """
    Function to compute the cross-over length and other lengths needed for layout

    Parameters: kappa: power coupling coefficient(Prof. Mookherjea's kappa^2)
                   lc: length, in um
                 lend: length, in um

    Return: L_c, in um
    """
    psy = (2.0/np.pi)*np.arcsin(np.sqrt(kappa))
    L_c = (lc+lend)/psy
    return L_c
    

def synthesizeFIRLattice(A_N, N, gamma):
    """
    Function to synthesize an FIR optical lattice filter using the algorithm outlined in Section 4.5 of Madsen
    and Zhao
    
    Paramters: A_N: Coefficient array of polynomial in z^-1 of degree N that is part of the 2x2 transfer
                    function
                 N:   Filter order
             gamma: Loss Coefficient per stage
             
    Return: kappalcs: List of power coupling coefficients kappa_n's, Lc's and lc + lend for each stage, as
                      well as c_n and s_n list index is same as n Format is:
                      kappalcs[n] = (kappa_n, L_c_n, lc + lend of stage n, c_n, s_n)
               phi_l:   List of phase terms phi_n list index is n-1
                 B_N: Polynomial (in z^-1) B_N(z)
    """
    B_N = findBPolyMA(A_N, plot=False)
    B_N_OG = B_N
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
            A_N1_tild = (1.0/gamma)*np.exp(1j*phi_n)*A_N1_tild
            A_N1 = A_N1_tild[0:A_N1_tild.size-1]#Build polynomial A_N1(z), and reduce order by 1 by eliminating the constant term(multiplying by z)
            #Shouldn't have complex coefs.
            for ii in range(A_N1.size):
                if np.imag(A_N1[ii]) < 2.0E-16:
                    A_N1[ii] = np.real(A_N1[ii])
        n = n - 1
        A_N = A_N1
        B_N = B_N1
    return kappalcs, phi_l, B_N_OG


def inverseFIRSynthesis(kappas, phis, gamma):
    """
    Function to get the 2x2 FIR transfer function of the filter given the power coupling ratio kappa for
    each stage. This is the "inverse" operation of the FIR synthesis algorithm

    Parameters: kappas: Array of kappa_n's 0 <= n <= N    N = filter order
                  phis: Array of phi_n's 1 <= n <= N
                 gamma: Loss coefficient per stage
                  
            
    Return: Polynomials A_N(z), B_N(z), A_N_R(z), B_N_R(z) that form 2x2 transfer function of filter
    """
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
def main():
    PI = np.pi
    plt.title('MA filter frequency response')
    plt.plot(np.flip(wvlength), 20*np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Wavelength [um]')
    plt.show()
                 
if __name__ == '__main__':
    main()
"""    
        
    
    
    
    
