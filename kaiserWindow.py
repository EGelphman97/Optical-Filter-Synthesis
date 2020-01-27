#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#January 24, 2020

import numpy as np
from scipy.special as ss

#Function that returns the mimimum of two doubles/floats
def minimum(arg1, arg2):
    if arg1 < arg2:
        return arg1
    elif arg2 < arg1:
        return arg2
    else:#Else, both arguments are equal
        return arg1


"""
Function to determine the parameters M(filter order) and beta for a Kaiser window to then obtain the coefficients of a multiband filter
Parameters:   specs: List of tuples (G_i,omega_pi, omega_si) i = 1,...,N N = number of bands. Filter magnitude response is equal
                     to G_i for omega_pi-1 <= omega <= omega_pi omega_0 = 0. omega_pi's represent passband edges of the
                     componet LPF's
                     G_i's are in linear scale, not dB
              delta_min: Minimim passband or stopband attentuation tolerance
Return:       Array which holds filter coeffients, index refers to power of z^-1
"""

def kaiserWindow(specs, delta_min):
    #First, determine transition width and attenuation
    delta_omegas = np.zeros(len(specs))
    #Determine the minimum separation delta_omega between the stopbands and passbands
    for ii in range(len(specs)):
        delta_omegas[ii] = specs[ii][2]-specs[ii][1]
    delta_omega_min = np.amin(delta_omegas)
    A = -20*np.log10(delta_min)

    #Determine the filter order, M
    M = np.ceil((A-8.0)/(2.285*delta_omega_min))
    
    #Build array of center frequencies
    omega_c = np.zeros(len(specs))#Build array of center frequencies
    for ii in range(len(specs)):
        omega_c[ii] = (specs[ii][1]+specs[ii][2])/2.0
    
    #Determine the parameter beta
    beta = 0.0
    if 21.0 <= A and A <= 50.0:
        beta = 0.5842*((A-21.0)**0.4) + 0.07886*(A-21.0)
    elif A > 50.0:
        beta = 0.1102*(A-8.7)
    #if A < 21.0, beta = 0.0 ======> Rectangular window

    #Build coef. array
    coefs = np.zeros(int(M+1), dtype=complex)
    alpha = M/2
    sinc_n = 0.0
    print(M)
    for ii in range(int(M+1)):
        #Evaluate sum of sinc functions that depend on the corner frequencies in omega_c
        for jj in range(len(specs)):
            G_jj = specs[jj][0]
            G_jj1 = 0.0
            if jj < len(specs) - 1:
                G_jj1 = specs[jj+1][0]
            denom = ii-alpha
            if denom == 0.0:
                sinc_n = sinc_n + ((G_jj-G_jj1)*(omega_c[jj]/np.pi))#Ideal LPF impulse response, need L'Hospital's Here
            else:
                sinc_n = sinc_n + (G_jj-G_jj1)*(np.sin(omega_c[jj]*denom)/(np.pi*denom))#Ideal LPF impulse response
        if A >= 21.0:#Kaiser window
            kaiser_window = ss.i0(beta*np.sqrt(1.0 - ((ii-alpha)/alpha)**2))/ss.i0(beta)
            coefs[ii] = sinc_n*kaiser_window
        else:#Rectangular window
            coefs[ii] = sinc_n
    return coefs
