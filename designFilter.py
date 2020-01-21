#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#January 21, 2020

"""
Python Script that has methods to obtain the transfer function H(z)
Of digital filters as an array of coefficients. These coefs. will then be passed
to latticeFilterSynthesis.py to determine the lattice parameters
Version 1.0.0
"""

import numpy as np
from scipy import special as ss

#Function that returns the mimimum of two doubles/floats
def minimum(arg1, arg2):
    if arg1 < arg2:
        return arg1
    elif arg2 < arg1:
        return arg2
    else:#Else, both arguments are equal
        return arg1

"""
Function to determine the low-pass FIR filter coefficients using a Kaiser window
Parameters:   ws: Stopband endge, 0 <= ws <= pi
              wp: Passband edge, 0 <= wp <= pi
              delt_p: Passband attenuation tolerance
              delt_s: Stopband attenuation tolerance
Return:       Array which holds filter coeffients, index refers to power of z^-1
"""
def kaiserWindow(ws, wp, delt_p, delt_s):
    #First, determine transition width and attenuation
    delta_w = ws - wp
    A = -20*np.log10(minimum(delt_p, delt_s))
    print("A = " + str(A))

    #Determine the filter order, M
    M = np.ceil((A-8.0)/(2.285*delta_w))
    print(M)
    #Build coef. array
    coefs = np.zeros(int(M+1), dtype=complex)
    alpha = M/2
    w_c = (wp + ws)/2#Center frequency
    
    #Determine the parameter beta
    beta = 0.0
    if 21.0 <= A and A <= 50.0:
        beta = 0.5842*((A-21.0)**0.41) + 0.07886*(A-21.0)
    elif A > 50.0:
        beta = 0.1102*(A-8.7)
    #If A < 21.0, beta = 0.0 ======> Rectangular window
   
    for ii in range(int(M+1)):
        sinc_n = np.sin((w_c*(ii-alpha))/(np.pi*(ii-alpha)))#Ideal LPF impulse response
        if A >= 21.0:#Kaiser window
            kaiser_window = ss.i0(beta*np.sqrt(1.0 - ((ii-alpha)/alpha)**2))/ss.i0(beta)
            print(kaiser_window)
            coefs[ii] = sinc_n*kaiser_window
        else:#Rectangular window
            coefs[ii] = sinc_n
    return coefs

"""

def main():
    print(kaiserWindow(0.65*np.pi, 0.4*np.pi, 0.1, 0.1))

if __name__ == '__main__':
    main()
"""
