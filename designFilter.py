#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#February 9, 2020

"""
Python Script that has methods to obtain the transfer function H(z)
Of digital filters as an array of coefficients. These coefs. will then be passed
to latticeFilterSynthesis.py to determine the lattice parameters
Version 1.0.3
"""

import numpy as np
from scipy.signal import kaiserord, firwin2, freqz, remez, firls
import matplotlib.pyplot as plt

"""
Function to determine the coefficients of a multiband FIR filter with corresponding passbands and gains using a Kaiser window

Parameters:   ripple: max. deviation in dB of the realized filter's frequnecy response from the ideal frequnecy response
              t_width: min. transition width for any band in the multiband filter. This needs to be such that 
              bands: list of 3-tuples (omega_p1, omega_p2, G) were omega_p1 <= omega <= omega_p2 is one passband of the multiband filter, G is the passband gain in linear scale
              plot: default is True, set to false to NOT plot the frequency response
              
Return:       coefs: ndarray which holds filter coeffients, index refers to power of z^-1
              order: order of filter 
"""
def designFIRFilterKaiser(ripple, t_width, bands, plot=True):
    PI = np.pi
    #t_width = determine t_width()
    n_coefs, beta = kaiserord(ripple, t_width)#Determine order and paramter beta of Kaiser window
    freq = []#Frequency points
    gain = []#Gain of filter at frequency points in freq
    
    #Build lists freq and gain
    for ii in range(len(bands)):
        if (bands[ii][0]-t_width not in freq) and (bands[ii][0] != 0.0):
            freq.append(bands[ii][0]-t_width)
            gain.append(0.0)
        freq.append(bands[ii][0])
        gain.append(bands[ii][2])
        freq.append(bands[ii][1])
        gain.append(bands[ii][2])
        if bands[ii][1] <= PI-t_width:
            freq.append(bands[ii][1]+t_width)
            gain.append(0.0)
    #print("Number of Coefs.: " + str(n_coefs))
    if 0.0 not in freq:
        freq.insert(0,0.0)
        gain.insert(0,0.0)
    if PI not in freq:
        freq.append(PI)
        gain.append(0.0)
    #print(np.array(freq))
    #print(np.array(gain))
    
    #Design the filter
    coefs = firwin2(n_coefs, freq, gain, window=('kaiser',beta), nyq=PI)
    
    if plot:
        w, h = freqz(coefs)
        plt.title('Kaiser Window filter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    order = n_coefs - 1
    return coefs, order

"""
Function to determine the coefficients of a multiband equiripple FIR filter with corresponding passbands and gains using the Parks-McClellan algorithm,
implemented using scipy.signals's remez()

Parameters:   order: Order of filter
              bands: list of 3-tuples (omega_p1, omega_p2, G) were omega_p1 <= omega <= omega_p2 is one passband of the multiband filter, G is the passband gain in linear scale
              plot: default is True, set to false to NOT plot the frequency response
              
Return:       coefs: ndarray which holds filter coeffients, index refers to power of z^-1
              order: order of filter 
"""
def designFIRFilterPMcC(order, t_width, bands, plot=True):
    PI = np.pi
    freq = []#Frequency points
    gain = []#Gain of filter for bands in freq, size of this list should be exactly half the size of freq
    
    #Build lists freq and gain
    for ii in range(len(bands)):
        if bands[ii][0] != 0.0:
            freq.append(bands[ii][0]-t_width)
            gain.append(0.0)
        freq.append(bands[ii][0])
        gain.append(bands[ii][2])
        freq.append(bands[ii][1])
        if bands[ii][1] != PI:
            freq.append(bands[ii][1]+t_width)
    if 0.0 not in freq:
        freq.insert(0,0.0)
    if PI not in freq:
        freq.append(PI)
        gain.append(0.0)

    weight = []#Weighting function to plug into remez exchnage algorithm
    for ii in range(len(gain)):
        if gain[ii] == 0.0:
            weight.append(4.0)
        else:
            weight.append(2.0)
    print(gain)
    print(weight)

    #Design the filter
    coefs = remez(order+1, freq, gain, weight, fs=2.0*PI)
    if plot:
        w, h = freqz(coefs)
        plt.title('Equiripple filter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    order = coefs.size - 1
    return coefs, order

"""
Function to determine the coefficients of a multiband FIR filter that is optimal in the least-squares sense. The filter coefficients are determined
using scipy.signal's firls()

Parameters:   order: Order of filter. This must be even, so numtaps = order + 1 is odd
              bands: list of 3-tuples (omega_p1, omega_p2, G) were omega_p1 <= omega <= omega_p2 is one passband of the multiband filter, G is the passband gain in linear scale
              plot: default is True, set to false to NOT plot the frequency response
              
Return:       coefs: ndarray which holds filter coeffients, index refers to power of z^-1
              order: order of filter 
"""
def designFIRFilterLS(order, t_width, bands, plot=True):
    PI = np.pi
    freq = []#Frequency points
    gain = []#Gain of filter for bands in freq, size of this list should be exactly the same size as freq
    weight = []#Weighting function chosen to minize passband loss
    
    #Build lists freq and gain
    for ii in range(len(bands)):
        if bands[ii][0] != 0.0:
            freq.append(bands[ii][0]-t_width)
            gain.append(0.0)
        freq.append(bands[ii][0])
        gain.append(bands[ii][2])
        freq.append(bands[ii][1])
        gain.append(bands[ii][2])
        if bands[ii][1] != PI:
            freq.append(bands[ii][1]+t_width)
            gain.append(0.0)
    if 0.0 not in freq:
        freq.insert(0,0.0)
        gain.append(0.0)
    if PI not in freq:
        freq.append(PI)
        gain.append(0.0)
    print(np.array(freq))
    print(np.array(gain))

    for ii in range(len(gain)):
        if ii % 2 == 0:
            if gain[ii] == 0.0:
                weight.append(5.0)
            else:
                weight.append(2.0)

    #Design the filter
    coefs = firls(order+1, freq, gain, weight, fs=2.0*PI)
    if plot:
        w, h = freqz(coefs)
        plt.title('Least-Squares filter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    order = coefs.size - 1
    return coefs, order
    

"""
def main():
    PI = np.pi
    bands = [(0.3*PI, 0.4*PI, 1.0), (0.6*PI, 0.75*PI,0.75)]
    coefs, n_coefs = designFIRFilterPMcC(50, 0.05*PI, bands)
    print(n_coefs)
    

if __name__ == '__main__':
    main()
"""
