#Eric Gelphman
#UC San Diego Department of Electrical and Computer Engineering
#March 3, 2020

"""
Python Script that has methods to obtain the transfer function H(z)
Of digital filters as an array of coefficients. These coefs. will then be passed
to latticeFilterSynthesis.py to determine the lattice parameters
Version 1.1.0
"""

import numpy as np
from scipy.signal import kaiser_beta, kaiserord, firwin2, freqz, remez, firls
import matplotlib.pyplot as plt

"""
Function to obtain the normalized angular frequency omega, 0 <= omega <= pi value from a given (continuous-time) wavelength value lamda
Parameters:  lamda0 = longest wavelength in range of interest, in nm
             lamda1 = shortest wavelength in range of interest, in nm
             lamda = wavelength you want to find normalized frequency for, in nm
Return: Normalized frequency omega, 0 <= omega <= pi
"""
def convertToNormalizedFrequency(lamda0, lamda1, lamda):
    c = 3.0E8#The speed of light
    lamda_0 = lamda0*(1.0E-9)
    lamda_1 = lamda1*(1.0E-9)
    lamda_ = lamda*(1.0E-9)
    f1 = c/lamda_1
    f0 = c/lamda_0
    f = c/lamda_
    omega = np.pi*(f-f0)/(f1-f0)
    return omega

"""
Function to perform a boolean operation on a list of stopbands to get corresponding list of passbands
Parameters: stopbands: list of tuples of corner (normalized) frequencies of stopbands
            t_width:   transition bandwidth (normalized frequency)
Return: list of tuples of passbands with corresponding gains
"""
def obtainPassbandsFromStopbands(stopbands, t_width):
    passbands = []
    if stopbands[0][0] == 0.0:
        if len(stopbands) == 1:
            passbands.append((stopbands[0][1]+t_width, np.pi, 1.0))
    else:
        passbands.append((0.0, stopbands[0][0]-t_width, 1.0))
        if stopbands[0][1] != np.pi and len(stopbands) == 1:
            passbands.append((stopbands[0][1]+t_width, np.pi, 1.0))
    if len(stopbands) == 2:
        passbands.append((stopbands[0][1]+t_width,stopbands[1][0]-t_width, 1.0))
        if stopbands[1][1] != np.pi:
            passbands.append((stopbands[1][1]+t_width, np.pi, 1.0))
    return passbands

"""
Function to obtain the bands(pass or stop), in units of normalized frequency, from the center wavelengths (in nm)
Parameters: center_wvlength: ndarray of center wavelengths, longest wavelength/lowest frequency is in position 0 of array
            band_wvlength:    desired width of pass (or stop) band, in nm
            endpoints: endpoints of wavelength of interest, longest wavelength/lowest frequency is in position 0 of array
            typef:        string that indicates type of filter, this parameter is either "Pass" or "Stop"
Return: List of tuples representing band (normalized) frequency intervals
"""
def obtainBandsFromCenterWvlengths(center_wvlengths, endpoints, band_wvlength, typef):
    bands = []
    for ii in range(center_wvlengths.size):
        lambda_lower = center_wvlengths[ii] - (band_wvlength/2.0)
        lambda_upper = center_wvlengths[ii] + (band_wvlength/2.0)
        omega_lower = convertToNormalizedFrequency(endpoints[0], endpoints[1], lambda_upper)
        omega_upper = convertToNormalizedFrequency(endpoints[0], endpoints[1], lambda_lower)
        if typef == "Pass":
            bands.append((omega_lower, omega_upper, 1.0))
        else:
            bands.append((omega_lower, omega_upper, 0.0))
    return bands
            
"""
Function to determine the coefficients of a multiband FIR filter with corresponding passbands (or stopbands) and gains using a Kaiser window

Parameters:   bands: list of 3-tuples (omega_p1, omega_p2, G) were omega_p1 <= omega <= omega_p2 is one band of the multiband filter,
                     G is the gain in the band in linear scale
              t_width: min. transition width for any band in the multiband filter
              num_coefs: Order of filter plus one(number of taps)
              beta: Kaiser parameter beta, depends on attenuation, determined using one of numpy's built-in functions
              filterType:'Pass" or 'Stop' indicating whether it is a bandpass or bandstop filter
              plot: default is True, set to false to NOT plot the frequency response
              
Return:       coefs: ndarray which holds filter coeffients, index refers to power of z^-1
              order: order of filter 
"""
def designFIRFilterKaiser(bands, t_width, num_coefs, beta, filterType, plot=True):
    PI = np.pi
    freq = []#Frequency points
    gain = []#Gain of filter at frequency points in freq
    
    #Build lists freq and gain
    for ii in range(len(bands)):
        if (bands[ii][0]-t_width not in freq) and (bands[ii][0] != 0.0):
            freq.append(bands[ii][0]-t_width)
            if filterType == 'Pass':
                gain.append(0.0)
            else:
                gain.append(1.0)
        freq.append(bands[ii][0])
        gain.append(bands[ii][2])
        freq.append(bands[ii][1])
        gain.append(bands[ii][2])
        if bands[ii][1] <= PI-t_width:
            freq.append(bands[ii][1]+t_width)
            if filterType == 'Pass':
                gain.append(0.0)
            else:
                gain.append(1.0)
    #print("Number of Coefs.: " + str(n_coefs))
    if 0.0 not in freq:
        freq.insert(0,0.0)
        if filterType == 'Pass':
            gain.insert(0,0.0)
        else:
            gain.insert(0,1.0)
    if PI not in freq:
        freq.append(PI)
        if filterType == 'Pass':
                gain.append(0.0)
        else:
                gain.append(1.0)
    
    #Design the filter
    order = num_coefs - 1
    #Numpy's built-in Kaiser function designs Type I or II filters by default, need to check that necessary conditions are met
    antiSym=False#Flag to design a symmetric or anti-symmetric filter
    if order % 2 == 1:#A Type II FIR linear phase system must have a zero at omega = PI
        if gain[len(gain)-1] != 0.0:#If gain of filter is not supposed to be 0 at omega = PI
            num_coefs = num_coefs + 1#Increase order by 1 to make it a Type I
        else:#If gain of filter is supposed to be 0 at omega = PI
            antiSym=True#Design a Type IV      
    coefs = np.flip(firwin2(num_coefs, freq, gain, window=('kaiser',beta), nyq=PI, antisymmetric=antiSym))#Need to flip so indices
    #match in numpy's poly1d class, highest power of z^-1 (see ECE 161B notes) is in index 0 of coefficent array
    
    if plot:
        w, h = freqz(coefs,worN=2048)
        plt.title('Kaiser Window filter frequency response')
        plt.plot(w, 20*np.log10(abs(h)), 'b')
        plt.ylabel('Amplitude [dB]', color='b')
        plt.xlabel('Frequency [rad/sample]')
        plt.show()
    order = num_coefs - 1
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
            weight.append(10.0)
        else:
            weight.append(1.0)

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

    for ii in range(len(gain)):
        if ii % 2 == 0:
            if gain[ii] == 0.0:
                weight.append(10.0)
            else:
                weight.append(1.0)

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
Function to iteratively design a FIR/MA filter using the Kaiser window method
Parameters: N_max: Maximum orderder allowed for filter
     center_freqs: Center (normalized) frequencies of the pass (or stop) bands, lower frequency is in index 0
            bands: List of tuples representing passband intervals, in units of normalized frequency
       filterType: Type of filter, is either "Pass" or "Stop"
Return: A_N: Coefficent array for filter, coef. of Z^-N term is in position 0 in array
          N: Filter order
    t_width: Transition band width, in normalized frequency
"""
def designKaiserIter(N_max, bands, atten, filterType):
    PI = np.pi
    max_twidth = 0.18*PI#What max. t_width is for single-band filter
    if len(bands) == 2:
        DSP_band_gap = bands[1][0] - bands[0][1]#Define max. transition width in terms of gap between passbands (or stopbands
        max_twidth = DSP_band_gap/4.0
    t_width = max_twidth/5.0
    num_coefs, beta = kaiserord(atten, t_width)#Determine parameter beta of Kaiser window
    N = num_coefs - 1
    while N > N_max:
        if t_width < max_twidth:
            t_width = 1.1*t_width
            num_coefs, beta = kaiserord(atten, t_width)#Determine parameter beta of Kaiser window
            N = num_coefs - 1
        else:
            atten = atten - 2.5
            num_coefs, beta = kaiserord(atten, t_width)#Determine parameter beta of Kaiser window
            N = num_coefs - 1
    A_N, N = designFIRFilterKaiser(bands, t_width, num_coefs, beta, filterType, plot=False)
    return A_N, N, t_width

"""      
def main():
    PI = np.pi
    stopbands = [(0.25*PI, 0.35*PI, 1.0), (0.55*PI, 0.65*PI, 1.0)]
    coefs, n_coefs = designFIRFilterKaiser(50, stopbands, 0.05*PI, 'Pass')
    

if __name__ == '__main__':
    main()
"""

